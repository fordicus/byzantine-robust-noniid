_FLAG_DATAFY_CLIENT_MESSAGES_  	= False
_FLAG_DATAFY_BYZANTINE_NOISES_ 	= False
_GLOBAL_DEBUGGER_COUNTER_		= 1			# @_FLAG_DATAFY_BYZANTINE_NOISES_

import argparse
import logging
import numpy as np
import torch
import gc
from scipy.io import savemat  				# @_FLAG_DATAFY_CLIENT_MESSAGES_
from typing import Union, Callable, Any
from collections import defaultdict

from .worker import TorchWorker
from .server import TorchServer
from .worker import ByzantineWorker

from codes.aggregator.clipping import ByRoAdapter

import tracemalloc	# monitor memory footage

FLAG_MONITOR_CPU_MEMORY = False


#==========================================================================================

def format_tensor_to_print(
	tensor: torch.Tensor, precision: int = 2, sci_mode: bool = True) -> str:
	
	r"""
		Format a PyTorch tensor as a string with specified precision and scientific notation

		Args:
		
			tensor (torch.Tensor): The tensor to format
			precision (int): Number of decimal places for each value
			sci_mode (bool): Whether to use scientific notation

		Returns:
		
			str: A formatted string representation of the tensor
	"""
	
	if sci_mode:
		
		formatted_values = [f"{v:.{precision}e}" for v in tensor.flatten().tolist()]
	else:
		
		formatted_values = [f"{v:.{precision}f}" for v in tensor.flatten().tolist()]
	
	return f"[{', '.join(formatted_values)}]"

#==========================================================================================
# 2025-01-20 (dynamic byzantine behavior)
#..................................................................................
# TorchWorker.byzantine_behavior = {0: BF, 1: LF, 2: mimic, 3: IPM, 4: ALIE}
#------------------------------------------------------------------------------------------

class IntraclientByzantine():
	
	def __init__(self, args):	# [utils.py] main(): trainer.prepareByzantines()
		
		from scipy.stats import norm as scipyNorm
		
		#..................................................................................
		# Honest-omniscience
		#..................................................................................
		# The `self.good_grads` always knows the honest clients' mini-batch gradients
		# because our dynamic Byzantine behavior is implemented under the setting:
		#
		#   ((self.args.client_delta > 0.0) and (self.args.f == 0)).
		#
		# In other words, every client is seen honest in our implementation while they
		# probabilistically return Byzantines instead. This behavior is simulated here
		# just before the aggregation happens. Before each client decides whether they
		# behave as honest or Byzantine, the honest mini-batch stochastic gradients are
		# safely memorized into `self.good_grads`. As such saved in advance, the reference
		# mechanism of some Byzantine behaviors (to the honest) is data-safe; it does not
		# happen that any Byzantine somehow mistakenly utilizes the information of some
		# other Byzantines. Always access `self.good_grads` to define Byzantines based
		# on honests, instead of polling other honest workers' `get_gradient` method.
		#..................................................................................
		
		self.good_grads = None
		self.good_ranks = None
		
		self.honest_avg = 0.0
		
		self.args = args
		
		#..................................................................................
		# Regardless of individual client's Byzantine behavior being selected, they're
		# omniscient, which includes the knowledge of the current mini-batch iteration's
		# index, `self.IB.t`. Thus, the 
		#
		#   `IntraclientByzantine::increment_mini_batch_iteration_t()` 
		#
		# method is always called first if
		#
		#  ((self.args.client_delta > 0.0) and (self.args.f == 0)): self.t += 1
		#
		# at the beginning of each aggregation round.
		#..................................................................................
		
		self.t = 0
		
		#..................................................................................
		# [BBG19] ALIE: codes.attacks.alittle
		#..................................................................................
		# Because our dynamic Byzantine behavior is invoked when
		#
		#   ((self.args.client_delta > 0.0) and (self.args.f == 0)),
		#
		# we should not refer to `args.f` to know the number of Byzantine messages, here.
		# Instead, `self.args.client_delta` is treated as the fraction of Byzantine
		# because, in expectation, this value converges to the constant `args.f`.
		#..................................................................................
		
		m = int(np.floor(self.args.client_delta * self.args.n))
		
		_tmp_s			= np.floor(self.args.n / 2 + 1) - m
		_tmp_cdf_value	= (self.args.n - m - _tmp_s) / (self.args.n - m)
		self.alie_z_max	= scipyNorm.ppf(_tmp_cdf_value)
		
		#..................................................................................
		# [XKG20] IPM: codes.attacks.xie
		#..................................................................................
			
		self.ipm_epsilon	= 0.1	# this was the choice in [KHJ22], safe to fix it, here
		
		#..................................................................................
		# [KHJ22] Mimic: codes.attacks.mimic
		#..................................................................................
			
		self.mimic_warmup		= self.args.mimic_warmup	# [KHJ22] `1` by default
		self.mimic_argmax		= True						# [KHJ22] `True` by default
		self.mimic_z			= None
		self.mimic_mu			= None
		self.mimic_target_rank	= -1
	
	#......................................................................................
	
	def increment_mini_batch_iteration_t(self):
		
		self.t += 1
		
	#--------------------------------------------------------------------------------------
	# mimic [KHJ22]
	#......................................................................................
		
	def mimic_init_callback(self):
		
		r"""
			Initialize z.
			
			self.good_grads == curr_good_grads
			self.good_ranks == curr_good_ranks
			self.honest_avg == curr_avg
		"""
		
		self.mimic_mu = self.honest_avg
		
		r = torch.Generator(device = self.honest_avg.device)
		r.manual_seed(0)
		
		self.mimic_z = torch.randn(
			len(self.honest_avg),
			generator = r,
			device = self.honest_avg.device
		)
		
		cumu = 0
		
		for g in self.good_grads:
			
			a	  = g - self.mimic_mu
			w	  = a.dot(self.mimic_z)
			cumu += w * a
			
		self.mimic_z = cumu / cumu.norm()
		
	def mimic_warmup_callback(self):
		
		r"""
			Update z in the warmup phase.
			
			self.good_grads == curr_good_grads
			self.good_ranks == curr_good_ranks
			self.honest_avg == curr_avg
		"""
		
		self.mimic_mu = self.t / (1 + self.t) * self.mimic_mu + self.honest_avg / (1 + self.t)

		cumu = 0
		
		for g in self.good_grads:
			
			a	  = g - self.mimic_mu
			w	  = a.dot(self.mimic_z)
			cumu += w * a
		
		self.mimic_z = self.t / (1 + self.t) * self.mimic_z + cumu / cumu.norm() / (1 + self.t)
		self.mimic_z = self.mimic_z / self.mimic_z.norm()
		
	def mimic_attack_callback(self):

		r"""
			self.good_grads == curr_good_grads
			self.good_ranks == curr_good_ranks
			self.honest_avg == curr_avg
		"""
		
		mv = None
		mi = None
		mg = None
		
		for i, g in enumerate(self.good_grads):
			
			d = g.dot(self.mimic_z)

			if self.mimic_argmax:
				
				if (mv is None) or (d > mv):
					
					mv = d; mg = g; mi = i;
			else:
				
				if (mv is None) or (d < mv):
					
					mv = d; mg = g; mi = i;
		
		return mv, mi, mg
		
	def mimic_maybe_setup_coordinator(self):
		
		r"""
			For the mimic attack, the coordinator logic is implemented within
			all the `ByzantineWorker` clients in Karimireddy's implementation [KHJ22].
			The coordinator is one of the `ByzantineWorker` clients that determines
			the `target_rank` from which the mimic attack vector is crafted.
			However, this coordinator behavior can also be simulated at this
			[simulator.py] from the `ParallelTrainer`'s side, easily. Thus, we leave
			this `maybe_setup_coordinator` method unimplemented.
		"""
		
		raise NotImplementedError
		
	def mimic_omniscient_callback(self):
		
		r"""
			self.good_grads == curr_good_grads
			self.good_ranks == curr_good_ranks
			self.honest_avg == curr_avg
		"""
		
		# Update z and mu
		
		if self.t == 0:
			
			self.mimic_init_callback()
			
		elif self.t < self.mimic_warmup:
			
			self.mimic_warmup_callback()

		# Find the target
		
		if self.t < self.mimic_warmup:
			
			mv, mi, mimic_gradient = self.mimic_attack_callback()
			self.mimic_target_rank = self.good_ranks[mi]

		else:
		
			# Fix device
			
			mimic_gradient = self.good_grads[self.mimic_target_rank]

		# handled instead by `increment_mini_batch_iteration_t` for all the clients
		# self.t += 1
		
		return mimic_gradient

#==========================================================================================

class DistributedSimulatorBase(object):
	
	r"""Simulate distributed programs with low memory usage.

		Functionality:
		1. randomness control: numpy, torch, torch-cuda
		2. add workers

		This base class is used by both trainer and evaluator.
	"""

	def __init__(self, metrics: dict, use_cuda: bool, debug: bool):
		
		r"""
			Args:
				metrics (dict): dict of metric names and their functions
				use_cuda (bool): Use cuda or not
				debug (bool):
		"""
		self.metrics		= metrics
		self.use_cuda		= use_cuda
		self.debug			= debug
		self.workers		= []

		self.json_logger	= logging.getLogger("stats")
		self.debug_logger	= logging.getLogger("debug")

#==========================================================================================

class ParallelTrainer(DistributedSimulatorBase):
	
	"""Synchronous and parallel training with specified aggregator."""

	def __init__(
		self,
		server: 					TorchServer,
		aggregator: 				Callable[[list], torch.Tensor],
		pre_batch_hooks: 			list,
		post_batch_hooks: 			list,
		max_batches_per_epoch: 		int,
		log_interval: 				int,
		log_directory: 				str,
		args: argparse.				Namespace,
		metrics: 					dict,
		use_cuda: 					bool,
		debug: 						bool,
		validation_loader: 			torch.utils.data.DataLoader = None,
		validation_std_th:			float = float(0.5)):
		
		r"""
			Args:
			
				aggregator (callable): A callable which takes a list of tensors and returns
					an aggregated tensor.
					
				max_batches_per_epoch (int): Set the maximum number of batches in an epoch.
					Usually used for debugging.
					
				log_interval (int): Control the frequency of logging training batches
				
				metrics (dict): dict of metric names and their functions
				
				validation_loader (DataLoader): Optional. A DataLoader for sampling 
					validation mini-batches during training.
					
				use_cuda (bool): Use cuda or not
				
				debug (bool):
		"""
		
		self.aggregator				= aggregator
		self.server					= server
		self.pre_batch_hooks		= pre_batch_hooks or []
		self.post_batch_hooks		= post_batch_hooks or []
		self.log_interval			= log_interval
		self.log_directory			= log_directory
		self.max_batches_per_epoch	= max_batches_per_epoch
		self.omniscient_callbacks	= []
		self.random_states			= {}
		self.args					= args
		
		self.validation_loader		= validation_loader
		self.validation_std_th		= validation_std_th
		
		self.validation_accuracy_sum  = 0.0
		self.validation_batches_count = 0
		
		self.IB = None
		
		self.history_is_byzantine		= []
		self.history_byzantine_behavior = []
		
		super().__init__(metrics, use_cuda, debug)
	
	#--------------------------------------------------------------------------------------
	
	def print_args(self):
		
		to_print = (f"\n[args]\n")
	
		args_dict = vars(self.args)

		max_key_length = max(len(key) for key in args_dict.keys())

		for key in sorted(args_dict.keys()):
			
			to_print += (f"{key:<{max_key_length}}: {args_dict[key]}\n")
		
		self.debug_logger.info(f"{to_print}")
		
	#--------------------------------------------------------------------------------------
	
	def aggregation_and_update(self, epoch):
		
		global _GLOBAL_DEBUGGER_COUNTER_
	
		#......................................................................................
		# if there are Byzantine workers, ask them to craft attacks based on the updated models
		#......................................................................................
		
		for omniscient_attacker_callback in self.omniscient_callbacks:
			omniscient_attacker_callback()
		
		#......................................................................................
		# Get gradients from all workers: torch.Tensor with gradients[i].shape = 1199882.
		# The output of `w.get_gradient()` is already momentum adjusted by beta.
		#......................................................................................
		
		types, gradients = self.parallel_get(lambda w: w.get_gradient())
		
		#......................................................................................
		# dispersion for in-distribution attacks using dimensionwise min-max distances
		#......................................................................................
		
		if ((self.args.dispersion_idea == 1) and 
			(self.args.attack in {"mimic", "IPM", "ALIE"})):
			
			dispersion_factor = torch.tensor(
				float(self.args.dispersion_factor),
				device = gradients[0].device,
				dtype  = gradients[0].dtype
			)
			
			_n_minus_f_		= int(self.args.n) - int(self.args.f)
			dimensionality 	= gradients[0].numel()
			
			stacked_honest 	= torch.empty(
				(_n_minus_f_, dimensionality),
				device = gradients[0].device,
				dtype  = gradients[0].dtype
			)
			
			cnt_honest = 0
			
			for w in range(len(self.workers)):
				
				if not isinstance(self.workers[w], ByzantineWorker):
					
					stacked_honest[cnt_honest] = gradients[w]
					cnt_honest += 1
					
			n_minus_f, d = stacked_honest.shape
			
			assert _n_minus_f_ == n_minus_f		# `stacked_honest` is of shape [n - f, d]

			# Expand dimensions to calculate pairwise differences
			
			expanded_1 = stacked_honest.unsqueeze(1)  # Shape: [n - f, 1, d]
			expanded_2 = stacked_honest.unsqueeze(0)  # Shape: [1, n - f, d]

			# Compute pairwise absolute differences along each dimension
			
			pairwise_diffs = torch.abs(expanded_1 - expanded_2)  # Shape: [n - f, n - f, d]

			# Create a mask to exclude diagonal elements
			
			mask = torch.eye(
				n_minus_f, device=pairwise_diffs.device, dtype=torch.bool
			).unsqueeze(-1)  # Shape: [n - f, n - f, 1]

			# Apply the mask during minimum and maximum calculations
			
			min_distances_along_dim0 = torch.masked_fill(
				pairwise_diffs, mask, float('inf')
			).min(dim=0)[0]  # Shape: [n - f, d]
			
			rho_min = min_distances_along_dim0.min(dim=0)[0]  # Shape: [d]

			max_distances_along_dim0 = torch.masked_fill(
				pairwise_diffs, mask, float('-inf')
			).max(dim=0)[0]  # Shape: [n - f, d]
			
			rho_max = max_distances_along_dim0.max(dim=0)[0]  # Shape: [d]

			# Compute the convex combination of min and max distances
			
			rho = (1 - dispersion_factor) * rho_min + dispersion_factor * rho_max  # Shape: [d]
			
			# Below, dispersion is added to the Byzantine vectors.
			# For investigation purposes, optionally, we save
			# what dispersion is being added to the Byzantine vectors.
			
			if _FLAG_DATAFY_BYZANTINE_NOISES_:
				
				stacked_dispersion = torch.empty(
					(int(self.args.f), dimensionality),
					device=gradients[0].device,
					dtype=gradients[0].dtype
				)
				
				stacked_dispersion.fill_(-float('inf'))
			
			cnt_byzantine = 0
			
			for w in range(len(self.workers)):
				
				if isinstance(self.workers[w], ByzantineWorker):
			
					noise = torch.randn_like(gradients[w]) * rho
					
					gradients[w] += noise
					
					if _FLAG_DATAFY_BYZANTINE_NOISES_:
						
						stacked_dispersion[cnt_byzantine] = noise
					
					cnt_byzantine += 1
			
			if _FLAG_DATAFY_BYZANTINE_NOISES_:
				
				savemat(
					f"{self.log_directory}\\dispersion_at_mini_batch_cnt"
					f"_{_GLOBAL_DEBUGGER_COUNTER_:05d}.mat",
					{'dispersion': stacked_dispersion.cpu().numpy()}
				)
				
				_GLOBAL_DEBUGGER_COUNTER_ += 1
				
		#......................................................................................
		
		if _FLAG_DATAFY_CLIENT_MESSAGES_:
		
			for i in range(len(gradients)):
			
				savemat(f"{self.log_directory}\\epoch_{epoch:02d} id_{i:02d} {types[i]}.mat",
					{'message': gradients[i].cpu().numpy()}
				)

		#--------------------------------------------------------------------------------------
		
		if FLAG_MONITOR_CPU_MEMORY: tracemalloc.start()

		#......................................................................................
		# 2025-01-20 (dynamic byzantine behavior)
		#......................................................................................
		
		if ((self.args.client_delta > 0.0) and (self.args.f == 0)):
			
			# [xie.py, mimic.py]
			
			# regardless of individual client's Byzantine behavior being selected,
			# they're omniscient, which includes the knowledge of the current
			# mini-batch iteration's index, `self.IB.t`
			
			self.IB.increment_mini_batch_iteration_t()
			
			self.IB.good_grads = []
			self.IB.good_ranks = []
			
			for (_id_, _client_) in enumerate(self.workers):
				
				if not isinstance(_client_, ByzantineWorker):
					
					self.IB.good_grads.append(_client_.get_gradient())
					self.IB.good_ranks.append(_id_)
			
			self.IB.honest_avg = (sum(self.IB.good_grads)) / len(self.IB.good_grads)
		
		for w in range(len(self.workers)):
			
			#..................................................................................
			# TorchWorker.byzantine_behavior = {0: BF, 1: LF, 2: mimic, 3: IPM, 4: ALIE}
			#..................................................................................
			#
			# TorchWorker.determine_byzantine_behavior() determines whether
			# 	a client is Byzantine in the current aggregation round,
			#	and what kind of Byzantine behavior it exhibits if so.
			#
			#..................................................................................
			#
			# Labelflipping (LF) is a special case which requires to be implemented
			# during the computation of the gradient instead of crafting a Byzantine message
			# simply before the aggregation stage (to simulate): TorchWorker.compute_gradient()
			#
			#..................................................................................
			
			if self.workers[w].is_byzantine == False:
				
				# if args.client_delta == 0.0: none of workers can be Byzantine
				# else: this client is not Byzantine in the current aggregation round
				
				continue
				
			#..................................................................................
			
			if   self.workers[w].byzantine_behavior == 0:	# BF [KHJ22]
				
				gradients[w] *= -1.0
				
			#..................................................................................
				
			elif self.workers[w].byzantine_behavior == 2:	# mimic [KHJ22]
				
				gradients[w] = self.IB.mimic_omniscient_callback()
				
			#..................................................................................
				
			elif self.workers[w].byzantine_behavior == 3:	# IPM [XKG20]
				
				gradients[w] = -self.IB.ipm_epsilon * self.IB.honest_avg
				
			#..................................................................................
				
			elif self.workers[w].byzantine_behavior == 4:	# ALIE [BBG19]
				
				stacked_gradients = torch.stack(self.IB.good_grads, 1)
				
				mu  = torch.mean(stacked_gradients, 1)
				std = torch.std(stacked_gradients, 1)

				gradients[w] = mu - std * self.IB.alie_z_max
				
				pass
				
			#..................................................................................

		#######################################################################################
		# The aggregation at the trusted server arises right here.
		#......................................................................................
		
		aggregated = self.aggregator(gradients)
		
		#......................................................................................
		
		if FLAG_MONITOR_CPU_MEMORY:
			
			current_cpu_mem, current_cpu_mem_peak = tracemalloc.get_traced_memory()
			print(
				f"- cpu memory: {current_cpu_mem / 1e6:.2f}"
				f" (peak: {current_cpu_mem_peak / 1e6:.2f}) MB"
				)
				
			tracemalloc.stop()

		#......................................................................................
		
		_IS_STRATIFIED_ = False		# Initial sanity check, previously
		
		if _IS_STRATIFIED_:
			
			print(f"Before measuring the quality of validation set")
			print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
			print(f"Cached Memory:	  {torch.cuda.memory_reserved()  / 1024**2:.2f} MB")
			print(f"")
			
			# Compute the class distribution
			
			unique_classes, counts	= torch.unique(val_target, return_counts=True)
			class_distribution		= dict(zip(unique_classes.cpu().tolist(), counts.cpu().tolist()))

			# Calculate total samples and ideal count
			
			total_samples	= sum(class_distribution.values())
			num_classes		= len(class_distribution)
			ideal_count		= total_samples / num_classes  # Assuming balanced classes
			
			# Calculate deviation and standard deviation
			
			deviations			= [(count - ideal_count) for count in class_distribution.values()]
			variance			= sum(dev**2 for dev in deviations) / num_classes
			standard_deviation	= variance**0.5
			
			if standard_deviation > self.validation_std_th:
				
				error_message = (
					f"Validation batch class imbalance detected!\n"
					f"Standard deviation vs. Threshold: {standard_deviation:.4f} > {self.validation_std_th:.4f}.\n"
					f"Ideal count per class: {ideal_count:.2f}.\n"
					f"Class distribution: {class_distribution}."
				)
				
				# self.debug_logger.error(error_message)
				raise RuntimeError(error_message)
				
			del unique_classes, counts, class_distribution, total_samples
			del num_classes, ideal_count, deviations, variance, standard_deviation

		#......................................................................................
		# assume that the model and optimizers are shared among workers
		#......................................................................................
		
		self.server.model.train()
		self.server.set_gradient(aggregated)
		self.server.apply_gradient()
		
		# if ((isinstance(self.aggregator, ByRoAdapter)) and
			# (self.validation_loader is not None)):
				
			# del val_data, val_target, validation_batch
			
		del types, gradients, aggregated
		
		#END: ParallelTrainer.aggregation_and_update()
		
	#--------------------------------------------------------------------------------------
	# 2025-01-20 (selective learning): Aggregate classwise accuracy, `eval_results`
	#......................................................................................
	
	def prepareByzantines(self):
			
		self.IB = IntraclientByzantine(self.args)
	
	#......................................................................................
		
	def train(self, epoch, eval_results=None):
		
		self.debug_logger.info(f"> Train epoch {epoch}")
		self.parallel_call(lambda worker: worker.train_epoch_start())

		progress = 0
		
		for batch_idx in range(self.max_batches_per_epoch):
		
			try:
			
				self._run_pre_batch_hooks(epoch, batch_idx)
				
				#..........................................................................
				# 2025-01-20 (dynamic byzantine behavior)
				#......................................................................
				# At each aggregation round, a candidate byzantine method is 
				# determined: `TorchWorker.byzantine_behavior` in [worker.py]:
				# {	0: BF, 1: LF, 2: mimic, 3: IPM, 4: ALIE	}
				#......................................................................
				
				is_byzantine_client = {i: None for i in range(self.args.n)}
				byzantine_behaviors = {i: None for i in range(self.args.n)}
				
				for i in range(len(self.workers)):
					
					self.workers[i].determine_byzantine_behavior()
					
					is_byzantine_client[i] = self.workers[i].is_byzantine
					byzantine_behaviors[i] = self.workers[i].byzantine_behavior
					
				self.history_is_byzantine.append(is_byzantine_client)
				self.history_byzantine_behavior.append(byzantine_behaviors)
				
				#..........................................................................
				
				_, results = self.parallel_get(
				
					lambda w: w.compute_gradient()
					
				)
				
				self.aggregation_and_update(epoch)
				
				torch.cuda.empty_cache()
				gc.collect()
				
				#..........................................................................

				progress += sum(res["length"] for res in results)
				
				if batch_idx % self.log_interval == 0:
					
					self.log_train(
						progress, batch_idx, epoch, results, 
						eval_results
					)
					
				self._run_post_batch_hooks(epoch, batch_idx)
				
			except StopIteration: break
			

	# -----------------------------------------------------------------------------	#
	#									utilities							   		#
	# -----------------------------------------------------------------------------	#

	def add_worker(self, worker: TorchWorker):
		
		worker.add_metrics(self.metrics)
		self.workers.append(worker)
		self.debug_logger.info(f"=> Add worker {worker}")

	#......................................................................................

	def register_omniscient_callback(self, callback):
		
		self.omniscient_callbacks.append(callback)
		
	#......................................................................................

	def cache_random_state(self) -> None:
		
		if self.use_cuda:
			self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
			
		self.random_states["torch"] = torch.get_rng_state()
		self.random_states["numpy"] = np.random.get_state()
		
	#......................................................................................

	def restore_random_state(self) -> None:
		
		if self.use_cuda:
			torch.cuda.set_rng_state(self.random_states["torch_cuda"])
		torch.set_rng_state(self.random_states["torch"])
		np.random.set_state(self.random_states["numpy"])
		
	#......................................................................................

	def parallel_call(self, f: Callable[[TorchWorker], None]) -> None:
		
		for w in self.workers:
			self.cache_random_state()
			f(w)
			self.restore_random_state()
			
	#......................................................................................

	def parallel_get(self, f: Callable[[TorchWorker], Any]) -> list:
	
		types   = []  # {TorchWorker, Byzantine such as LableFlippingWorker}
		results = []  # stochastic gradient
		
		for w in self.workers:
			
			self.cache_random_state()
			types.append(str(w))
			results.append(f(w))
			self.restore_random_state()
			
		return types, results
		
	#......................................................................................

	def _run_pre_batch_hooks(self, epoch, batch_idx):
		
		[f(self, epoch, batch_idx) for f in self.pre_batch_hooks]
		
	#......................................................................................

	def _run_post_batch_hooks(self, epoch, batch_idx):
		
		[f(self, epoch, batch_idx) for f in self.post_batch_hooks]

	# -----------------------------------------------------------------------------	#
	#								log information									#
	# -----------------------------------------------------------------------------	#

	def __str__(self):
		
		return (
			"ParallelTrainer("
			f"aggregator={self.aggregator}, "
			f"max_batches_per_epoch={self.max_batches_per_epoch}, "
			f"log_interval={self.log_interval}, "
			f"metrics={list(self.metrics.keys())}"
			f"use_cuda={self.use_cuda}, "
			f"debug={self.debug}, "
			")"
		)

	#................................................................................
	# 2025-01-20 (selective learning): Aggregate classwise accuracy
	#................................................................................
	# Client-side classwise accuracy computation is unnecessary;
	#   Each worker uses `NONIIDLTSampler` to train on its non-IID dataset, 
	#   with honest workers computing valid gradients and Byzantine workers
	#   manipulating theirs, while the server aggregates all updates using
	#   `ByRoAdapter`, which optimizes clipping radii to ensure robust aggregation;
	#   despite manipulations, each worker computes Top-1 Accuracy and classwise
	#   accuracy on its own dataset, reflecting local performance metrics unaffected
	#   by gradient tampering, which are logged for detailed evaluation.
	#................................................................................
	
	def log_train(self, progress, batch_idx, epoch, results=None, eval_results=None):
		
		r = {
			"_meta": {"type": "train" if results else "evaluation"},
			"E": epoch,
			"B": batch_idx,
		}

		# Handle training results
		
		if results:
			
			length = sum(res["length"] for res in results)
			
			r["Length"] = length
			r["Loss"] = sum(res["loss"] * res["length"] for res in results) / length
			
			for metric_name in self.metrics:
				
				r[metric_name] = (
					sum(res["metrics"][metric_name] * res["length"] for res in results)
					/ length
				)

		# Handle evaluation results
		
		if eval_results:
			
			r.update(eval_results)

		# Output to console
		
		if results:
			
			total = len(self.workers[0].data_loader.dataset)
			
			pct = 100 * progress / total if progress else 0
			
			to_print = (
				f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%)] "
				f"Loss: {r['Loss']:.4f} "
				+ " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
			)
			
		else:
			to_print = (
				f"[E{r['E']:2}] Eval Loss: {r['Loss']:.4f} "
				f"top1={r['top1']:.2f}%\n\tClass Accuracy:\n"
				+ "".join(
					f"\tClass {cls}: {acc:.2f}%\n"
					for cls, acc in r.get("ClassAccuracy", {}).items()
				)
			)
		
		if ((isinstance(self.server.aggregator, ByRoAdapter)) and
			(self.validation_loader is not None)):
			
			to_print += (f" [omega: {self.server.aggregator.omega.item():.2e}]")
			to_print += (
				f"[gd (bk): {self.server.aggregator.gd_cum_cnt:3d} "
				f"({self.server.aggregator.bk_cum_cnt:3d})]"
			)
			to_print += (f"[maturity: {self.server.aggregator.flag_maturity}]")
			
			if self.server.aggregator.tau != None and False:
				
				to_print += (f"[tau: {format_tensor_to_print(self.server.aggregator.tau)}]")
				
			if self.server.aggregator.info != None:
				
				to_print += (f"{self.server.aggregator.info}")
			
			#................................................................................
			# 2025-01-20 (selective learning): per-class accuracy
			#................................................................................
			# (unnecessary: client-side)
			# to_print += "\n\tClass Accuracy:\n" + "".join(
			# 	f"\tClass {cls}: {acc:.2f}%\n" for cls, acc in class_accuracy.items()
			# )
			#................................................................................
			if "ClassAccuracy" in r:
				
				to_print += "\n\tClass Accuracy:\n" + "".join(
					f"\tClass {cls}: {acc:.2f}%\n" for cls, acc in r["ClassAccuracy"].items()
				)
			#................................................................................
			
		if self.args.use_cuda and False:
				
			peak_cuda_memory = torch.cuda.max_memory_allocated() / 1e9	# [GB]
			
			to_print += (f" @cuda-mem: {peak_cuda_memory:.2f} [GB]")

		self.debug_logger.info(to_print)

		# Output to file
		self.json_logger.info(r)

#==========================================================================================

class DistributedEvaluator(DistributedSimulatorBase):
	
	def __init__(
	
		self,
		model: torch.nn.Module,
		data_loader: torch.utils.data.DataLoader,
		loss_func: torch.nn.modules.loss._Loss,
		device: Union[torch.device, str],
		metrics: dict,
		use_cuda: bool,
		debug: bool,
		log_identifier_type="validation",
	):
		
		super().__init__(metrics, use_cuda, debug)
		
		self.model = model
		self.data_loader = data_loader
		self.loss_func = loss_func
		self.device = device
		self.log_identifier_type = log_identifier_type

	def __str__(self):
		
		return (
			"DistributedEvaluator("
			f"use_cuda={self.use_cuda}, "
			f"debug={self.debug}, "
			")"
		)

	# 2025-01-20 (selective learning)
	
	def evaluate(self, epoch):
		
		self.model.eval()
		r = {
			"_meta":   {"type": self.log_identifier_type},
			"E":	   epoch,
			"Length":  0,
			"Loss":	   0,
		}
		
		# Dynamically initialize metrics from self.metrics
		
		for name in self.metrics:
			
			r[name] = 0
		
		correct_predictions = defaultdict(int)
		total_per_class = defaultdict(int)

		with torch.no_grad():
			
			for _, (data, target) in enumerate(self.data_loader):
				
				data, target = data.to(self.device), target.to(self.device)
				output = self.model(data)
				
				# Compute loss
				
				r["Loss"]   += self.loss_func(output, target).item() * len(target)
				r["Length"] += len(target)

				# Compute top-1 accuracy and classwise accuracy
				
				_, pred = output.max(1)
				
				for t, p in zip(target, pred):
					
					correct_predictions[t.item()] += (t == p).item()
					total_per_class[t.item()] += 1
				
				# Update top1 metric dynamically from self.metrics
				
				for name, metric in self.metrics.items():
					
					r[name] += metric(output, target) * len(target)

		for name in self.metrics:
			
			r[name] /= r["Length"]
			
		r["Loss"] /= r["Length"]
		
		# Calculate classwise accuracy
		
		class_accuracy = {
			cls: (correct_predictions[cls] / total_per_class[cls] * 100.0)
			if total_per_class[cls] > 0 else 0.0
			for cls in total_per_class
		}
		r["ClassAccuracy"] = class_accuracy

		# Output handled by `log_train`; log evaluation results directly
		
		self.json_logger.info(r)
		
		to_print = (
			f"[E{r['E']:2}] Eval Loss: {r['Loss']:.4f} "
			f"top1={r['top1']:.2f}%\n\n\tClass Accuracy:\n"
			+ "".join(
				f"\t\tClass {cls}: {acc:.2f}%\n"
				for cls, acc in r.get("ClassAccuracy", {}).items()
			)
		)
		self.debug_logger.info(to_print)

		return r  # Ensure the dictionary is returned
