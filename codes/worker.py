import torch
import random
from collections import defaultdict
from typing import Optional, Union, Callable, Any, Tuple


class TorchWorker(object):
	
	r"""
		A worker for distributed training.
		Compute gradients locally and store the gradient.
	"""

	def __init__(
		self,
		data_loader:	torch.utils.data.DataLoader,
		model:			torch.nn.Module,
		optimizer:		torch.optim.Optimizer,
		loss_func:		torch.nn.modules.loss._Loss,
		device:			Union[torch.device, str],
		client_delta:	float = 0.0,
		client_id:		int = -1,
	):
		
		self.data_loader	= data_loader
		self.model			= model
		self.optimizer		= optimizer
		self.loss_func		= loss_func
		self.device			= device
		
		#......................................................................................
		# 2025-01-20 (dynamic byzantine behavior)
		#......................................................................................
		#
		#	TorchWorker behaves as a Byzantine according to `args.client_delta` \in [0, 1):
		#
		#		[simulator.py] ParallelTrainer::train():
		#
		#			TorchWorker.determine_byzantine_behavior()
		#				self.byzantine_behavior = {0: BF, 1: LF, 2: mimic, 3: IPM, 4: ALIE}
		#
		#			TorchWorker.compute_gradient():
		#				if self.rng_is_byzantine.random() < self.client_delta
		#
		#......................................................................................
		
		self.client_id		= client_id
		self.client_delta	= client_delta
		
		self.rng_is_byzantine	= random.Random(self.client_id)
		self.rng_byzantine_mode = random.Random(self.client_id)
		self.byzantine_behavior = -1
		self.is_byzantine		= False
		
		#......................................................................................

		# self.running has attribute:
		#   - `train_loader_iterator`: data iterator
		#   - `data`: last data
		#   - `target`: last target
		
		self.running = {}
		self.metrics = {}
		self.state	 = defaultdict(dict)

	def determine_byzantine_behavior(self, low: int = 0, high: int = 4) -> int:
		
		r"""................................................................
		
			Call this method 
				before TorchWorker.compute_gradient()
				within [simulator.py] for a dynamic Byzantine behavior
			
			Draw a random integer between `low` and `high` (inclusive).
			
			Args:
				low (int): Lower bound for the random number (default: 0).
				high (int): Upper bound for the random number (default: 4).
			
			Returns:
				int: Random integer in the range [low, high].
				
		................................................................."""
		
		if self.client_delta > 0.0:
		
			self.is_byzantine = self.rng_is_byzantine.random() < self.client_delta
			self.byzantine_behavior = self.rng_byzantine_mode.randint(low, high)
			
		else:	# args.client_delta == 0.0
			
			self.is_byzantine		 = False
			self.byzantine_behavior = -1

	def add_metric(self, name: str,
		callback: Callable[[torch.Tensor, torch.Tensor], float],):
			
		r"""
			The `callback` function takes predicted and groundtruth value
			and returns its metric.
		"""
		
		if name in self.metrics or name in ["loss", "length"]:
			
			raise KeyError(f"Metrics ({name}) already added.")

		self.metrics[name] = callback


	def add_metrics(self, metrics: dict):
		
		for name in metrics:
			
			self.add_metric(name, metrics[name])


	def __str__(self) -> str:
		
		return "TorchWorker"


	def train_epoch_start(self) -> None:
		
		self.running["train_loader_iterator"] = iter(self.data_loader)
		self.model.train()


	def compute_gradient(self) -> Tuple[float, int]:
		
		results = {}

		data, target = self.running["train_loader_iterator"].__next__()
		data, target = data.to(self.device), target.to(self.device)
		
		#......................................................................................
		#
		# Implements the label-flipping done in [KHJ22]. The difference here is that 
		# every honest client can be a label-flipping Byzantine by probability.
		# As we just flip the target labels when computing the mini-batch stochastic
		# gradient, THE TRAINING ACCURACY WILL BE MISLEADING. However, we only care
		# the test accuracy, thus, THE IMPLEMENTAITON HERE IS FINE.
		#
		#......................................................................................
		#
		# [simulator.py] ParallelTrainer.aggregation_and_update()
		#
		# At each aggregation round, a candidate byzantine method is determined,
		# which is accessible as `TorchWorker.byzantine_behavior` in [worker.py]:
		# {	0: BF, 1: LF, 2: mimic, 3: IPM, 4: ALIE	}
		#......................................................................................
		
		# e.g., [0, 1) < 0.3 = self.client_delta
		
		if ((self.is_byzantine) and
			(self.byzantine_behavior == 1)):
			
			target = 9 - target		# Label-flipping: `1`
			
		#......................................................................................
		
		self.optimizer.zero_grad()
		output = self.model(data)
		loss = self.loss_func(output, target)
		loss.backward()
		self._save_grad()

		self.running["data"] = data
		self.running["target"] = target

		# 2025-01-20 (selective learning)
		
		results["class_accuracy"] = self._compute_class_accuracy(output, target)

		results["loss"] = loss.item()
		results["length"] = len(target)
		results["metrics"] = {}
		
		for name, metric in self.metrics.items():
			results["metrics"][name] = metric(output, target)
			
		return results


	# 2025-01-20 (selective learning)
	
	def _compute_class_accuracy(self, output, target):
		
		pred = output.argmax(dim=1)
		correct = pred.eq(target)
		class_accuracy = {}
		
		for cls in target.unique():
			
			mask = target == cls
			
			if mask.sum().item() > 0:
				
				class_accuracy[int(cls)] = correct[mask].float().mean().item() * 100
				
		return class_accuracy
		

	def get_gradient(self) -> torch.Tensor: return self._get_saved_grad()
	

	def apply_gradient(self) -> None: self.optimizer.step()
	

	def set_gradient(self, gradient: torch.Tensor) -> None:
		
		beg = 0
		for p in self.model.parameters():
			end = beg + len(p.grad.view(-1))
			x = gradient[beg:end].reshape_as(p.grad.data)
			p.grad.data = x.clone().detach()
			beg = end


	def _save_grad(self) -> None:
		
		for group in self.optimizer.param_groups:
			
			for p in group["params"]:
				
				if p.grad is None: continue
					
				param_state = self.state[p]
				param_state["saved_grad"] = torch.clone(p.grad).detach()


	def _get_saved_grad(self) -> torch.Tensor:
		
		layer_gradients = []
		
		for group in self.optimizer.param_groups:
			
			for p in group["params"]:
				
				param_state = self.state[p]
				layer_gradients.append(param_state["saved_grad"].data.view(-1))
				
		return torch.cat(layer_gradients)


class MomentumWorker(TorchWorker):
	
	def __init__(self, momentum, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		self.momentum = momentum

	def _save_grad(self) -> None:
		
		for group in self.optimizer.param_groups:
			
			for p in group["params"]:
				
				if p.grad is None: continue
					
				param_state = self.state[p]
				
				if "momentum_buffer" not in param_state:
					
					param_state["momentum_buffer"] = torch.clone(p.grad).detach()
					
				else:
					
					param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)

	def _get_saved_grad(self) -> torch.Tensor:
		
		layer_gradients = []
		
		for group in self.optimizer.param_groups:
			
			for p in group["params"]:
				
				param_state = self.state[p]
				layer_gradients.append(param_state["momentum_buffer"].data.view(-1))
				
		return torch.cat(layer_gradients)


class ByzantineWorker(TorchWorker):
	
	def configure(self, simulator):
		
		# call configure after defining DistribtuedSimulator
		
		self.simulator = simulator
		simulator.register_omniscient_callback(self.omniscient_callback)

	def compute_gradient(self) -> Tuple[float, int]:
		
		# Use self.simulator to get all other workers
		# Note that the byzantine worker does not modify the states directly.
		
		return super().compute_gradient()

	def get_gradient(self) -> torch.Tensor:
		
		# Use self.simulator to get all other workers
		
		return super().get_gradient()

	def omniscient_callback(self):
		
		raise NotImplementedError

	def __str__(self) -> str:
		
		return "ByzantineWorker"
