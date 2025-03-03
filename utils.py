_LOG_DIR_GLOBAL_IN_UTILS_DOT_PY_				= None		# @_FLAG_DATAFY_CLIENT_MESSAGES_IN_ONE_DIMENSION_
_FLAG_DATAFY_CLIENT_MESSAGES_IN_ONE_DIMENSION_	= False

from scipy.io import savemat								# @_FLAG_DATAFY_CLIENT_MESSAGES_IN_ONE_DIMENSION_

import argparse
import numpy as np
from   sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict

import os, math, torch, json
from torchvision import datasets, transforms
import torch.nn.functional as F

# Utility functions

from codes.tasks.mnist	import Net, mnist, mnist_validation_dataset_loader
from codes.utils		import top1_accuracy, initialize_logger

# Attacks

from codes.attacks.labelflipping	import LableFlippingWorker
from codes.attacks.bitflipping		import BitFlippingWorker
from codes.attacks.mimic			import MimicAttacker, MimicVariantAttacker
from codes.attacks.xie				import IPMAttack
from codes.attacks.alittle			import ALittleIsEnoughAttack

# Main Modules

from codes.worker		import TorchWorker, MomentumWorker
from codes.server		import TorchServer
from codes.simulator	import ParallelTrainer, DistributedEvaluator

# IID vs Non-IID

from codes.sampler import (
	DistributedSampler,
	NONIIDLTSampler,
)

# Aggregators

from codes.aggregator.base 					import Mean
from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.clipping 				import Clipping
from codes.aggregator.clipping 				import ByRoAdapter
from codes.aggregator.rfa 					import RFA
from codes.aggregator.trimmed_mean 			import TM
from codes.aggregator.krum 					import Krum

# DataLoader

from codes.tasks.mnist import (
	mnist_validation_retrieve_batch,
	mnist_validation_sort_batch_by_labels,
	mnist_validation_plot_images,
	mnist_validation_get_class_distribution
)

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

def get_args():

	parser = argparse.ArgumentParser(description="")

	#======================================================================================
	# Utility
	#......................................................................................
	
	parser.add_argument("--use-cuda",		 action = "store_true", default = False)
	parser.add_argument("--debug",			 action = "store_true", default = False)
	parser.add_argument("--seed",			 type = int, default = 0)
	parser.add_argument("--log_interval",	 type = int, default = 10)
	parser.add_argument("--dry-run",		 action = "store_true", default = False)
	parser.add_argument("--identifier",		 type = str, default = "debug", help = "")
	parser.add_argument("--plot",			 action = "store_true", default = False,
		help="If plot is enabled, then ignore all other options.",
	)

	#......................................................................................
	# Experiment configuration
	#......................................................................................
	
	parser.add_argument("-n",		type = int, help = "Number of workers")
	parser.add_argument("-f",		type = int, help = "Number of Byzantine workers.")
	parser.add_argument("--attack", type = str, default = "NA",  help = "Type of attacks.")
	parser.add_argument("--agg",	type = str, default = "avg", help = "")
	parser.add_argument("--noniid", 
		action="store_true",
		default=False,
		help="[HP] noniidness.",
	)
	
	parser.add_argument("--LT", action="store_true", default=False, help="Long tail")
	
	parser.add_argument("--bucketing",
		type=float, default=0.0, help="bucket size s")
	
	parser.add_argument("--momentum",	  type=float, default=0.0,  help="[HP] momentum")
	parser.add_argument("--clip-tau",	  type=float, default=10.0, help="[HP] momentum")
	parser.add_argument("--clip-scaling", type=str,   default=None, help="[HP] momentum")

	parser.add_argument(
		"--mimic-warmup",
		type=int, default=1,
		help="the warmup phase in iterations."
	)

	parser.add_argument(
		"--op", type=int, default=1,
		help="[HP] controlling the degree of overparameterization. "
		"Only used in exp8.",
	)
	
	#======================================================================================
	# adaptive centered clipping
	#......................................................................................
	
	parser.add_argument("--gVonly", action = "store_true", default = False,
		help="Only the validation gradient gV is used for training."
	)
	
	parser.add_argument("--beta_gV", type = float, default = 0.0,
		help = "Momentum weight for validation gradient"
	)
	
	parser.add_argument("--val_set_sz", type = int, default = 100,
		help = "# of validation set data points"
	)
	
	parser.add_argument("--train_mb_sz", type = int, default = 32,
		help = "# of training mini-batch size"
	)
	
	#--------------------------------------------------------------------------------------
	# 2025-01-20 (selective learning, intermittent byzantine)
	#......................................................................................
	
	parser.add_argument("--client_delta", type = float, default = 0.0,
		help = "Per-client byzantine fraction (coin-flip)"
	)
	
	r"""...................................................................................
	
		related to the collaborative learning (collaborative/selective learning)
	
		if args.target == "1111111111":
	
			We're working on the vanilla Byzantine optimization setting:
			- initial guess of tau = max || x0 - xi || in each mini-bath iteration
			- {MAX_BATCHES_PER_EPOCH = 30, EPOCHS = 20} @exp3.py
			
		else:
		
			- e.g., {MAX_BATCHES_PER_EPOCH = 25, EPOCHS = 4} @exp3.py
			- e.g., initial guess of tau = 3 (fixed vanilla choice)
			
	...................................................................................."""
	
	parser.add_argument("--target",
		type	= str,
		default	= "1111111111", 
		help	= "target classes written as a binary string"
	)
	
	#--------------------------------------------------------------------------------------
	# graveyard experiments
	#......................................................................................
	
	parser.add_argument("--bucketing_idea",
		type=int, default=0, help="switch-case for different bucketing ideas")
		
	parser.add_argument("--dispersion_idea",	# simulator.py
		type=int, default=0,
		help="flag for different dispersion ideas for in-distribution attacks")
		
	parser.add_argument("--dispersion_factor",
		type=float, default=1.0, help="a value between 0 and 1 to control dispersion")
		
	parser.add_argument("--projection_idea",
		type=int, default=0,
		help="switch-case for different projection ideas for in-distribution attacks")
	
	parser.add_argument("--proj_warm_it",
		type=int, default=1,
		help="number of online power iterations to warm up for locally-sensitive bucketing")
		
	parser.add_argument("--generic_it",
		type=int, default=1,
		help="number of online power iterations generally for locally-sensitive bucketing")
	
	#......................................................................................	
	# EOL: parser
	#......................................................................................

	args = parser.parse_args()
	
	assert ((args.client_delta >= 0.0) and (args.client_delta < 1.0)), (
		f"args.client_delta: {args.client_delta}"
	)
	
	if ((args.client_delta > 0.0) and (args.f > 0)):
		
		raise RuntimeError(
			f"Invalid configuration: `args.client_delta > 0.0` simulates\n"
			f"dynamic Byzantine behavior where all clients are implemented\n"
			f"as (honest) MomentumWorker. Hence, the number of Byzantine\n"
			f"clients `args.f` must be zero if `args.client_delta > 0.0`."
		)

	if args.n <= 0 or args.f < 0 or args.f >= args.n:
		raise RuntimeError(f"n={args.n} f={args.f}")

	assert args.bucketing >= 0.0, float(args.bucketing)
	assert args.momentum  >= 0.0, float(args.momentum)
	assert len(args.identifier) > 0.0

	return args


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
DATA_DIR = ROOT_DIR + "datasets/"
EXP_DIR  = ROOT_DIR + f"outputs/"

LR				  = 0.01
_TEST_BATCH_SIZE_ = 128

#==========================================================================================

def _get_aggregator(args, device, model, loss_func, server, 
	validation_loader: torch.utils.data.DataLoader = None):
	
	if args.agg == "avg": return Mean()
	if args.agg == "cm":  return CM()
	
	if args.agg == "cp":
		
		if	  args.clip_scaling is None:	 tau = args.clip_tau
		elif  args.clip_scaling == "linear": tau = args.clip_tau / (1 - args.momentum)
		elif  args.clip_scaling == "sqrt":	 tau = args.clip_tau / np.sqrt(1 - args.momentum)
		else: raise NotImplementedError(args.clip_scaling)
		
		print("")
		print(f"args.clip_scaling: {args.clip_scaling}")
		print(f"tau: {tau}", flush=True)
		
		return Clipping(tau=tau, n_iter=3)
		
	if args.agg == "byro":
		
		assert args.clip_tau > 0, "Clipping radius must be positive."

		is_personalization	 = (args.target != "1111111111")
		# is_dynamic_byzantine = ((args.client_delta > 0.0) and (args.f == 0))
		
		if is_personalization:
			
			tau0 = 3.0
			
		else:
			
			# Byzantine: tau0 = max || x0 - xi ||
			
			tau0 = None
		
		return ByRoAdapter(
		
			device			  = device,
			server			  = server,
			model 			  = model,
			loss_func		  = loss_func,
			tau0		  	  = tau0,
			validation_loader = validation_loader,
			gVonly			  = args.gVonly,
			beta_gV			  = args.beta_gV
			
		)

	if args.agg == "rfa": return RFA(T=8)
	if args.agg == "tm":  return TM(b=args.f)
	
	if args.agg == "krum":
		
		T = int(np.ceil(args.n / args.bucketing)) if args.bucketing > 0 else args.n
		
		return Krum(n=T, f=args.f, m=1)

	raise NotImplementedError(args.agg)

#==========================================================================================

def plot_projected_data(projected_data, n_byzantine=5):
	
	r"""
		Plot 1D projected data points, with last `n_byzantine` points
		marked as Byzantine (pink crosses) and the rest as honest (grey bubbles).
		
		Args:
			projected_data (numpy.ndarray): 1D array of projected points.
			n_byzantine (int): Number of Byzantine points at the end of the data.
	"""
	
	n_total = len(projected_data)
	honest_data = projected_data[:n_total - n_byzantine]
	byzantine_data = projected_data[n_total - n_byzantine:]

	# Plot honest data (grey bubbles)
	
	plt.figure(figsize=(8, 4))
	plt.scatter(honest_data, np.zeros_like(honest_data), color='grey', alpha=0.7, label='Honest', s=50)

	# Plot Byzantine data (pink crosses)
	
	plt.scatter(byzantine_data, np.zeros_like(byzantine_data), color='pink', marker='x', s=70, label='Byzantine')

	# Adding labels and title
	
	plt.xlabel("Data Index")
	plt.ylabel("Projected Value")
	plt.title("Scatter Plot of 1D Projected Data Points")
	plt.legend()
	plt.grid(True)
	plt.show()

def sklearn_svd_based_pca(tensors, args):
	
	# Convert the list of torch tensors to a single numpy array for PCA
	# Move tensors to CPU and convert to numpy for PCA
	
	data = torch.stack([tensor.cpu() for tensor in tensors]).numpy()	# Shape: (n, d)
	data = data[~np.isnan(data).any(axis=1)]							# Shape: (k, d) with k <= n
	if data.shape[0] < (int(args.n) - int(args.f)): return None			# Invalid input vectors
	projected_data = PCA(n_components=1).fit_transform(data)			# Shape: (k, 1)

	if 0:
		print(f"tensors: {type(tensors)} with {len(tensors)}")
		print(f"data: {type(data)} with {data.shape}")
		print(f"projected_data: {type(projected_data)} with {projected_data.shape}")
	
	return projected_data

_z_ = None	# the principal direction for online power iterations

def bucketing_wrapper(args, aggregator, s):
	
	r"""
		Applies Locally-sensitive Bucketing (LSB) on input vectors and aggregates within each bucket.
	"""

	def aggr_locally_sensitive(inputs):
		
		global _z_
		
		if 0:
			
			print(f"\ttype(inputs): {type(inputs)}")
			for i in range(len(inputs)):
				print(f"\t\ttype(inputs[{i}]): {type(inputs[i])} ", end='')
				print(f"with shape {inputs[i].shape}")
		
		if not isinstance(inputs, list) or len(inputs) == 0:
			
			raise ValueError("Error: 'inputs' must be a non-empty list of Torch tensors.")
				
		# Step 1: Project inputs to 1D

		if args.projection_idea == 0:
			
			# online PCA
			
			X 	 = torch.stack(inputs)					# Shape: (n, d)
			mask = ~torch.isnan(X).any(dim=1)			# Boolean mask where rows without NaN are True
			X	 = X[mask]								# Remove rows with NaN values
			
			if X.size(0) < (int(args.n) - int(args.f)):
				
				_PROJECTED_DATA_ = None					# Invalid input vectors
				
			else:
			
				mean_X	= X.mean(dim=0, keepdim=True)	# Shape: (1, d)
				A		= X - mean_X					# Shape: (n, d)
				
				if aggregator.mini_batch_cnt > 0:
					
					for generic_it in range(args.generic_it):
						
						_z_ = A.T @ (A @ _z_)
						_z_ /= torch.linalg.norm(_z_)
					
				else:
					
					if _z_ is None:
						
						# consuming the very first mini-batch
						
						_z_ = torch.randn(inputs[0].shape, dtype=inputs[0].dtype, device=inputs[0].device)
						_z_ /= torch.linalg.norm(_z_)
						
					else: raise ValueError("_z_ has already been set values.")
					
					for warm_it in range(args.proj_warm_it):
						
						_z_ = A.T @ (A @ _z_)
						_z_ /= torch.linalg.norm(_z_)
				
				_PROJECTED_DATA_ = X @ _z_	# Shape: (n, 1)

		elif args.projection_idea == 999:
			
			# sklearn default SVD-based PCA
			
			_PROJECTED_DATA_ = sklearn_svd_based_pca(inputs, args)
			
		else: raise NotImplementedError(args.projection_idea)
		
		if _FLAG_DATAFY_CLIENT_MESSAGES_IN_ONE_DIMENSION_:
			
			savemat(f"{_LOG_DIR_GLOBAL_IN_UTILS_DOT_PY_}\\mini_batch_cnt_{aggregator.mini_batch_cnt:05d}_in_1D.mat",
				{'projected_messages': _PROJECTED_DATA_.cpu().numpy()}
			)
		
		if _PROJECTED_DATA_ is None:
			
			return aggregator(inputs)
		
		if isinstance(_PROJECTED_DATA_, torch.Tensor):

			cpu_np_z = _PROJECTED_DATA_.cpu().numpy()  	# temporary implementation approach
			
		elif isinstance(_PROJECTED_DATA_, np.ndarray):
			
			cpu_np_z = _PROJECTED_DATA_					# from sklearn
		
		# Step 2: Calculate median of the projected data
		
		v_med = np.median(cpu_np_z)
		
		# Step 3: Compute absolute differences from the median
		
		differences = np.abs(cpu_np_z - v_med)
		
		# Step 4: Calculate threshold (h) as the median of these absolute differences
		
		h = float(np.median(np.unique(differences)) * s)
		
		# if np.isnan(h): return aggregator(inputs)
		
		# Initialize dictionary to store bucket indices and corresponding vectors
		
		buckets = {}

		# Step 5: Assign each input vector to a bucket based on its signed distance from the median
		
		for i, (x_proj, x_vec) in enumerate(zip(cpu_np_z, inputs)):
			
			delta = x_proj - v_med
			
			try:
				
				if delta >= 0:
					
					bucket_index = int(np.floor(delta / h)) + 1
					
				else:
					
					bucket_index = - (int(np.floor(-delta / h)) + 1)
					
			except:
				
				print(f"\ninputs:")
				
				for i in range(int(args.n)):
					
					print(f"[client:{i:03d}] ", end='')
					print(inputs[i])

			# Add the vector to the appropriate bucket in the dictionary
			
			if bucket_index not in buckets:
				
				buckets[bucket_index] = []
				
			buckets[bucket_index].append(x_vec)

		# Step 6: Aggregate within each bucket and store in reshuffled_inputs
		
		reshuffled_inputs = []
		
		for bucket_vectors in buckets.values():
			
			# Stack bucket vectors to perform mean on GPU
			
			bucket_stack = torch.stack(bucket_vectors)	# Shape: (bucket_size, d)
			g_bar = bucket_stack.mean(dim=0)			# Mean along the first dimension
			reshuffled_inputs.append(g_bar)
		
		if 0:
			
			print(f"reshuffled_inputs: {type(reshuffled_inputs)}. {len(reshuffled_inputs)}")
			
			for i in range(len(reshuffled_inputs)):
				
				print(f"\t\ttype(inputs[{i}]): {type(inputs[i])} ", end='')
				print(f"with shape {inputs[i].shape}")
				
		# print(f"- aggregator: {aggregator}")

		return aggregator(reshuffled_inputs)
		
	def aggr_random_shuffle(inputs):
		
		indices = list(range(len(inputs)))
		np.random.shuffle(indices)

		T = int(np.ceil(args.n / int(s)))

		reshuffled_inputs = []
		
		for t in range(T):
			
			indices_slice = indices[t * int(s) : (t + 1) * int(s)]
			g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
			reshuffled_inputs.append(g_bar)
			
		return aggregator(reshuffled_inputs)

	if args.bucketing_idea == 0:
		
		return aggr_random_shuffle
	
	elif args.bucketing_idea == 1:
		
		return aggr_locally_sensitive
	
	else:
		
		raise NotImplementedError(
			f"args.bucketing_idea == {args.bucketing_idea}"
		)

def get_aggregator(args, device, model, loss_func, server,
	validation_loader: torch.utils.data.DataLoader = None):
	
	aggr = _get_aggregator(args, device, model, loss_func, server, validation_loader)
	
	# args: `--bucketing 0` means no pre-aggregation by default
	
	if args.bucketing == 0: return aggr

	return bucketing_wrapper(args, aggr, args.bucketing)

#------------------------------------------------------------------------------------------

def get_sampler_callback(args, rank, flag_gV = False):
	
	r"""
		Get sampler based on the rank of a worker.
		The first `n - f` workers are good, and the rest are Byzantine
	"""
	
	if not flag_gV:		# applies to train and test dataset
		
		n_good = args.n - args.f
		
		if rank >= n_good:	# Byzantine workers
			
			return lambda x: DistributedSampler(
			
				num_replicas = n_good,
				rank		 = rank % (n_good),
				shuffle		 = True,
				dataset		 = x,
				
			)
			
		else:	# honest workers

			return lambda x: NONIIDLTSampler(
			
				alpha		 = not args.noniid,
				beta		 = 0.5 if args.LT else 1.0,
				num_replicas = n_good,
				rank		 = rank,
				shuffle		 = True,
				dataset		 = x,
				
			)
		
	else:	# applies to validation dataset
		
		return lambda x: NONIIDLTSampler(
		
			alpha		 = True,	# iid
			beta		 = 1,		# no longtailness
			num_replicas = 1,		# the sole server
			rank		 = 0,		# id of the server
			shuffle		 = True,
			dataset		 = x,
			
		)

#------------------------------------------------------------------------------------------

def get_test_sampler_callback(args):
	
	# This alpha argument that determines iid is
	# not important as there is only 1 replica
	
	return lambda x: NONIIDLTSampler(
	
		alpha		 = True,
		beta		 = 0.5 if args.LT else 1.0,
		num_replicas = 1,
		rank		 = 0,
		shuffle		 = False,
		dataset		 = x,
		
	)

#------------------------------------------------------------------------------------------

def initialize_worker(
	args,
	trainer,
	worker_rank,
	model,
	optimizer,
	loss_func,
	device,
	train_batch_size,
	kwargs):
	
	train_loader = mnist(
	
		data_dir		 = DATA_DIR,
		train			 = True,		# torchvision-level train-test split
		download		 = True,
		batch_size		 = train_batch_size,
		sampler_callback = get_sampler_callback(args, worker_rank),
		**kwargs,
		
	)

	if worker_rank < args.n - args.f:
		
		return MomentumWorker(
			momentum	 = args.momentum,
			data_loader  = train_loader,
			model		 = model,
			loss_func	 = loss_func,
			device		 = device,
			optimizer	 = optimizer,
			client_delta = args.client_delta,
			client_id	 = worker_rank,
			**kwargs,
		)

	if args.attack == "BF":
		
		return BitFlippingWorker(
			data_loader = train_loader,
			model		= model,
			loss_func	= loss_func,
			device		= device,
			optimizer	= optimizer,
			client_id	= worker_rank,
			**kwargs,
		)

	if args.attack == "LF":
		
		return LableFlippingWorker(
			revertible_label_transformer = lambda target: 9 - target,
			data_loader	= train_loader,
			model		= model,
			loss_func	= loss_func,
			device		= device,
			optimizer	= optimizer,
			client_id	= worker_rank,
			**kwargs,
		)

	if args.attack == "mimic":
		
		attacker = MimicVariantAttacker(
			warmup		= args.mimic_warmup,
			data_loader	= train_loader,
			model		= model,
			loss_func	= loss_func,
			device		= device,
			optimizer	= optimizer,
			client_id	= worker_rank,
			**kwargs,
		)
		attacker.configure(trainer)
		return attacker

	if args.attack == "IPM":
		
		attacker = IPMAttack(
			epsilon		= 0.1,
			data_loader	= train_loader,
			model		= model,
			loss_func	= loss_func,
			device		= device,
			optimizer	= optimizer,
			client_id	= worker_rank,
			**kwargs,
		)
		attacker.configure(trainer)
		return attacker

	if args.attack == "ALIE":
		
		attacker = ALittleIsEnoughAttack(
			n			= args.n,
			m			= args.f,
			# z=1.5,
			data_loader = train_loader,
			model		= model,
			loss_func	= loss_func,
			device		= device,
			optimizer	= optimizer,
			client_id	= worker_rank,
			**kwargs,
		)
		attacker.configure(trainer)
		return attacker

	raise NotImplementedError(f"No such attack {args.attack}")

torch.set_printoptions(sci_mode=True, precision=2)

#==========================================================================================

def test_validation_loader(
	kwargs,
	data_dir,
	train,
	flag_vis,
	flag_sim_epoch,
	download = True,
	batch_size = 10,
	validation_seed = 0):
	
	validation_loader_for_training = mnist_validation_dataset_loader(

		data_dir		 = DATA_DIR,
		train			 = train,
		download		 = download,
		batch_size		 = batch_size,
		seed			 = validation_seed,
		**kwargs,
		
	)
	
	# 1. Verify the Total Number of Batches
	
	print(f"")
	print(f"======================================================================")

	# 2: Visualization of a mini-batch
	
	if flag_vis:
	
		# Specify the batch number to inspect (zero-based indexing)
		
		batch_number = 0  # First batch

		# Retrieve the specified batch
		
		try:
			
			images, labels = mnist_validation_retrieve_batch(
				validation_loader_for_training, batch_number
			)
			
			print(f"Retrieved Batch {batch_number}:")
			print(f" - Images shape: {images.shape}")  # Expected: [60, 1, 28, 28]
			print(f" - Labels shape: {labels.shape}")  # Expected: [60]
			
		except ValueError as e:
			
			print(e)
			# Handle the error as needed
			images, labels = None, None

		# Proceed only if the batch was successfully retrieved
		
		if images is not None and labels is not None:
			
			# Sort the batch by labels in ascending order
			
			sorted_images, sorted_labels = mnist_validation_sort_batch_by_labels(
				images, labels)

			# Plot the sorted batch
			
			mnist_validation_plot_images(
				sorted_images,
				sorted_labels,
				cols=10,
				figsize_per_image=(2, 2),
				window_title=f"Sample Images from Batch {batch_number}"
			)
		
		print(f"")

	# 3. Iterate Through Batches to Inspect Shapes and Label Distributions
	
	if flag_sim_epoch:
		
		print("Inspecting the batches:")
		
		for batch_idx, (images, labels) in enumerate(validation_loader_for_training, 1):
			
			print(
				f"Batch {batch_idx:04d}. "
				f"shape: {images.shape}. "	# Expected: [batch_size, 1, 28, 28]
				f"label dist. {torch.bincount(labels)}"
			)
			
			#if batch_idx == 3: break
			
		sorted_distribution = mnist_validation_get_class_distribution(
				validation_loader_for_training
			)
			
		print(f"\nOverall class distribution up to Batch:")
		
		for i, (k, v) in enumerate(sorted_distribution.items()):
			print(f"[#{i:02d}] class {k} counts {v}")
			
		print(f"")
	
	print(f"dataset:				 {validation_loader_for_training.dataset}")
	print(f"batch_size:			  {batch_size}")
	print(f"Total number of batches: {len(validation_loader_for_training)}")
	print(f"")
	
	print(f"======================================================================")
	print(f"")

#==========================================================================================

def serialize_byzantine_history(trainer: ParallelTrainer, output_dir: str):
	
	r"""
		Serializes the `self.history_is_byzantine` and 
		`trainer.history_byzantine_behavior` to JSON files.
		
		Args:
			output_dir (str): Directory where the serialized files will be saved.
	"""
	
	
	os.makedirs(output_dir, exist_ok=True)	# Ensure the output directory exists
	
	# File paths for serialization
	
	is_byzantine_path		= os.path.join(output_dir, "history_is_byzantine.json")
	byzantine_behavior_path = os.path.join(output_dir, "history_byzantine_behavior.json")
	
	# Serialize the history lists to JSON
	
	with open(is_byzantine_path, "w") as f:
		
		json.dump(trainer.history_is_byzantine, f, indent=4)
	
	with open(byzantine_behavior_path, "w") as f:
		
		json.dump(trainer.history_byzantine_behavior, f, indent=4)

#==========================================================================================

def main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH, target_classes = None):
	
	if 0:
		
		print(f"torch.__version__: {torch.__version__}")
		return
	
	global _LOG_DIR_GLOBAL_IN_UTILS_DOT_PY_
	
	_LOG_DIR_GLOBAL_IN_UTILS_DOT_PY_ = LOG_DIR	# @ParallelTrainer

	initialize_logger(LOG_DIR)

	if args.use_cuda and not torch.cuda.is_available():
		
		print("=> There is no cuda device!!!!")
		device = "cpu"
		args.use_cuda = False
		
	else:
		
		device = torch.device("cuda" if args.use_cuda else "cpu")
		
	# kwargs = {"num_workers": 1, "pin_memory": True} if args.use_cuda else {}
	
	kwargs = {"pin_memory": True} if args.use_cuda else {}
	
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	model = Net().to(device)

	# Each optimizer contains a separate `state` to store info like `momentum_buffer`
	
	optimizers = [torch.optim.SGD(model.parameters(), lr=LR) for _ in range(args.n)]
	server_opt = torch.optim.SGD(model.parameters(),  lr=LR)

	loss_func = F.nll_loss

	metrics = {"top1": top1_accuracy}

	server  = TorchServer(
		optimizer	= server_opt,
		device		= device,
		model		= model,
		loss_func   = loss_func
	)
	
	if False:
		
		test_validation_loader(
			kwargs,
			DATA_DIR,
			train			= True,
			flag_vis		= True,
			flag_sim_epoch  = False,
			batch_size		= 10,
			validation_seed = args.seed
		)

		return
	
	validation_loader_for_training = mnist_validation_dataset_loader(

		data_dir		 = DATA_DIR,
		train			 = True,
		download		 = True,
		batch_size		 = args.val_set_sz,
		seed			 = 0,					# reproducibility
		target_classes	 = target_classes,		# Pass target classes to filter
		**kwargs,
		
	)
	
	trainer = ParallelTrainer(
		server					= server,
		aggregator				= get_aggregator(
									args, device, model, loss_func,
									server, validation_loader_for_training),
		pre_batch_hooks			= [],
		post_batch_hooks		= [],
		max_batches_per_epoch	= MAX_BATCHES_PER_EPOCH,
		log_interval			= args.log_interval,
		log_directory			= LOG_DIR,
		args					= args,
		metrics					= metrics,
		use_cuda				= args.use_cuda,
		debug					= False,
		validation_loader		= validation_loader_for_training)
	
	trainer.print_args()
	
	test_loader = mnist(
	
		data_dir		 = DATA_DIR,
		train			 = False,
		download		 = True,
		batch_size		 = _TEST_BATCH_SIZE_,
		sampler_callback = get_test_sampler_callback(args),
		**kwargs,
		
	)

	evaluator = DistributedEvaluator(
		model		= model,
		data_loader	= test_loader,
		loss_func	= loss_func,
		device		= device,
		metrics		= metrics,
		use_cuda	= args.use_cuda,
		debug		= False,
	)
	
	for worker_rank in range(args.n):
		
		worker = initialize_worker(
			args,
			trainer,
			worker_rank,
			model			 = model,
			optimizer		 = optimizers[worker_rank],
			loss_func		 = loss_func,
			device			 = device,
			train_batch_size = int(args.train_mb_sz),
			kwargs			 = {},
		)
		
		trainer.add_worker(worker)
		
	trainer.debug_logger.info("")

	if not args.dry_run:
		
		#--------------------------------------------------------------------------------------
		# 2025-01-20 (selective learning, intermittent byzantine)
		#......................................................................................
		
		if ((args.client_delta > 0.0) and (args.f == 0)):
			
			trainer.prepareByzantines()
			
			assert trainer.IB != None, (
				f"(args.client_delta == {args.client_delta} > 0.0)\n"
				f"& (trainer.IB == None)"
			)
			
			trainer.debug_logger.info(
				f">> args.client_delta: {args.client_delta}.\n"
				f"\ttrainer.IB:\n"
			)
			
			if 0:
				
				for attr, value in trainer.IB.__dict__.items():
					
					if attr == "args": continue
					
					trainer.debug_logger.info(f"\t\t{attr}: {value}")
					
				trainer.debug_logger.info(f"")
			
		#......................................................................................
		
		for epoch in range(1, EPOCHS + 1):
			
			#................................................................................
			# 2025-01-20 (selective learning): Aggregate classwise accuracy, `eval_results`
			#................................................................................
			
			#the `log_train` method is called inside with `eval_results = None`
			
			trainer.train(epoch)
			
			# Evaluation results are logged directly within the evaluator
			
			eval_results = evaluator.evaluate(epoch)
			
			# trainer.log_train(
			# 	progress	 = None,	# Not needed for evaluation-only logging
			# 	batch_idx	 = None,	# Not needed for evaluation-only logging
			# 	epoch		 = epoch,
			# 	results		 = None,	# Not used for evaluation logging
			# 	eval_results = eval_results,
			# )
			
			# Update worker data loaders for next epoch
			
			trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))
			
			if isinstance(trainer.aggregator, ByRoAdapter):
				
				if trainer.aggregator.tau is not None:
				
					if args.f > 0:
						
						to_print = (
							f"\ttau-G: {format_tensor_to_print(trainer.aggregator.tau[:int(args.n - args.f)])}\n"
							f"\ttau-B: {format_tensor_to_print(trainer.aggregator.tau[-int(args.f):])}\n"
						)
						
						trainer.debug_logger.info(to_print)
						
					else:
						
						to_print = (
							f"\ttau-G: {format_tensor_to_print(trainer.aggregator.tau[:int(args.n)])}\n"
						)
						trainer.debug_logger.info(to_print)
						
		# When experimenting on dynamic Byzantine behaviors,
		
		if ((args.client_delta > 0.0) and (args.f == 0)):
			
			serialize_byzantine_history(trainer, LOG_DIR)			
			
		if isinstance(trainer.aggregator, ByRoAdapter):
			
			adipmb = (
				float(trainer.aggregator.gd_tot_cnt)
				/ float(trainer.aggregator.mini_batch_cnt)
			)
			
			trainer.debug_logger.info(
				f">> average descent iterations per mini-batch "
				f"for adaptive centered clipping: {round(adipmb):d}\n"
			)
