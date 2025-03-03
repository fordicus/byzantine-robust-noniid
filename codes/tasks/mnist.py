import 	torch, numpy
import 	torch.nn as nn
import 	torch.nn.functional as F
from	typing import Callable, Optional, Dict		# loss_func
from	torchvision import datasets, transforms
from	collections import defaultdict
import	matplotlib.pyplot as plt
import	matplotlib, math
from	..aggregator.clipping import Omega
from	..sampler import PatchedMNIST, StratifiedBatchSampler

_MNIST_NORMALIZE_MEAN_ = 0.1307
_MNIST_NORMALIZE_STD_  = 0.3081

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

class Net(nn.Module):
	
	#--------------------------------------------------------------------------------------
	# the definition of the model layers
	#......................................................................................
	
	def __init__(self):
		
		super(Net, self).__init__()
		
		self.conv1		= nn.Conv2d(1, 32, 3, 1)	# the first convolutional layer
		self.conv2		= nn.Conv2d(32, 64, 3, 1)	# the second convolutional layer
		self.dropout1	= nn.Dropout(0.25)			# the first dropout layer
		self.dropout2	= nn.Dropout(0.5)			# the second dropout layer
		self.fc1		= nn.Linear(9216, 128)		# the first fully connected layer
		self.fc2		= nn.Linear(128, 10)		# the output layer
		
		print(f"\n>> d: {sum(p.numel() for p in self.parameters())}")

	#--------------------------------------------------------------------------------------
	# forward computation that does not interfere with any internal status of the model
	#......................................................................................

	def __loss_stateless__(self,		# loss_func: [[pred, target], loss]
		x:				torch.Tensor,	# the input mini-batch tensor
		y:				torch.Tensor,	# the ground truth tensor
		loss_func:		Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
		theta:			Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
		# a scalar tensor representing the loss
		
		r"""
			Compute the loss in a stateless manner using functional_call.
			If [theta] is provided, it overrides the current model parameters.

			Args:
				loss_func (callable): 	the loss function to compute the loss
				x (torch.Tensor):		the input mini-batch
				y (torch.Tensor):		the ground truth target for the input data
				theta (dict, optional): a dictionary of model parameters to use
										instead of the current parameters

			Returns:
				torch.Tensor: the computed loss
		"""
		
		state = {
			**(theta if theta is not None else dict(self.named_parameters())),
			**dict(self.named_buffers())
		}

		# pred = torch.nn.utils.stateless.functional_call(self, state, (x,))
		pred = torch.func.functional_call(self, state, (x,))

		return loss_func(pred, y)

	#--------------------------------------------------------------------------------------
	# the forward computation to achieve [tau.grad]
	#......................................................................................

	def __forward_with_tau__(self,
		X,				# client messages, shape [n, d]
		v,				# the robust agg., shape [d]
		tau,			# clipping radii,  shape [n]
		eta,			# the stepsize for nn params
		loss_func,		# the definition of loss for nn
		x,				# input mini-batch dataset
		y):				# the ground truth for x
		
		#..................................................................................
		# step 1. compute the centered clipping, [agg]
		#..................................................................................
		
		diffs = X - v												# [n, d]
		norms = torch.norm(diffs, dim = 1, keepdim = True) + 1e-7	# [n, 1]
		frac  = tau.view(-1, 1) / norms								# [n, 1]
		clip  = torch.clamp(frac, max=1.0)							# [n, 1]
		summ  = diffs * clip.expand_as(diffs)						# [n, d]
		agg	  = v + summ.mean(dim=0)								# [d]
		
		#..................................................................................
		# step 2. forward propagation using [agg]
		#..................................................................................
		
		head	  = 0
		theta_new = {}
		
		for name, theta in self.named_parameters():
			
			numel			=  theta.numel()
			theta_new[name]	=  theta - eta * agg[head : head + numel].view_as(theta)
			head			+= numel
		
		return self.__loss_stateless__(
			x = x, y = y, loss_func = loss_func, theta = theta_new
		)

	#--------------------------------------------------------------------------------------
	
	def forward(self, x,			# input dataset, 			e.g., val_data
		y				= None,		# the ground truth for x,	e.g., val_target
		tau				= None,		# clipping radii, 			shape [n]
		X				= None,		# the client messages,		shape [n, d]
		v				= None,		# the robust aggregation, 	shape [d]
		stopping_eps	= None,		# the max-norm threshold for gradient w.r.t. tau
		max_iter		= None,		# the maximum iterations for tau
		max_back		= None,		# the backtracking max iters
		omega			= None,		# the stepsize for tau
		eta				= None,		# the stepsize for nn params
		loss_func		= None,		# the definition of loss
		loss_prev		= None,		# the loss function value
		verbose_level	= 0):
		
		if (y 				is not None and
			tau 			is not None and
			X 				is not None	and
			v 				is not None	and
			stopping_eps	is not None	and
			max_iter		is not None and
			max_back		is not None and
			omega 			is not None and
			eta 			is not None	and
			loss_func		is not None	and
			loss_prev		is not None and
			verbose_level	is not None):
			
			# deprecated clipping radii optimization via validation loss
			
			if verbose_level >= 1: print(f"\n- forward invoked by ByRoAdapter")
			
			FLAG_EPSILON_GRADIENT	= False
			FLAG_VAL_LOSS_OPTIMIZED = False
			FLAG_STEPSZ_TOO_SMALL	= False
			
			PRINT_DESCENT_ITERS				= 0
			PRINT_BACKTRACKING_ITERS_TOTAL_ = 0
			
			for _i_ in range(max_iter):
				
				PRINT_BACKTRACKING_ITERS = 0
		
				#..........................................................................
				# compute the gradient of the mini-batch validation loss w.r.t. tau
				#..........................................................................
				
				if tau.grad is not None: tau.grad.zero_()
				
				loss_tmp = self.__forward_with_tau__(X, v, tau, eta, loss_func, x, y)
				loss_tmp.backward()
				
				loss_prev.set(loss_tmp.item())	# this may be problematic
				
				del loss_tmp
				
				grad_tau = tau.grad.clone().detach()
				tau_prev = tau
				
				if 1: grad_norm = torch.max(torch.abs(grad_tau)).item()
				else: grad_norm = torch.norm(grad_tau).item()
				
				if grad_norm <= stopping_eps and _i_ > 0:
					
					FLAG_EPSILON_GRADIENT = True
					
					break
				
				PRINT_TAU_BEFORE = tau.cpu().detach().numpy()
				
				val_loss_before	 = loss_prev.item()
				
				#..........................................................................
				# monotonic gradient descent for tau (backtracking)
				#..........................................................................
				
				with torch.no_grad():
					
					if max_back <= 0:			# no backtracking
						
						tau = tau - omega.item() * grad_tau
						
						tau = torch.where(		# projection
						
							tau < 0,
							torch.tensor(1e-7, dtype=tau.dtype),
							tau
						)
							
						loss = self.__forward_with_tau__(
						
							X, v, tau, eta, loss_func, x, y
							
						).item()
						
						loss_prev.set(loss)
						
						del loss
						
					else:						# backtracking
						
						for _j_ in range(max_back):
							
							tau = tau - omega.item() * grad_tau
							
							tau = torch.where(	# projection
								
								tau < 0,
								torch.tensor(1e-7, dtype=tau.dtype),
								tau
							)
							
							loss = self.__forward_with_tau__(
								X, v, tau, eta, loss_func, x, y
							).item()
							
							if loss < loss_prev.item(): 
								
								loss_prev.set(loss)
								
								break
								
							else:
								
								tau = tau_prev
								omega.halve()
								
								# if omega.item() <= 1e+2:
								# 	
								# 	FLAG_STEPSZ_TOO_SMALL = True
								# 	
								# 	break
								
								PRINT_BACKTRACKING_ITERS		+= 1
								PRINT_BACKTRACKING_ITERS_TOTAL_ += 1
								
							del loss
							
					# if omega.item() <= 1e+2:
					# 	
					# 	FLAG_STEPSZ_TOO_SMALL = True
					# 	
					# 	break
				
				tau	= tau.detach().requires_grad_()
				v	= v.detach()
				
				PRINT_TAU_AFTER = tau.cpu().detach().numpy()
				
				val_loss_after = loss_prev.item()
				val_loss_delta = val_loss_after - val_loss_before
				
				if verbose_level >= 2:
					
					print("")
					
					print(f"\t[{_i_ + 1:02d}] grad_norm:\t\t{grad_norm:.2e}")
					print(f"\t[{_i_ + 1:02d}] tau.grad:\t\t{format_tensor_to_print(grad_tau.cpu())}")
					
					print(f"\t[{_i_ + 1:02d}] tau (before):\t{format_tensor_to_print(PRINT_TAU_BEFORE)}")
					print(f"\t[{_i_ + 1:02d}] tau (after): \t{format_tensor_to_print(PRINT_TAU_AFTER)}")
					
					print(f"\t[{_i_ + 1:02d}] omega (stepsize):\t{omega.item():.2e}")
					
					print(f"\t[{_i_ + 1:02d}] val_loss (before): {val_loss_before:.2e}")
					print(f"\t[{_i_ + 1:02d}] val_loss (after) : {val_loss_after:.2e}. "
						f"Δ: {val_loss_delta:.2e} "
						f"({PRINT_BACKTRACKING_ITERS:02d} backtracking steps invoked)",
						end = ""
					)
				
				PRINT_DESCENT_ITERS += 1
				
				if verbose_level >= 2 and PRINT_BACKTRACKING_ITERS >= max_back:
					
					print(f" WARNING: {PRINT_BACKTRACKING_ITERS} >= max_back\n ")
					
				elif verbose_level >= 2: print("")
				
				if verbose_level >= 2: print("")
				
				if val_loss_delta < 0 and abs(val_loss_delta) <= stopping_eps and _i_ > 0:
					
					FLAG_VAL_LOSS_OPTIMIZED = True
					
					break
			
			if verbose_level >= 1:
			
				print(
					f"\t[{PRINT_DESCENT_ITERS:02d}] descent iterations for tau; ", 
					f"[{PRINT_BACKTRACKING_ITERS_TOTAL_:02d}] cummulative backtracking steps"
				)
				
				print(f"\t[FLAG_EPSILON_GRADIENT]   {FLAG_EPSILON_GRADIENT}")
				print(f"\t[FLAG_VAL_LOSS_OPTIMIZED] {FLAG_VAL_LOSS_OPTIMIZED}")
				print(f"\t[FLAG_STEPSZ_TOO_SMALL]   {FLAG_STEPSZ_TOO_SMALL}", end = "\n\n")

			return tau

		# the standard forward path
		
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

#==========================================================================================

def mnist(
	data_dir,
	train,
	download,
	batch_size,
	shuffle			 = None,
	sampler_callback = None,
	dataset_cls		 = datasets.MNIST,
	drop_last		 = True,
	**loader_kwargs):

	# if `sampler_callback` is not None, `shuffle` is neglected
	
	if sampler_callback is not None and shuffle is not None:
		
		raise ValueError(
			f"mnist::sampler_callback: {sampler_callback}, "
			f"mnist::shuffle: {shuffle}"
		)

	dataset = dataset_cls(
	
		data_dir,
		train	  = train,
		download  = download,
		transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize((_MNIST_NORMALIZE_MEAN_,), (_MNIST_NORMALIZE_STD_,)),
			]
		),
		
	)

	sampler = sampler_callback(dataset) if sampler_callback else None
	
	return torch.utils.data.DataLoader(
	
		dataset,
		batch_size = batch_size,
		shuffle	   = shuffle,
		sampler	   = sampler,
		drop_last  = drop_last,
		**loader_kwargs,
		
	)

#==========================================================================================

def mnist_validation_dataset_loader(
	data_dir,
	train,
	download,
	batch_size,
	seed,
	target_classes = None,  # Optional parameter to specify target classes
	**loader_kwargs):

	dataset = PatchedMNIST(
	
		data_dir,
		train	  = train,
		download  = download,
		transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize((_MNIST_NORMALIZE_MEAN_,), (_MNIST_NORMALIZE_STD_,)),
			]
		),
		
	)
	
	#--------------------------------------------------------------------------------------
	# 2025-01-20 (selective learning): Aggregate classwise accuracy, `eval_results`
	#......................................................................................
	# If target_classes are provided, filter the dataset
	#......................................................................................
	
	if target_classes is not None:
		
		filtered_indices = [
			idx for idx, label in enumerate(dataset.targets)
			if label in target_classes
		]
		
		dataset.targets = dataset.targets[filtered_indices]
		dataset.data	= dataset.data[filtered_indices]

	sampler = StratifiedBatchSampler(dataset, batch_size, seed)
	
	return torch.utils.data.DataLoader(
	
		dataset,
		batch_sampler = sampler,
		**loader_kwargs,
		
	)

#------------------------------------------------------------------------------------------

def mnist_validation_retrieve_batch(
	data_loader: torch.utils.data.DataLoader,
	batch_number: int):
	
	r"""
		Retrieves the images and labels for a specified batch number.

		Args:
			data_loader (DataLoader): The DataLoader instance.
			batch_number (int): The zero-based batch number to retrieve.

		Returns:
			tuple: (images, labels) tensors if batch is found.

		Raises:
			ValueError: If batch_number is negative or exceeds total batches.
	"""
	
	if batch_number < 0:
		
		raise ValueError("batch_number should be >= 0")
		
	for batch_idx, (images, labels) in enumerate(data_loader):
		
		if batch_idx == batch_number:
			
			return images, labels
	
	raise ValueError(
		f"Batch number {batch_number} exceeds the total number of batches ({batch_idx})."
	)

#------------------------------------------------------------------------------------------

def mnist_validation_sort_batch_by_labels(
	images: torch.Tensor, labels: torch.Tensor):
	
	r"""
		Sorts images and labels in ascending order of labels.

		Args:
			images (torch.Tensor): Batch of images.
			labels (torch.Tensor): Corresponding labels.

		Returns:
			tuple: (sorted_images, sorted_labels) sorted in ascending order of labels.
	"""
	
	sorted_labels, sorted_indices = labels.sort()
	sorted_images = images[sorted_indices]
	return sorted_images, sorted_labels

#------------------------------------------------------------------------------------------

def mnist_validation_plot_images(
	images: torch.Tensor,
	labels: torch.Tensor,
	cols: int = 10,
	figsize_per_image: tuple = (2, 2),
	window_title: str = "Batch Images"):
	
	r"""
		Plots images in a grid with corresponding labels.

		Args:
			images (torch.Tensor): Batch of images to plot.
			labels (torch.Tensor): Corresponding labels.
			cols (int, optional): Number of columns in the grid. Defaults to 10.
			figsize_per_image (tuple, optional): Size of each image in inches. Defaults to (2, 2).
			window_title (str, optional): Title of the figure window. Defaults to "Batch Images".
	"""
	
	batch_size = images.size(0)
	rows = math.ceil(batch_size / cols)
	figsize = (cols * figsize_per_image[0], rows * figsize_per_image[1])

	fig, axes = plt.subplots(rows, cols, figsize=figsize)
	axes = axes.flatten()

	for idx in range(batch_size):
		ax = axes[idx]
		img = images[idx].squeeze().numpy()
		label = labels[idx].item()
		ax.imshow(img, cmap='gray')
		ax.set_title(f"Label: {label}", fontsize=8)
		ax.axis('off')

	# Hide any remaining subplots
	for idx in range(batch_size, len(axes)):
		axes[idx].axis('off')

	# Set figure window title based on matplotlib version
	if matplotlib.__version__ >= "3.4":
		fig.canvas.manager.set_window_title(window_title)
	else:
		fig.canvas.set_window_title(window_title)

	plt.tight_layout()
	plt.subplots_adjust(top=0.95)
	plt.show()
	plt.close(fig)

#------------------------------------------------------------------------------------------

def mnist_validation_get_class_distribution(
	data_loader: torch.utils.data.DataLoader,
	up_to_batch: int = None):
	
	r"""
		Computes the class distribution up to a specified batch number.

		Args:
			data_loader (DataLoader): The DataLoader instance.
			up_to_batch (int, optional): The zero-based batch number up to which to count.
										  If None, counts all batches.

		Returns:
			dict: Sorted class distribution.
	"""
	
	overall_distribution = defaultdict(int)

	for batch_idx, (images, labels) in enumerate(data_loader):
		
		if up_to_batch is not None and batch_idx > up_to_batch:
			break
			
		for label in labels:
			
			label_value = label.item()
			
			if isinstance(label_value, str):
				
				try:
					
					label_value = int(label_value)
					
				except ValueError:
					
					print(f"Non-integer label encountered: {label_value}. Skipping.")
					continue
					
			overall_distribution[label_value] += 1

	sorted_distribution = dict(sorted(overall_distribution.items()))
	return sorted_distribution

#------------------------------------------------------------------------------------------