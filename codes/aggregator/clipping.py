import torch, math, types
import torch.nn.functional as F
import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.io, os

from .base import _BaseAggregator
from .base import _BaseAsyncAggregator

debug_logger = logging.getLogger("debug")

#==========================================================================================
# Omega: backtracking-friendly number, previously used for initial exploration of A-CC
#------------------------------------------------------------------------------------------

class Omega:
	
	def __init__(self, value):
		
		self.value		   = value
		self.initial_value = value
		
	def halve(self):			   self.value *= 0.5
	def decay(self, factor = 0.9): self.value *= factor
	def item(self):				   return self.value
	def reset(self):			   self.value = self.initial_value
	
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
# ByRoAdapter (A-CC)
#------------------------------------------------------------------------------------------

class ByRoAdapter(_BaseAggregator):
	
	def __init__(self,
		device,
		server,
		model,
		loss_func,					# the definition of the loss function
		validation_loader: 			torch.utils.data.DataLoader = None,
		tau0			 = None,	# override the initial guess of tau with a scalar
		lambda_			 = 10.0,	# the approximation parameter for softmin
		omega		 	 = 7.5e+1,	# the stepsize for tau's gradient update
		stopping_eps	 = 1e-2,	# the threshold for gradient w.r.t. tau
		max_iter		 = 50,		# the maximum iterations for tau
		max_back		 = -1,		# positive numbers, e.g., 5, allows backtracking
		beta_gV			 = 0.9,		# the momentum weight for the validation gradient
		angular_term	 = False,	# flag:   maximize angular similarity
		angular_weight	 = 1.00,	# weight: the angular similarity term
		l2_square_term	 = False,	# flag:   ||tau||^2 regularization
		l2_square_weight = 1e-1,	# weight: ||tau||^2 regularization
		momentum_tau	 = False,	# flag:	  momentum optimizing tau
		beta_tau		 = 0.00,	# weight: momentum weight optimizing tau
		reuse_tau		 = False,	# initialize tau with the previous result
		vis_landscpe_2D	 = False,
		gVonly 			 = False,
		verbose_level	 = 0):
		
		self.device	= device		# the Torch tensors' computation device
		self.model	= model			# the definition of the neural network model
		
		self.gVonly	 = gVonly		# only the validation gradient gV is used for training
		self.beta_gV = beta_gV		# the momentum weight for the validation gradient
		self.gV		 = None			# the validation gradient (w/ or w/o momentum)
		
		self.dtype  = next(self.model.parameters()).dtype
		
		self.optimizer = server.optimizer
		
		self.eta = (
		
			self.optimizer.param_groups[0]['lr']
			
		)	# the stepsize for the neural network model
		
		self.d = sum(p.numel() for p in self.model.parameters())
		
		self.tau0 = tau0
		
		self.lambda_ = torch.tensor(
			lambda_, device = self.device, dtype = self.dtype
		)
		
		self.omega = Omega(omega)
			
		self.stopping_eps = stopping_eps
		self.max_iter	  = max_iter
		self.max_back	  = max_back
		
		self.loss_func	  = loss_func

		self.tau = None
		
		if validation_loader != None:
			
			self.validation_loader = validation_loader
			self.validation_iter   = iter(self.validation_loader)
		
		self.v = None		# the robust aggregation of the client messages, shape: [d]
		
		self.flag_maturity  = False
		
		self.angular_term   = angular_term
		self.angular_weight = angular_weight
		
		self.l2_square_term   = l2_square_term
		self.l2_square_weight = l2_square_weight
		
		self.momentum_tau = momentum_tau
		self.beta_tau	  = beta_tau
		
		if self.angular_term:
			
			print(
				f"\n>> A-CC maximizes angular similarity with "
				f"importance factor {self.angular_weight}"
			)
			
		if self.l2_square_term:
			
			print(
				f">> A-CC has L2-square regularization with "
				f"importance factor {self.l2_square_weight}"
			)
			
		if self.angular_term or self.angular_term: print(f"")
		
		self.reuse_tau		= reuse_tau
		
		self.gd_tot_cnt = 0		# all-time counts					 for tau descents
		self.gd_cum_cnt = 0		# per mini-batch counts				 for tau descents
		self.bk_cum_cnt	= 0		# per mini-batch backtracking counts for tau descents
		
		self.vis_landscpe_2D = vis_landscpe_2D
		self.verbose_level	 = verbose_level
		
		self.info = None
		
		server.aggregator = self	# @ server.py
		
		super(ByRoAdapter, self).__init__()

	#--------------------------------------------------------------------------------------
	# set_validation_data: to set the validation mini-batch to optimize tau for
	#......................................................................................

	def set_validation_data(self, val_data, val_target):
		
		self.val_data	= val_data.to(self.device)
		self.val_target	= val_target.to(self.device)

	#--------------------------------------------------------------------------------------
	# get mini-batch indefinitely using the next method
	#......................................................................................

	def get_next_batch(self):
		
		try:	# attempt to get the next batch
			
			data, target = next(self.validation_iter)
			
		except StopIteration:	# If exhausted, reinitialize the iterator as well
			
			self.validation_iter = iter(self.validation_loader)
			
			data, target = next(self.validation_iter)
			
		return data, target

	#--------------------------------------------------------------------------------------
	# the standard centered clipping
	#......................................................................................

	def _centered_clipping_(self, tau, v0, X):
		
		r"""
			Args:
			
				tau: (n,) Torch tensor, clipping radii
				v0:  (d,) Torch tensor, initial guess for CC
				X:   (n, d) Torch tensor, client messages

			Returns:
			
				agg: (d,) Torch tensor, aggregated vector
		"""
		
		n		= X.shape[0]
		diffs	= X - v0
		norms	= torch.sqrt(torch.sum(diffs**2, dim=1)) + 1e-7
		weights = torch.minimum(torch.ones_like(norms), tau / norms)
		agg		= v0 + (1 / n) * torch.sum(diffs * weights.unsqueeze(1), dim=0)

		return agg

	#--------------------------------------------------------------------------------------
	# compute the softmin function
	#......................................................................................
	
	def _softmin_(self, z):
		
		r"""
			Args:
			
				z:	 (n,) Torch tensor, clipping's operands
			
			Returns:
			
				val: (n,) Torch tensor, result of the _softmin_ function
		"""

		if True:	# a numerically stable approach

			return (
				-(1 / self.lambda_) * torch.logaddexp(-self.lambda_, -self.lambda_ * z)
			)
			
		else:		# an equivalent expression
		
			return (
				-(1 / self.lambda_)
				* torch.log(
					torch.exp(-self.lambda_) 
					+ torch.exp(-self.lambda_ * z)
				)
			)
	
	#--------------------------------------------------------------------------------------
	# compute the derivative of the softmin function
	#......................................................................................

	def _softmin_derivative_(self, z):
		
		r"""
			Args:
			
				z:	   (n,) Torch tensor, input values
			
			Returns:
			
				deriv: (n,) Torch tensor, the derivative of _softmin_
		"""
		
		if True:	# a numerically stable approach
			
			log_denom = torch.logaddexp(-self.lambda_, -self.lambda_ * z)
			
			return torch.exp(-self.lambda_ * z - log_denom)
			
		else:		# an equivalent expression
			
			exp_neg_lambda_z = torch.exp(-self.lambda_ * z)
			
			return (
				exp_neg_lambda_z
				/ ( torch.exp(-self.lambda_) + exp_neg_lambda_z )
			)
	
	#--------------------------------------------------------------------------------------
	# centered clipping using the softmin function
	#......................................................................................
	
	def _centered_clipping_soft_(self, tau, v0, X):
		
		r"""
			Args:
			
				tau:	 (n,) Torch tensor, clipping radii
				v0:		 (d,) Torch tensor, initial guess for centered clipping
				X:		 (n, d) Torch tensor, client messages

			Returns:
			
				agg:	 (d,) Torch tensor, aggregated result after centered clipping
		"""
		
		n		= X.shape[0]
		diffs	= X - v0														# (n, d)
		norms	= torch.sqrt(torch.sum(diffs**2, dim=1)) + 1e-7					# (n,)
		z		= tau / norms													# (n,)
		weights = self._softmin_(z)												# (n,)
		agg		= v0 + (1 / n) * torch.sum(diffs * weights.unsqueeze(1), dim=0)	# (d,)
		
		return agg
	
	#--------------------------------------------------------------------------------------
	# compute the gradient of phi(tau) w.r.t. tau for softmin clipping
	#......................................................................................
	
	def _get_grad_phi_softmin_(self, tau, X, v0):
		
		r"""
			Args:
			
				tau: (n,)	Torch tensor, clipping radii
				X:   (n, d) Torch tensor, client messages
				v0:  (d,)	Torch tensor, initial guess for aggregation

			Returns:
			
				grad_phi: (d, n) Torch tensor, gradient of phi with respect to tau
		"""
		
		n, d  = X.size()
		diffs = X - v0														# (n, d)
		norms = torch.norm(diffs, dim=1, keepdim=True) + 1e-7				# (n, 1)
		z	  = tau / norms.squeeze()										# (n,)

		softmin_deriv = self._softmin_derivative_(z).unsqueeze(1)			# (n, 1)
		
		grad_phi = (1 / n) * softmin_deriv * (diffs / norms)				# (n, d)

		return grad_phi.T													# (d, n)
	
	#--------------------------------------------------------------------------------------
	# compute the gradient using forward finite differences (helper)
	#......................................................................................
	
	def _finite_difference_gradient_(self, tau, X, v0, gV, h = 1e-2):
		
		r"""
			Args:
			
				tau:	 (n,) Torch tensor, clipping radii
				X:		 (n, d) Torch tensor, client messages
				v0:		 (d,) Torch tensor, initial guess for CC
				gV:		 (d,) Torch tensor, validation gradient
				h:		 Step size for finite differences
			
			Returns:
			
				grad_fd: (n,) Torch tensor, finite difference gradient
		"""

		grad_fd = torch.zeros_like(tau)
		psi_0	= self._evaluate_psi_(tau, gV, v0, X)
		
		for i in range(tau.size(0)):
			
			tau_perturbed	  = tau.clone()
			tau_perturbed[i] += h
			psi_perturbed	  = self._evaluate_psi_(tau_perturbed, gV, v0, X)
			grad_fd[i]		  = (psi_perturbed - psi_0) / h

		return grad_fd

	#--------------------------------------------------------------------------------------
	# compute the angle between two Torch tensors in degrees
	#......................................................................................
	
	def compute_angle_in_degrees(self, tensor1, tensor2):
		
		tensor1 = tensor1.float()
		tensor2 = tensor2.float()

		dot_product = torch.dot(tensor1, tensor2)

		norm1 = torch.norm(tensor1) + 1e-7
		norm2 = torch.norm(tensor2) + 1e-7

		if norm1 == 0 or norm2 == 0:
			
			raise ValueError("One of the tensors has zero magnitude, so the angle is undefined.")
		
		cos_theta = dot_product / (norm1 * norm2)
		
		cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
		
		angle_radians = torch.acos(cos_theta)
		angle_degrees = angle_radians * (180 / torch.pi)
		
		return angle_degrees.item()
		
	#--------------------------------------------------------------------------------------
	# compute the cosine similarity between gV and phi
	#......................................................................................

	def compute_cosine_similarity(self, gV, phi):
		
		r"""
			compute the angular similarity between gV and phi(tau).

			Args:
			
				gV:	   (d,) Torch tensor, validation gradient
				phi:   (d,) Torch tensor, aggregation

			Returns:
			
				c_tau: scalar, cosine similarity term c(tau)
		"""
		
		norm_gV	 = torch.norm(gV)  + 1e-7
		norm_phi = torch.norm(phi) + 1e-7

		return torch.dot(gV, phi) / (norm_gV * norm_phi)
		
	#--------------------------------------------------------------------------------------
	# compute the gradient of the angular similarity with respect to tau
	#......................................................................................

	def _get_grad_c_(self, tau, gV, X, v0):
		
		r"""
			Args:
			
				tau:	 (n,)   Torch tensor, clipping radii
				gV:		 (d,)   Torch tensor, validation gradient
				X:		 (n, d) Torch tensor, client messages
				v0:		 (d,)   Torch tensor, initial guess for aggregation

			Returns:
			
				grad_c:  (n,)	Torch tensor, gradient of c(tau) with respect to tau
		"""

		# compute phi(tau) using softmin centered clipping
		
		phi = self._centered_clipping_soft_(tau, v0, X)						# (d,)

		# compute norms and cosine similarity
		
		norm_gV  = torch.norm(gV)  + 1e-7
		norm_phi = torch.norm(phi) + 1e-7
		c_tau	 = torch.dot(gV, phi) / (norm_gV * norm_phi)

		# compute the gradient of phi(tau) w.r.t tau
		
		grad_phi = self._get_grad_phi_softmin_(tau, X, v0)					# (d, n)

		# precompute scaled tensors to avoid redundant computation
		
		scaled_gV  = gV / (norm_gV * norm_phi)								# (d,)
		scaled_phi = (c_tau / (norm_phi ** 2)) * phi						# (d,)

		# subtract scaled terms first, then perform matrix-vector multiplication
		
		diff_term = scaled_gV - scaled_phi									# (d,)
		grad_c	  = torch.matmul(diff_term, grad_phi)						# (n,)

		return grad_c														# (n,)
		
	#--------------------------------------------------------------------------------------
	# validate gradient of the angular similarity using finite differences (helper)
	#......................................................................................

	def finite_difference_grad_c(self, tau, gV, X, v0, epsilon = 1e-4):
		
		r"""
			Args:
			
				tau:	 (n,)   Torch tensor, clipping radii
				gV:	  	 (d,)   Torch tensor, validation gradient
				X:	   	 (n, d) Torch tensor, client messages
				v0:	  	 (d,)   Torch tensor, initial guess for aggregation
				epsilon: small scalar step size for finite differences

			Returns:
			
				grad_c_fd: (n,) Torch tensor, numerical gradient of c(tau)
		"""
		
		grad_c_fd = torch.zeros_like(tau)

		for i in range(tau.size(0)):
			
			tau_perturbed = tau.clone()
			
			# forward step
			
			tau_perturbed[i] += epsilon
			phi_plus = self._centered_clipping_soft_(tau_perturbed, v0, X)
			c_plus	 = self.compute_cosine_similarity(gV, phi_plus)
			
			# backward step
			
			tau_perturbed[i] -= 2 * epsilon
			phi_minus = self._centered_clipping_soft_(tau_perturbed, v0, X)
			c_minus	  = self.compute_cosine_similarity(gV, phi_minus)
			
			# numerical gradient approximation
			
			grad_c_fd[i] = (c_plus - c_minus) / (2 * epsilon)
		
		return grad_c_fd

	#--------------------------------------------------------------------------------------
	# compute psi(tau) = || gV - phi(tau) ||^2 using _centered_clipping_soft_
	#......................................................................................
	
	def _evaluate_psi_(self, tau, gV, v0, X):
		
		r"""
			Args:
			
				X:		 (n, d) Torch tensor, client messages
				v0:		 (d,) Torch tensor, initial guess for CC
				gV:		 (d,) Torch tensor, validation gradient
				tau:	 (n,) Torch tensor, clipping radii

			Returns:
			
				psi:	 scalar, value of the loss function
		"""
		
		phi = self._centered_clipping_soft_(tau, v0, X)
		
		psi = 0.5 * torch.sum((gV - phi) ** 2)
		
		if self.angular_term:
			
			psi -= (self.angular_weight * self.compute_cosine_similarity(gV, phi))
			
		if self.l2_square_term:
			
			reg = ((self.l2_square_weight / 2.0) * torch.sum(tau ** 2))
			
			# print(f"\t- reg: {reg}")
			
			psi += reg
		
		return psi
	
	#--------------------------------------------------------------------------------------
	# compute gradient of psi(tau) using subroutines
	#......................................................................................
	
	def _get_grad_psi_(self, tau, X, v0, gV):
		
		r"""
			Args:
			
				X:		 (n, d) Torch tensor, client messages
				v0:		 (d,) Torch tensor, initial guess for CC
				gV:		 (d,) Torch tensor, validation gradient
				tau:	 (n,) Torch tensor, clipping radii
			
			Returns:
			
				grad_tau: (n,) Torch tensor, gradient w.r.t. tau
		"""
		
		n	  = X.size(0)
		phi	  = self._centered_clipping_soft_(tau, v0, X)			# (d,)
		diffs = X - v0												# (n, d)
		norms = torch.norm(diffs, dim=1) + 1e-7						# (n,)
		z	  = tau / norms											# (n,)

		softmin_deriv = self._softmin_derivative_(z)				# (n,)
		
		dir_values = (							  					# (n,)
			torch.sum(
				diffs * (gV - phi).unsqueeze(0),					# (n, d)
				dim=1
			)	/ norms
		)

		grad_tau = (-1 / n) * dir_values * softmin_deriv			# (n,)
		
		if self.angular_term:
			
			# self._get_grad_c_() returns the negated gradient by definition
			
			grad_tau -= (self.angular_weight * self._get_grad_c_(tau, gV, X, v0))
			
		if self.l2_square_term:
			
			reg_grad = (self.l2_square_weight * tau)
			
			# print(f"\t- torch.norm(reg_grad): {torch.norm(reg_grad)}")
			
			grad_tau += reg_grad
		
		return grad_tau

	#--------------------------------------------------------------------------------------
	# compute the gradient of the validation loss with respect to model parameters,
	# and return it as a flattened Torch tensor
	#......................................................................................
	
	def _get_gV_(self):
		
		r"""
			Returns:
			
				gV (torch.Tensor): The gradient vector with shape (d,) flattened.
		"""
		
		if self.val_data   is None: raise RuntimeError("_get_gV_(): `self.val_data` is not set.")
		if self.val_target is None: raise RuntimeError("_get_gV_(): `self.val_target` is not set.")
		if self.loss_func  is None: raise RuntimeError("_get_gV_(): `self.loss_func` is not set.")
		
		gV = torch.zeros( self.d,
			device = self.device,
			dtype  = self.dtype
		)
		
		USE_MODEL = True
		
		if USE_MODEL:
			
			self.model.zero_grad()
			
			val_loss = self.model.__loss_stateless__(
			
				x		  = self.val_data,
				y		  = self.val_target,
				loss_func = self.loss_func
				
			)
			
			val_loss.backward()

			head = 0
			
			for theta in self.model.parameters():
				
				numel = theta.numel()
				
				if theta.grad is not None:
					
					gV[head : head + numel] = theta.grad.clone().detach().view(-1)
					
				head += numel
				
			assert head == self.d, f"Mismatch in flattened gradient size: expected {self.d}, got {head}"
			
		else:
		
			self.optimizer.zero_grad()
		
			val_loss = self.loss_func(self.model(self.val_data), self.val_target)
		
			val_loss.backward()

			head = 0
			
			for group in self.optimizer.param_groups:
				
				for theta in group['params']:
					
					numel = theta.numel()

					if theta.grad is not None:
						
						gV[head : head + numel] = theta.grad.clone().detach().view(-1)
					
					head += numel
					
			assert head == self.d, f"Mismatch in flattened gradient size: expected {self.d}, got {head}"
		
		del val_loss
		
		if self.gV != None:
			
			self.gV = (1 - self.beta_gV) * gV + self.beta_gV * self.gV
			
		else:
			
			self.gV = gV
		
		return self.gV.clone().detach()

	#--------------------------------------------------------------------------------------
	# optimize tau using backtracking gradient descent
	#......................................................................................
	
	def _optim_tau_(self, X, v0, gV):
		
		r"""
			Args:
			
				X:	 (n, d) Torch tensor, client messages
				v0:	 (d,  ) Torch tensor, initial guess for CC
				gV:	 (d,  ) Torch tensor, validation gradient
			
			Returns:
			
				tau: (n,) Torch tensor, optimized clipping radii
		"""
		
		with torch.no_grad():
		
			if self.reuse_tau and self.tau != None:
				
				# reuse tau from the previous aggregation round
				# no benefit observed at the moment
				
				tau = self.tau.clone().detach()
				
			else:
				
				# we provide an initial guess, otherwise
				
				if self.tau0 != None:
					
					# we want to override the initial tau with `self.tau0`
					
					tau = torch.full(
							(X.size(0),),
							self.tau0,
							dtype  = X.dtype,
							device = X.device
						)
					
				else:
					
					# under Byzantine setting, max||x0 - xi|| is beneficial
					
					diffs = torch.norm(X - v0.unsqueeze(0), dim = 1)
					
					tau = torch.full(
						(X.size(0),),
						torch.max(diffs),
						dtype  = X.dtype,
						device = X.device
					)
				
			if self.momentum_tau:
				
				momentum = torch.zeros_like(
					tau,
					device = self.device,
					dtype  = self.dtype
				)
			
			loss = self._evaluate_psi_(tau, gV, v0, X)
			
			self.gd_cum_cnt	   = 0
			self.bk_cum_cnt	   = 0
			
			self.flag_maturity = False

			if  self.verbose_level >= 2:
				
				print(f"")
				print(f"\tmax_iter:			{self.max_iter}")
				print(f"\tmax_back:			{self.max_back}")
				print(f"\tomega (stepsize):	{self.omega.item():.2e}")
				print(f"\tflag_maturity:    {self.flag_maturity}")
				print(f"")

			for i in range(self.max_iter):
				
				if self.flag_maturity: break

				tau_prev  = tau.clone()
				grad_tau  = self._get_grad_psi_(tau, X, v0, gV)
				norm_grad = torch.norm(grad_tau)
				
				if self.verbose_level >= 2:
					
					print(f"\t[{i:04d}] ||grad_tau||: {norm_grad:.2e}", end = ', ')
					
				if norm_grad <= self.stopping_eps and i > 0:
					
					self.flag_maturity = True
					
					break

				if self.max_back <= 0:	# no backtracking linesearch
					
					if not self.momentum_tau:
					
						tau -= self.omega.item() * grad_tau
						
					elif self.beta_tau > 0.0:
					
						momentum = (1 - self.beta_tau) * grad_tau + self.beta_tau * momentum
					
						tau	-= self.omega.item() * momentum
						
					else:
						
						raise ValueError(
							f"Invalid parameters: momentum_tau={self.momentum_tau}, "
							f"beta_tau={self.beta_tau}"
						)
					
					self.gd_tot_cnt += 1
					self.gd_cum_cnt += 1
					
					loss_new = self._evaluate_psi_(tau, gV, v0, X)
					
					delta = loss_new - loss
					
					if abs(delta) <= self.stopping_eps:
						
						self.flag_maturity = True
							
					loss = loss_new
						
					if self.verbose_level >= 2:
				
						print(f"loss: {loss:.2e}", end = ', ')
						print(f"delta: {delta:.2e}")
						
				else:	# backtracking linesearch
				
					for j in range(self.max_back):
						
						tau	-= self.omega.item() * grad_tau
						
						self.gd_tot_cnt += 1
						self.gd_cum_cnt += 1
						
						loss_new = self._evaluate_psi_(tau, gV, v0, X)

						if loss_new <= loss:
							
							delta = loss_new - loss
							
							if abs(delta) <= self.stopping_eps:
								
								self.flag_maturity = True
								
							loss = loss_new
							
							if self.verbose_level >= 2:
						
								print(f"loss: {loss:.2e}", end = ', ')
								print(f"delta: {delta:.2e}")
							
							break
							
						else:
							
							self.omega.decay()
							tau = tau_prev
							self.bk_cum_cnt += 1
							
							if self.omega.item() <= self.stopping_eps:
								
								self.omega.reset()
								self.flag_maturity = False
								
								break
						
			if self.verbose_level >= 1:
				
				print(f"")
				print(f"\ttau:		  {format_tensor_to_print(tau)}")
				print(f"\tbk_cum_cnt: {self.bk_cum_cnt}")
				print(f"")

			return tau

	#--------------------------------------------------------------------------------------
	# __call__: optimize tau via validation gradient and return aggregation
	#......................................................................................

	def __call__(self, inputs):
		
		# inputs: a Python list of `n` linearized `d`-dimensional Torch tensors

		X = torch.stack(inputs).to(self.device)	
		
		if ((self.validation_loader != None) and 
			(self.mini_batch_cnt == 0)):	 		# fix at the beginning
			
			data, target = self.get_next_batch()
			
			self.set_validation_data(data, target)
			
		if self.v is None:
			
			self.v = torch.zeros_like(
				inputs[0],
				device = self.device,
				dtype  = self.dtype
			)
		
		if self.vis_landscpe_2D and self.mini_batch_cnt >= 30:	# skip the first epoch
			
			v0 = torch.clone(self.v).detach()
			gV = self._get_gV_()
			
			# Generate the 2D evaluation grid for tau
			
			tau_x = torch.arange(-99, 501, 5).float()
			tauX, tauY = torch.meshgrid(tau_x, tau_x, indexing="ij")
			
			# Fixed tau for the first client
			
			fixed_tau = 10.0
			n = 3  # Assuming n = 3 clients
			
			# Initialize the energy landscape
			
			energy2D = torch.empty(tauX.size(), device=self.device).fill_(float('nan'))
			
			# Allocate tau tensor
			
			tau = torch.empty(n, device=self.device, dtype=self.dtype)
			tau[0] = fixed_tau  # First client always uses the fixed clipping radius

			# Iterate over the 2D grid and evaluate the psi loss
			
			prt_freq = 100
			cnt_xy = 1
			
			for x in range(tauX.size(0)):
				
				for y in range(tauY.size(1)):
					
					tau[1] = tauX[x, y]  # Second client
					tau[2] = tauY[x, y]  # Byzantine client
					
					# Evaluate the loss and save to energy2D
					
					if cnt_xy % prt_freq == 0:
					
						print(f"{cnt_xy} / {tauX.size(0) * tauY.size(1)}: {cnt_xy / (tauX.size(0) * tauY.size(1)) * 100.0:3.2f}%%")
					
					energy2D[x, y] = self._evaluate_psi_(tau, gV, v0, X)
					
					cnt_xy += 1
			
			# Convert to NumPy array for potential visualization
			
			energy2D_np = energy2D.cpu().numpy()
			
			plt.figure(figsize=(10, 8))
			plt.contourf(tauX.cpu().numpy(), tauY.cpu().numpy(), energy2D_np, levels=100, cmap="viridis")
			plt.colorbar(label="Energy (Psi Loss)")
			plt.xlabel("Tau (Second Client)")
			plt.ylabel("Tau (Byzantine Client)")
			plt.title("Energy Landscape (2D)")
			plt.show()
			
			save_pne = r"C:\Temp\energy2D.mat"
			os.makedirs(os.path.dirname(save_pne), exist_ok=True)
			scipy.io.savemat(save_pne, {'energy2D': energy2D_np})

			print(f"Saved energy2D to {save_pne}\n")

		if not self.gVonly:		# our approach
			
			v0 = torch.clone(self.v).detach()
			gV = self._get_gV_()
			
			with torch.no_grad():
				
				tau = self._optim_tau_(X, v0, gV)
				
				self.tau = tau.clone().detach()

				self.v = self._centered_clipping_(self.tau, v0, X)
				
				self.info = (f"[angle: {self.compute_angle_in_degrees(gV, self.v):.2f}]")
			
				del X, v0, gV, tau
			
				self.mini_batch_cnt += 1  	# @_BaseAggregator

				return torch.clone(self.v).detach()
		
		else:	# only the validation gradient gV is used for training
			
			self.mini_batch_cnt += 1  		# @_BaseAggregator
			
			return torch.clone(self._get_gV_()).detach()
			
		# else:	# legacy sanity check of self._centered_clipping_soft_
		# 	
		# 	self.info = f"[angle: {self.compute_angle_in_degrees(self._get_gV_(), torch.mean(X, dim = 0)):.2f}]"
		# 	
		# 	# self.v = self._centered_clipping_(self.tau, self.v, X)
		# 	self.v = self._centered_clipping_soft_(self.tau, self.v, X)
		# 
		# 	return torch.clone(self.v).detach()
		
	def __str__(self):
		
		return (
			f"\n"
			f"ByRoClipping:\n\n"
			f"\t- stopping_eps:\t{self.stopping_eps:.2e},\n"
			f"\t- omega:\t{self.omega.item():.2e},\n"
			f"\t- max_iter:\t{self.max_iter},\n"
			f"\t- max_back:\t\t{self.max_back},\n"
			f"\t- verbose_level:\t{self.verbose_level}"
			f"\n"
		)

class Clipping(_BaseAggregator):
	
	def __init__(self, tau, n_iter=1):
		
		self.tau = tau
		self.n_iter = n_iter
		super(Clipping, self).__init__()
		self.momentum = None

	def clip(self, v):
		
		v_norm = torch.norm(v)
		scale = min(1, self.tau / v_norm)
		return v * scale

	def __call__(self, inputs):
		
		if self.momentum is None:
			
			self.momentum = torch.zeros_like(inputs[0])

		for _ in range(self.n_iter):
			
			self.momentum = (
				sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
				+ self.momentum
			)
		
		self.mini_batch_cnt += 1	# @_BaseAggregator
		
		return torch.clone(self.momentum).detach()

	def __str__(self):
		
		return "Clipping (tau={}, n_iter={})".format(self.tau, self.n_iter)
