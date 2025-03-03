from typing import Callable
import torch


class TorchServer(object):
	
	def __init__(self,
		optimizer: torch.optim.Optimizer,
		device					= None,
		model: torch.nn.Module	= None,
		loss_func: Callable		= None
		):
		
		r"""
			Initialize TorchServer with a reference to the model and its optimizer.
			
			Args:
				optimizer (torch.optim.Optimizer):	The optimizer for the model.
				device (torch.device, optional):	Device for computations. Default is None.
				model (torch.nn.Module):			The global model.
		"""
		
		self.optimizer	= optimizer
		self.device		= device
		self.model		= model
		self.loss_func	= loss_func
		
		self.aggregator = None		# bucketing_wrapper hides the name

	def apply_gradient(self) -> None:
		
		r""" Apply the aggregated gradient to the model. """
		
		self.optimizer.step()

	def set_gradient(self, gradient: torch.Tensor) -> None:
		
		r"""
			Set the aggregated gradient to the model parameters.

			Args:
				gradient (torch.Tensor): Flattened gradient vector to distribute across the model.
		"""
		
		beg = 0
		
		for group in self.optimizer.param_groups:
			
			for p in group["params"]:
				
				if p.grad is None:
					continue
					
				# for p in self.model.parameters():
				
				end			= beg + len(p.grad.view(-1))
				x			= gradient[beg:end].reshape_as(p.grad.data)
				p.grad.data	= x.clone().detach()
				beg			= end
