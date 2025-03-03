import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

from torchvision import datasets
from collections import defaultdict
import numpy as np

#==========================================================================================

class DistributedSampler(Sampler):
	
	r"""
		Sampler that restricts data loading to a subset of the dataset.
		It is especially useful in conjunction with
		:class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
		process can pass a DistributedSampler instance as a DataLoader sampler,
		and load a subset of the original dataset that is exclusive to it.

		.. note::
		
			Dataset is assumed to be of constant size.

		Arguments:
		
			dataset: Dataset used for sampling.
			num_replicas (optional): Number of processes participating in
				distributed training.
			rank (optional): Rank of the current process within num_replicas.
			shuffle (optional): If true (default), sampler will shuffle the indices
	"""

	def __init__(self, dataset, num_replicas = None, rank = None, shuffle = True):
		
		if num_replicas is None:
			
			if not dist.is_available():
				
				raise RuntimeError("Requires distributed package to be available")
				
			num_replicas = dist.get_world_size()
			
		if rank is None:
			
			if not dist.is_available():
				
				raise RuntimeError("Requires distributed package to be available")
				
			rank = dist.get_rank()
			
		self.dataset	  = dataset
		self.num_replicas = num_replicas
		self.rank		  = rank
		self.epoch		  = 0
		self.num_samples  = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
		self.total_size   = self.num_samples * self.num_replicas
		self.shuffle	  = shuffle

	def __iter__(self):
		
		# deterministically shuffle based on epoch
		
		g = torch.Generator()
		g.manual_seed(self.epoch)
		
		if self.shuffle:
			
			indices = torch.randperm(len(self.dataset), generator=g).tolist()
			
		else:
			
			indices = list(range(len(self.dataset)))

		# add extra samples to make it evenly divisible
		
		indices += indices[: (self.total_size - len(indices))]
		assert len(indices) == self.total_size

		# subsample
		
		indices = indices[self.rank : self.total_size : self.num_replicas]
		assert len(indices) == self.num_samples

		return iter(indices)

	def __len__(self):
		
		return self.num_samples

	def set_epoch(self, epoch):
		
		self.epoch = epoch

	def __str__(self):
		
		return "DistributedSampler(num_replicas={num_replicas},rank={rank},shuffle={shuffle})".format(
			num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
		)

#------------------------------------------------------------------------------------------

class NONIIDLTSampler(DistributedSampler):
	
	r"""
		NONIID + Long-Tail sampler.

		alpha: alpha controls the noniidness.
			- alpha = 0 refers to completely noniid
			- alpha = 1 refers to iid.

		beta: beta controls the long-tailness.
			- Class i takes beta ** i percent of data.
	"""

	def __init__(self, alpha, beta, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		
		self.alpha = alpha
		self.beta  = beta
		
		assert beta  >= 0 and beta <= 1
		assert alpha >= 0

	def __iter__(self):
		
		# The dataset are not shuffled across nodes.
		
		g = torch.Generator()
		g.manual_seed(0)

		if self.shuffle:
			
			indices = torch.randperm(len(self.dataset), generator=g).tolist()
			
		else:
			
			indices = list(range(len(self.dataset)))

		nlabels = len(self.dataset.classes)
		indices = []
		
		for i in range(nlabels):
			
			label_indices = torch.nonzero(self.dataset.targets == i)
			label_indices = label_indices.flatten().tolist()
			label_selected = int(len(label_indices) * self.beta ** i)
			# discard the rest of label_indices[label_selected:]
			indices += label_indices[:label_selected]

		# Adjust after removing data points.
		
		self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
		self.total_size  = self.num_samples * self.num_replicas

		# add extra samples to make it evenly divisible
		
		indices += indices[: (self.total_size - len(indices))]
		assert len(indices) == self.total_size

		if self.alpha:	# IID
			
			indices = indices[self.rank : self.total_size : self.num_replicas]
			
		else:			# NON-IID
			
			indices = indices[
				self.rank * self.num_samples : (self.rank + 1) * self.num_samples
			]
			
		assert len(indices) == self.num_samples

		if self.shuffle:
			
			g = torch.Generator()
			g.manual_seed(self.rank ** 3 + self.epoch)
			idx_idx = torch.randperm(len(indices), generator=g).tolist()
			indices = [indices[i] for i in idx_idx]

		return iter(indices)

	def __str__(self):
		return "NONIIDLTSampler"

#==========================================================================================

class PatchedMNIST(datasets.MNIST):
	
	r"""
		A subclass of torchvision's MNIST dataset that supports batched index retrieval.
		This allows efficient data fetching for a list of indices, improving compatibility
		with custom samplers like the `StratifiedBatchSampler`.
	"""
	
	def __getitem__(self, index):
		
		
		r"""
			Fetch a single or batched sample.

			Args:
			
				index (int or list): An integer or a list of indices.

			Returns:
			
				A tuple of images and targets if index is a list.
				Otherwise, returns a single image and its target.
		"""
		
		if isinstance(index, list):
			
			# Batch mode: process a list of indices
			
			images, targets = zip(*(super(PatchedMNIST, self).__getitem__(idx) for idx in index))
			images  = torch.stack(images, dim=0)  # Shape: [batch_size, 1, 28, 28]
			targets = torch.tensor(targets, dtype=torch.long)  # Shape: [batch_size]
			
			return images, targets

		# Single index mode: process a single sample
		
		img, target = super(PatchedMNIST, self).__getitem__(index)
		target = target.item() if isinstance(target, torch.Tensor) and target.numel() == 1 else target
		
		return img, target

#------------------------------------------------------------------------------------------

class StratifiedBatchSampler(Sampler):
    
    r"""
        A custom sampler that generates batches with stratified class distributions.

        Args:
        
            dataset: The dataset containing samples and labels.
            batch_size: Number of samples per batch.
            seed: An integer seed for reproducible shuffling.

        Attributes:
        
            class_indices (defaultdict): A mapping from class labels to sample indices.
            seed (int): Seed number for shuffling operations.
            rng (np.random.Generator): Numpy random generator initialized with the seed.
    """
    
    def __init__(self, dataset, batch_size, seed):
        
        self.dataset    = dataset
        self.batch_size = batch_size
        self.seed       = seed
        self.rng        = np.random.default_rng(self.seed)
        
        # self.class_indices: a dictionary where each key is a class label, and 
        # the value is a list of indices of samples belonging to that class
        
        self.class_indices = defaultdict(list)

        # extract class labels and organize indices
        
        targets = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else np.array(dataset.targets)
        targets = targets.astype(int)
        
        for idx, label in enumerate(targets):
            self.class_indices[label].append(idx)

        # shuffle indices within each class
        
        for key in self.class_indices:
            self.rng.shuffle(self.class_indices[key])

        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        self.samples_per_class = self.batch_size // self.num_classes
        self.extra_samples = self.batch_size % self.num_classes

    def __iter__(self):
        
        r"""
            Generate stratified batches.

            Yields:
                A list of indices for each batch.
        """
        
        # Create a copy of class indices and shuffle them
        class_iters = {}
        for cls in self.classes:
            indices = self.class_indices[cls].copy()
            self.rng.shuffle(indices)
            class_iters[cls] = iter(indices)
        
        for _ in range(self.__len__()):
            batch = []
            # Add samples_per_class from each class
            for cls in self.classes:
                for _ in range(self.samples_per_class):
                    try:
                        batch.append(next(class_iters[cls]))
                    except StopIteration:
                        # Refill iterator if exhausted
                        indices = self.class_indices[cls].copy()
                        self.rng.shuffle(indices)
                        class_iters[cls] = iter(indices)
                        batch.append(next(class_iters[cls]))
            # Add extra samples from randomly selected classes
            extra_classes = self.rng.choice(self.classes, self.extra_samples, replace=True)
            for cls in extra_classes:
                try:
                    batch.append(next(class_iters[cls]))
                except StopIteration:
                    # Refill iterator if exhausted
                    indices = self.class_indices[cls].copy()
                    self.rng.shuffle(indices)
                    class_iters[cls] = iter(indices)
                    batch.append(next(class_iters[cls]))
            yield batch

    def __len__(self):
        """
        Returns the number of batches per epoch.

        Returns:
            int: Total number of batches.
        """
        return math.ceil(len(self.dataset) / float(self.batch_size))



