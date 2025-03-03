r"""
	Modifications to the original code from GitHub repository (Karimireddy et al.):
		https://github.com/epfml/byzantine-robust-noniid-optimizer/commits/main/codes/aggregator/krum.py
		Original Commit SHA: 1cbe52148d974e864a45f6f3e24802e7f11ed9f3

	Issues in the Original Code (SHA: 1cbe52148d974e864a45f6f3e24802e7f11ed9f3):

		1. Condition Check for Byzantine Workers (n >= 2f + 3):
		   - The original code does not ensure that the condition `n >= 2f + 3` is
		   dynamically maintained if the input size `n` changes after bucketing.
		   This condition is essential for Byzantine resilience and correct operation
		   of the Krum algorithm.

		2. Behavior of Multi-Krum with m >= 1:
		   - Both the original and modified versions yield the same output when `m = 1`
		   because only a single worker with the smallest score is selected.
		   However, the original version returns the mere sum of the selected inputs for
		   `m > 1`, which deviates from the intended behavior of Multi-Krum.
		   The modified version correctly averages the selected values by dividing by
		   the number of selected indices (`len(top_m_indices)`), aligning with the 
		   algorithm’s design for balanced aggregation of multiple workers.

		3. Avoidance of Double Squaring:
		   - The original code unnecessarily squares the Euclidean distances twice in
		   `_compute_scores`, raising distances to the fourth power instead of the
		   intended squared distances. The modified version removes this extra squaring,
		   ensuring that scores are calculated based on squared distances, as required
		   by the Krum algorithm.

	Summary: 

		These modifications resolve the identified issues in the original code, enhancing
		the accuracy, robustness, and adherence of the Multi-Krum implementation to
		the intended algorithmic design as described in the referenced paper.
"""

from .base import _BaseAggregator

def _compute_scores(distances, i, n, f):
	
	"""Compute scores for node i.

	Arguments:
		distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
		i {int} -- index of worker, starting from 0.
		n {int} -- total number of inputs
		f {int} -- Total number of Byzantine workers.

	Returns:
		float -- Krum distance score of i.
	"""
	
	s = []
	
	for j in range(n):
		if i != j:
			dist = distances[i][j] if i < j else distances[j][i]
			s.append(dist ** 2)
	_s = sorted(s)[: n - f - 2]
	
	return sum(_s)


def multi_krum(distances, n, f, m):
	
	"""Multi-Krum algorithm

	Arguments:
		distances {dict} -- A dict of dict of distances. distances[i][j] = dist.
		n {int} -- Total number of inputs.
		f {int} -- Total number of Byzantine workers.
		m {int} -- Number of workers to select for aggregation.

	Returns:
		list -- List of indices of selected workers.
	"""

	scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
	sorted_scores = sorted(scores, key=lambda x: x[1])
	
	return [x[0] for x in sorted_scores[:m]]

def _compute_euclidean_distance(v1, v2):
	
	return (v1 - v2).norm()


def pairwise_euclidean_distances(vectors):
	
	"""Compute the pairwise Euclidean distances.

	Arguments:
		vectors {list} -- A list of tensors.

	Returns:
		dict -- A nested dict of distances {i: {j: distance}}.
	"""
	n = len(vectors)
	distances = {i: {} for i in range(n)}
	
	for i in range(n):
		
		for j in range(i + 1, n):
			
			dist = _compute_euclidean_distance(vectors[i].flatten(), vectors[j].flatten())
			distances[i][j] = dist
			
	return distances


class Krum(_BaseAggregator):
	
	r"""
		This class implements the Multi-Krum algorithm.

		Reference:
		Blanchard, Peva, Rachid Guerraoui, and Julien Stainer.
		"Machine learning with adversaries: Byzantine tolerant gradient descent."
		Advances in Neural Information Processing Systems. 2017.
	"""

	def __init__(self, n, f, m):
		
		self.initial_n = n
		self.initial_f = f
		self.m = m
		self.top_m_indices = None
		super(Krum, self).__init__()

	def __call__(self, inputs):
		
		n = len(inputs)
		
		r"""
			Krum assumes `n >= 2f + 3`. However, after bucketing,
			`n` may decrease, thus, we have to ensure `f` is in
			the valid range to apply Krum.
		"""
		
		if n < 2 * self.initial_f + 3:
			
			f = int(max((float(n) - 3.0) / 2.0, 0.0))
			print(f'Krum(n={n},f={f})')
			
		else:
			
			f = self.initial_f

		distances = pairwise_euclidean_distances(inputs)
		top_m_indices = multi_krum(distances, n, f, self.m)
		values = sum(inputs[i] for i in top_m_indices) / len(top_m_indices)
		self.top_m_indices = top_m_indices
		
		self.mini_batch_cnt += 1	# @_BaseAggregator
		
		return values

	def __str__(self):
		
		return f"Krum (m={self.m})"
