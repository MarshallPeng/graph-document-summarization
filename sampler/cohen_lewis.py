import numpy as np
import math
import random
from tqdm import tqdm


class CohenLewis:

    def __init__(self, A):
        self.A = A
        self.d = A.shape[0]
        self.n = A.shape[1]
        self.delta = 0.1
        self.scores = self._preprocess()  # dim 1 x d
        self.gamma = np.sum(self.scores ** 2)

    def _preprocess(self):
        """
        Upon input of the d x n matrix A, compute L1 norm of each row
        :return:
        """
        return np.linalg.norm(self.A, axis=1, ord=1)

    def _score(self, r):
        """
        Returns the score(r) i.e. L1 norm of the rth row
        :param r:
        :return:
        """
        return self.scores[r]

    def cohen_lewis(self):
        """
        Return (i,j) with probability proportional to (A^T A)_{ij}
        :return:
        """
        # Sample a feature with probability proportional to square of score
        r = random.choices(np.arange(self.d), weights=self.scores ** 2)

        # Sample i with probability \frac{A_{ri}}{score(r)}
        # what if weights are negative? 
        i = random.choices(np.arange(self.n), weights=np.squeeze(self.A[r]))[0] 

        # eliminate probability of j == i
        j_prob = np.squeeze(self.A[r])
        j_prob[i] = 0

        # Sample j with probability \frac{A_{rj}}{score(r)}
        j = random.choices(np.arange(self.n), weights=j_prob)[0]

        return (i, j) if j >= i else (j, i)


def find_similar_pairs(A, K):
    """
    Return pairs of columns of A with dot product >= K
    :param A:
    :param K:
    :return:
    """
    sampler = CohenLewis(A)
    N = (math.ceil((sampler.gamma / K) * math.log(sampler.gamma / (K * sampler.delta)))) 
    R = {}

    print(f'Sampling {N} pairs')
    for i in tqdm(range(N), position=0, leave=True):
        sample = sampler.cohen_lewis()

        if sample not in R:
            R[sample] = {}
            R[sample]['count'] = 0
            R[sample]['dot_product'] = np.dot(A[..., sample[0]], A[..., sample[1]])
        R[sample]['count'] += 1

    return R
