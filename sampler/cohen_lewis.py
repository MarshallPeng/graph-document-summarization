import numpy as np
import math
import random
from tqdm import tqdm


class CohenLewis:

    def __init__(self, A):
        self.A = A
        self.d = A.shape[0]
        self.n = A.shape[1]
        self.delta = 0.2
        self.scores = self._preprocess()  # dim 1 x d
        self.weights = self.scores ** 2
        self.arangen = np.arange(self.n)
        self.aranged = np.arange(self.d)
        #self.normalized_weights = self.weights/self.weights.sum()
        #self.normalized_A = self.A/self.scores[:,np.newaxis]
        self.gamma = np.sum(self.weights)
        print(self.gamma - np.sum(A.T @ A))
        self.eps = 0.5
        self.coeff = 3.5
        self.acc = 25

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
        r = random.choices(self.aranged, self.weights)

        # Sample i with probability \frac{A_{ri}}{score(r)}
        # what if weights are negative? 
        
        weight_r = np.squeeze(self.A[r])
        i = random.choices(self.arangen, weight_r)[0]

        # eliminate probability of j == i
        
        weight_r[i] = 0

        # Sample j with probability \frac{A_{rj}}{score(r)}
        j = random.choices(self.arangen, weight_r)[0]

        return (i, j) if j >= i else (j, i)


def find_similar_pairs(A, K):
    """
    Return pairs of columns of A with dot product >= K
    :param A:
    :param K:
    :return:
    """
    sampler = CohenLewis(A)
    N = (math.ceil((sampler.gamma / K) * math.log(sampler.gamma / (K * sampler.delta)))) // sampler.acc
    R = {}

    print(f'Sampling {N} pairs using naive Cohen-Lewis')
    for i in tqdm(range(N), position=0, leave=True):
        sample = sampler.cohen_lewis()
        product = 0
        for i in range(len(A)):
            product += A[i][sample[0]] * A[i][sample[1]]
        #product = np.dot(A[..., sample[0]], A[..., sample[1]])
        if sample not in R and product >= K:
            R[sample] = {}
            # R[sample]['count'] = 0
            R[sample]['dot_product'] = product
        # R[sample]['count'] += 1

    return R


def find_similar_pairs_without_false_positive(A, K):
    """
    Return pairs of columns of A with dot product >= K
    :param A:
    :param K:
    :return:
    """
    sampler = CohenLewis(A)
    print(f'There can be at most {sampler.gamma / K} pairs with dot product at least {K}')
    N = math.ceil((sampler.gamma / K) * math.log(sampler.gamma / (K * sampler.delta)) * (1 / sampler.eps ** 2)) // sampler.acc
    threshold = int(sampler.coeff * math.ceil((1 - sampler.eps / 2) * math.log(sampler.gamma / (K * sampler.delta)) * (1 / sampler.eps ** 2))) // sampler.acc
    R = {}
    S = {}

    print(f'Sampling {N} pairs using Cohen-Lewis')
    for i in tqdm(range(N), position=0, leave=True):
        sample = sampler.cohen_lewis()

        if sample not in R:
            R[sample] = {}
            R[sample]['count'] = 0
        R[sample]['count'] += 1
        if R[sample]['count'] >= threshold:
            S[sample] = {}

    return S



def matrix_multiply_for_loop(A, B):
    result = []  # final result
    for i in range(len(A)):
        row = []  # the new row in new matrix
        for j in range(len(B[0])):
            product = 0  # the new element in the new row
            for v in range(len(A[i])):
                product += A[i][v] * B[v][j]
            row.append(product)  # append sum of product into the new row
        result.append(row)  # append the new row into the final result
    return result


def find_similar_pairs_brute_force(A, K):
    """
    Return pairs of columns of A with dot product >= K
    :param A:
    :param K:
    :return:
    """
    
    
    gram = matrix_multiply_for_loop(A.T, A)
    R = {}

    
    for i in range(len(gram)):
        for j in range(i, len(gram[0])):
            if gram[i][j] >= K:
                R[(i, j)] = {}
                R[(i, j)]['dot_product'] = gram[i][j]

    return R
