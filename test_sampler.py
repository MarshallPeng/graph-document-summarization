import numpy as np
from sampler.cohen_lewis import find_similar_pairs, find_similar_pairs_without_false_positive, find_similar_pairs_brute_force
from visualize.CircleVisualizer import CircleVisualizer
import time
import torch
# from torchsample.transforms import RangeNormalize

# from visualize.HistogramVisualizer import HistogramVisualizer

d = 20
n = 100
K = 10


if __name__ == "__main__":
    A = np.random.randint(2, size=(d, n))
    # norm_01 = RangeNormalize(0, 1)
    # A = norm_01(torch.tensor(A)).numpy()

    # Sparse out the matrix:
    #A[A > 1] = 0
    #A[A != 0] = 1
    # start = time.process_time()
    # R = find_similar_pairs_without_false_positive(A, K)
    # print(f'Cohen Lewis (false positive removed) sample time: {time.process_time() - start}')

    start = time.process_time()
    S = find_similar_pairs(A, K)
    print(f'Cohen Lewis sample time: {time.process_time() - start}')

    start = time.process_time()
    T = find_similar_pairs_brute_force(A, K)
    print(f'For Loop Matrix Mul time: {time.process_time() - start}')


    R = find_similar_pairs_without_false_positive(A, K)
    print(A)
    print(R)
    #result = []
    #for i in R:
    #    result.append(R[i]['dot_product'])


    # greater_than = set()
    # less_than = set()
    #
    # for i, j in R:
    #     dot_product = np.dot(A[..., i], A[..., j])
    #     if dot_product >= K:
    #         greater_than.add((i, j))
    #     else:
    #         less_than.add((i, j))
    #
    # print(len(greater_than))
    # print(greater_than)self.gamma = np.sum(self.gram_matrix)


    X = A.T @ A
    np.fill_diagonal(X, -1)
    print(sorted(np.asarray(X).flatten())[::-1])
    #print(sorted(result)[::-1])

    vis = CircleVisualizer(A, K - 1, R)
    vis.visualize_brute_force()
    vis.visualize_sampled_2()
