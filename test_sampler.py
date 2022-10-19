import numpy as np
from sampler.cohen_lewis import find_similar_pairs, find_similar_pairs_without_false_positive
from visualize.CircleVisualizer import CircleVisualizer

d = 20
n = 30
K = 8


if __name__ == "__main__":
    A = np.random.randint(2, size=(d, n))

    # Sparse out the matrix:
    #A[A > 1] = 0
    #A[A != 0] = 1

    R = find_similar_pairs_without_false_positive(A, K)
    print(A)
    result = []
    for i in R:
        result.append(R[i]['dot_product'])

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
    gram = np.asarray(X).flatten()

    vis = CircleVisualizer(A, K, R)
    vis.visualize_brute_force()
    vis.visualize_sampled()
