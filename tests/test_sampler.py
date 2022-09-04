import numpy as np
from sampler.cohen_lewis import find_similar_pairs


d = 100
n = 1000
K = d ** 0.25


if __name__ == "__main__":
    A = np.random.randint(10, size=(d, n))

    # Sparse out the matrix:
    A[A > 1] = 0
    A[A != 0] = 1

    R = find_similar_pairs(A, K)

    print(R)

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
    # print(greater_than)
