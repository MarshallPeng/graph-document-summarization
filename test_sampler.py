import numpy as np
from sampler.cohen_lewis import find_similar_pairs
from visualize.CircleVisualizer import CircleVisualizer

d = 20
n = 30
K = 10


if __name__ == "__main__":
    A = np.random.randint(2, size=(d, n))

    # Sparse out the matrix:
    #A[A > 1] = 0
    #A[A != 0] = 1

    R = find_similar_pairs(A, K)
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
    print("d = ", d)
    print("n = ", n)
    X = A.T @ A
    print("gram = ", X)
    np.fill_diagonal(X, -1)
    gram = np.asarray(X).flatten()
    print(sorted(gram)[::-1])
    print(sorted(result)[::-1])

    vis = CircleVisualizer(A, K, R)
    vis.visualize_brute_force()
    vis.visualize_sampled()
