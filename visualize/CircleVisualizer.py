import numpy as np
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
import sys
import time
np.set_printoptions(threshold=sys.maxsize)


def matrix_multiply_for_loop(A, B):
    start = time.process_time()
    result = []  # final result
    for i in range(len(A)):
        row = []  # the new row in new matrix
        for j in range(len(B[0])):
            product = 0  # the new element in the new row
            for v in range(len(A[i])):
                product += A[i][v] * B[v][j]
            row.append(product)  # append sum of product into the new row
        result.append(row)  # append the new row into the final result
    print(f'For Loop Matrix Mul time: {time.process_time() - start}')
    return result


class CircleVisualizer:
    """
    build a visualization tool for the sampled edges as a graph.
    So allocate i (sentences) on a unit circle as nodes,
    use brute force to compute the dot products between i,j and color the edge i,j depending on their dot product.
    Then do the same with the result given by Cohen-Lewis except coloring them using a binary value (whether > K).
    """

    def __init__(self, A, K, R):
        self.A = A
        self.d = A.shape[0]
        self.n = A.shape[1]
        self.R = R
        self.K = K
        self.gram_matrix = None

    def visualize_brute_force(self, use_numpy=False):
        """
        Create Circle Visualization by brute force calculating dot product between each pair
        :return:
        """
        node_names = [f'{i}' for i in range(self.n)]
        if use_numpy:
            self.gram_matrix = np.array(self.A.T @ self.A, dtype=np.float32)
        else:
            self.gram_matrix = np.array(matrix_multiply_for_loop(self.A.T, self.A), dtype=np.float32)

        self.gram_matrix[self.gram_matrix < (self.K * 1.0)] = np.NaN
        np.fill_diagonal(self.gram_matrix, np.NaN)  # get rid of diagonals

        fig, ax = plt.subplots(figsize=(20, 20), facecolor='black',
                               subplot_kw=dict(polar=True))

        plot_connectivity_circle(self.gram_matrix, node_names, ax=ax, title="Brute Force", fontsize_title=36)

        fig.savefig("brute_force_visualize", dpi=300)

    def visualize_sampled(self):
        """
        Create circle visualization using results of sampling
        :return:
        """
        node_names = [f'{i}' for i in range(self.n)]

        con = np.zeros((self.n, self.n))
        con[:] = np.NaN
        for pair in self.R:
            con[pair[1], pair[0]] = self.R[pair]['dot_product'] if self.R[pair]['dot_product'] >= self.K else np.NaN

        fig, ax = plt.subplots(figsize=(20, 20), facecolor='black',
                               subplot_kw=dict(polar=True))
        plot_connectivity_circle(con, node_names, ax=ax, title="Cohen Lewis", fontsize_title=36)
        fig.savefig("sampled_visualize", dpi=300)
