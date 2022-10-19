import numpy as np
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

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

    def visualize_brute_force(self):
        """
        Create Circle Visualization by brute force calculating dot product between each pair
        :return:
        """
        node_names = [f'{i}' for i in range(self.n)]
        gram_matrix = np.array(self.A.T @ self.A, dtype=np.float32)
        gram_matrix[gram_matrix <= (self.K * 1.0)] = np.NaN
        np.fill_diagonal(gram_matrix, np.NaN)  # get rid of diagonals

        fig, ax = plt.subplots(figsize=(20, 20), facecolor='black',
                               subplot_kw=dict(polar=True))

        plot_connectivity_circle(gram_matrix, node_names, ax=ax, title="Brute Force")

        fig.savefig("brute_force_visualize", dpi=300)

    def visualize_sampled(self):
        """
        Create circle visualization using results of sampling
        :return:
        """
        node_names = [f'{i}' for i in range(self.n)]

        con = np.zeros((self.n, self.n))
        for pair in self.R:
            con[pair[1], pair[0]] = self.R[pair]['dot_product'] if self.R[pair]['dot_product'] >= self.K else np.NaN

        print(con)

        fig, ax = plt.subplots(figsize=(20, 20), facecolor='black',
                               subplot_kw=dict(polar=True))
        plot_connectivity_circle(con, node_names, ax=ax, title="Cohen Lewis")
        fig.savefig("sampled_visualize", dpi=300)
