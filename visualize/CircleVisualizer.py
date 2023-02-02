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

    def __init__(self, A, K, R, title):
        self.A = A
        self.d = A.shape[0]
        self.n = A.shape[1]
        self.R = R
        self.K = K
        self.gram_matrix = None
        self.title = title


    def visualize(self):
        """
        Create circle visualization using results of sampling
        :return:
        """
        node_names = [f'{i}' for i in range(self.n)]

        con = np.zeros((self.n, self.n))
        con[:] = np.NaN
        for pair in self.R:
            if 'dot_product' not in self.R[pair]:
                self.R[pair]['dot_product'] = np.dot(self.A[..., pair[0]], self.A[..., pair[1]])
            con[pair[1], pair[0]] = self.R[pair]['dot_product']
            
            #if self.R[pair]['dot_product'] >= self.K:
            #    con[pair[1], pair[0]] = self.R[pair]['dot_product']
            #else:
            #    con[pair[1], pair[0]] = np.NaN

        fig, ax = plt.subplots(figsize=(20, 20), facecolor='black',
                               subplot_kw=dict(polar=True))
        plot_connectivity_circle(con, node_names, ax=ax, title=self.title, fontsize_title=36)
        fig.savefig("sampled_visualize", dpi=300)
