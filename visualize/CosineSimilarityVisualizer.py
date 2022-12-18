import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CosineSimilarityVisualizer:
    """
    Displays a heatmap of the similarity between each column of a matrix
    """

    def __init__(self, matrix, num_sentences=None):
        if num_sentences is not None:
            matrix = matrix[:, :num_sentences]
        self.matrix = matrix
        self.cosine_similarity_matrix = self.compute_cosine_similarity()

    def compute_cosine_similarity(self):
        n = np.linalg.norm(self.matrix, axis=0).reshape(1, self.matrix.shape[1])
        return self.matrix.T.dot(self.matrix) / n.T.dot(n)

    def visualize(self):
        sns.heatmap(self.cosine_similarity_matrix, cmap='coolwarm')
        plt.title("Cosine Similarity of Sentence Embedding Pairs")
        plt.show()
