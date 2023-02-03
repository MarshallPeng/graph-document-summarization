"""

1. Generate an encoding for each sentence in an article (using BERT or SBERT, no need to fine-tune at this time)
2. Calculating distances between each pair of sentence and cluster them
3. Manually check samples from each cluster to see if they share similar semantic meanings

"""
import torch
from torchsample.transforms import RangeNormalize

from embeddings.sbert import SBert
from loader.load import SQuALITYLoader
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt

# Load data from SQuality
loader = SQuALITYLoader()

# initialize sentence bert
sbert = SBert()

doc_id = '63833'
sentences = loader.get_sentences(loader.data['dev'][doc_id])
embeddings = sbert.encode(sentences)

norm_01 = RangeNormalize(0, 1)
A = norm_01(torch.tensor(embeddings)).numpy()

n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters).fit(A)

cluster_map = pd.DataFrame()
cluster_map['sentences'] = pd.DataFrame(sentences)
cluster_map['clusters'] = kmeans.labels_
cluster_map.sort_values(by=['clusters'])


# Generate a bunch of random colors
def random_color_code():
    return '#{:02X}{:02X}{:02X}'.format(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


colors = [random_color_code() for _ in range(n_clusters)]

# Write sentences to file according to cluster number
with open("index.html", "w") as file:
    file.write("<html>\n")
    file.write("<head>\n")
    file.write("<title>Cluster Map Sentence</title>\n")
    file.write("</head>\n")
    file.write("<body>\n")

    for row in cluster_map.iterrows():
        line = "<span style='color:" + str(colors[row[1][1]]) + ";'>" + str(row[1][0]) + " </span>"
        line = "<br />".join(line.split("\n"))
        file.write(line)

    file.write("</body>\n")
    file.write("</html>\n")
