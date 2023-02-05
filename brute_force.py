"""

1. Generate an encoding for each sentence in an article (using BERT or SBERT, no need to fine-tune at this time)
2. Calculating distances between each pair of sentence and cluster them
3. Manually check samples from each cluster to see if they share similar semantic meanings

"""
import torch
import torchsample
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

from text_summarization import text_summarization


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

"""
4. Summarizing each cluster with a text summarization model (like T5) and check samples to see if the combination of the summarzations well-summarize the original articles
5. Apply another text suumarization model on the summarized text to generate article-level summarization and manually check quality
"""

summarized_clusters = []
for i in range(n_clusters):

    cluster = cluster_map.loc[cluster_map['clusters'] == i]['sentences'].tolist()
    cluster_text = "\n".join(cluster)
    summarized_clusters.append(text_summarization(cluster_text, "T5"))

print("Summary of %d clusters:" % n_clusters)
for index, summary in enumerate(summarized_clusters):
    print("Cluster %d has summary %s" % (index, summary))

article_summary = text_summarization("\n".join(summarized_clusters), "T5")
print("Article level summarization:")
print(article_summary)

"""
6. Evaluate the output of 5 based on existing datasets (1 short, 1 paper-length)
7. Do 5 on even longer text corpora and check the summarization quality
8. Record time cost for (4+5), 6, and 7
"""