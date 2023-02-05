from embeddings.sbert import SBert
from loader.load import SQuALITYLoader
from sampler.cohen_lewis import find_similar_pairs_without_false_positive
from visualize.CircleVisualizer import CircleVisualizer
import pandas as pd
import numpy as np
import torch


from visualize.CosineSimilarityVisualizer import CosineSimilarityVisualizer

# Load data from SQuality
loader = SQuALITYLoader()

# initialize sentence bert
sbert = SBert()

# Get sentences
id_to_sentences = {doc_id: loader.get_sentences(document) for doc_id, document in loader.data['dev'].items()}

id_to_embeddings = {}
for doc_id in id_to_sentences:
    print(f'Processing document {doc_id} ...')
    id_to_embeddings[doc_id] = sbert.encode(id_to_sentences[doc_id])
    break

#print(id_to_embeddings)

for doc_id in id_to_embeddings:
    A = id_to_embeddings[doc_id].T
    d = A.shape[0]
    n = A.shape[1]
    print("number of pairs:", n ** 2)
    # zero out negative values


    cos_sim_vis = CosineSimilarityVisualizer(A, 20)
    cos_sim_vis.visualize()

    #K = 0.5
    #R = find_similar_pairs_without_false_positive(A, K)


    # greater_than = set()
    # less_than = set()

    # print(R)

    # for pair in R:
    #     if R[pair]['dot_product'] >= K:
    #         greater_than.add(pair)
    #     else:
    #         less_than.add(pair)

    # print(f'num greater than: {len(greater_than)}')
    # print(f'num less than: {len(less_than)}')




    # Generate Circle Visualization
    #vis = CircleVisualizer(A, K, R)
    #vis.visualize_brute_force()
    #vis.visualize_sampled_2()
