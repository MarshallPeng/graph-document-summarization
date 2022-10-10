from embeddings.sbert import SBert
from loader.load import SQuALITYLoader
from sampler.cohen_lewis import find_similar_pairs, CohenLewis
import numpy as np
from random import randint

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

print(id_to_embeddings)

for doc_id in id_to_embeddings:
    A = id_to_embeddings[doc_id].T
    d = A.shape[0]
    n = A.shape[1]

    # zero out negative values
    A[A < 0] = 0

    K = 0.5
    R = find_similar_pairs(A, K)

    greater_than = set()
    less_than = set()

    for pair in R:
        if R[pair]['dot_product'] >= K:
            greater_than.add(pair)
        else:
            less_than.add(pair)

    print(f'num greater than: {len(greater_than)}')
    print(f'num less than: {len(less_than)}')
