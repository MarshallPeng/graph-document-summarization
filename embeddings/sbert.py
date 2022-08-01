from sentence_transformers import SentenceTransformer


class SBert:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.sentence_to_embeddings = {}

    def encode(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
        # self.sentence_to_embeddings = {sentence: embedding for sentence, embedding in zip(sentences, embeddings)}

