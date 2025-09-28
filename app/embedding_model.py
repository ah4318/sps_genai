import spacy

class EmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def get_embedding(self, word: str):
        token = self.nlp(word)
        return token.vector.tolist()

    def get_similarity(self, word1: str, word2: str):
        return self.nlp(word1).similarity(self.nlp(word2))
