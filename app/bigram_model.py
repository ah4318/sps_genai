
import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus):
        self.bigram_probs = self.build_bigram_model(corpus)

    def build_bigram_model(self, corpus):
        model = defaultdict(lambda: defaultdict(int))
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                model[words[i]][words[i + 1]] += 1

        # Convert counts to probabilities
        for word in model:
            total_count = float(sum(model[word].values()))
            for next_word in model[word]:
                model[word][next_word] /= total_count
        return model

    def generate_text(self, start_word, length=10):
        word = start_word.lower()
        if word not in self.bigram_probs:
            return f"Start word '{start_word}' not found in the corpus."
        result = [word]
        for _ in range(length - 1):
            next_words = list(self.bigram_probs[word].keys())
            probs = list(self.bigram_probs[word].values())
            if not next_words:
                break
            word = random.choices(next_words, probs)[0]
            result.append(word)
        return " ".join(result)

