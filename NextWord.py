import nltk
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.probability import FreqDist
import string
nltk.download('punkt')
text = """Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful."""
words = word_tokenize(text)

words = [word.lower() for word in words if word.isalnum()]
bigram_freq = FreqDist(bigrams(words))
unigram_freq = FreqDist(words)

def predict_next_word(prev_word):
    """Predict the next word given the previous word using bigram probabilities."""
    next_words = [(bigram[1], freq) for bigram, freq in bigram_freq.items() if bigram[0] == prev_word]
    if not next_words:
        return "No prediction available."
    next_words_sorted = sorted(next_words, key=lambda x: x[1], reverse=True)
    return next_words_sorted[0][0]
prev_word = "language"
next_word = predict_next_word(prev_word)

print(f"The most likely next word after '{prev_word}' is '{next_word}'.")

