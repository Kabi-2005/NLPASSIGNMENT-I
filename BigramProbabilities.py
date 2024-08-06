import nltk
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.probability import FreqDist
nltk.download('punkt')
text = """Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful."""

words = word_tokenize(text)
words = [word.lower() for word in words if word.isalnum()]
unigram_freq = FreqDist(words)
bigram_freq = FreqDist(bigrams(words))
bigram_probabilities = {}
for bigram in bigram_freq:
    first_word = bigram[0]
    bigram_prob = bigram_freq[bigram] / unigram_freq[first_word]
    bigram_probabilities[bigram] = bigram_prob
print("Bigram Probabilities:")
for bigram, prob in bigram_probabilities.items():
    print(f"{bigram}: {prob:.4f}")

