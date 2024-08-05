import nltk
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.probability import FreqDist

# Download necessary NLTK data
nltk.download('punkt')

# Sample text corpus
text = """Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful."""

# Tokenize the text into words
words = word_tokenize(text)

# Remove punctuation and convert to lower case
words = [word.lower() for word in words if word.isalnum()]

# Compute unigrams and bigrams
unigram_freq = FreqDist(words)
bigram_freq = FreqDist(bigrams(words))

# Compute bigram probabilities
bigram_probabilities = {}
for bigram in bigram_freq:
    first_word = bigram[0]
    bigram_prob = bigram_freq[bigram] / unigram_freq[first_word]
    bigram_probabilities[bigram] = bigram_prob

# Display the bigrams and their probabilities
print("Bigram Probabilities:")
for bigram, prob in bigram_probabilities.items():
    print(f"{bigram}: {prob:.4f}")

