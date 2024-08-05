import nltk
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.probability import FreqDist
import string

# Download necessary NLTK data
nltk.download('punkt')

# Sample text corpus
text = """Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful."""

# Tokenize the text into words
words = word_tokenize(text)

# Remove punctuation and convert to lower case
words = [word.lower() for word in words if word.isalnum()]

# Compute bigrams and their frequencies
bigram_freq = FreqDist(bigrams(words))
unigram_freq = FreqDist(words)

def predict_next_word(prev_word):
    """Predict the next word given the previous word using bigram probabilities."""
    next_words = [(bigram[1], freq) for bigram, freq in bigram_freq.items() if bigram[0] == prev_word]
    if not next_words:
        return "No prediction available."
    
    # Sort next words by frequency in descending order
    next_words_sorted = sorted(next_words, key=lambda x: x[1], reverse=True)
    
    # Return the most probable next word
    return next_words_sorted[0][0]

# Example usage
prev_word = "language"
next_word = predict_next_word(prev_word)

print(f"The most likely next word after '{prev_word}' is '{next_word}'.")

