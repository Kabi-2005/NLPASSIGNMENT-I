import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string
nltk.download('punkt')
text = """Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful."""

words = word_tokenize(text)
words = [word.lower() for word in words if word.isalnum()]
unigrams = FreqDist(words)
print("Unigrams and their frequencies:")
for unigram, freq in unigrams.items():
    print(f"{unigram}: {freq}")

