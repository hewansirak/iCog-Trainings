import re
from nltk.corpus import wordnet
import nltk
nltk.download("wordnet")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def expand_synonyms(text):
    words = text.split()
    synonyms = set()
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
    return text + " " + " ".join(list(synonyms)[:10])  # Limit extra noise
