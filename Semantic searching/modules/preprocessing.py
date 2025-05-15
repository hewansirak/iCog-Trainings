import re
import spacy
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r"\s+", " ", text) 
    text = text.replace("\n", " ").strip()
    return text

def expand_synonyms(text):
    synonyms = {
        "author": ["writer", "researcher"],
        "paper": ["document", "article"],
        "study": ["research", "analysis"]
    }
    for word, syns in synonyms.items():
        for syn in syns:
            text = text.replace(word, f"{word} {syn}")
    return text

def extract_entities(text):
    doc = nlp(text)
    return [ent.text.strip().lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

def highlight_entities(text, entities):
    for ent in sorted(set(entities), key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(ent)})\b", re.IGNORECASE)
        text = pattern.sub(r"**🟦 \1**", text)
    return text
