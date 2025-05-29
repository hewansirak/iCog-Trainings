from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad") # Finetunes Q&A dataset

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result["answer"], result["score"]
