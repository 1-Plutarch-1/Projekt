from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
chatbot = pipeline("question-answering", model="bert-base-german-cased")

class Question(BaseModel):
    question: str
    context: str

@app.post("/ask")
def ask(question: Question):
    answer = chatbot(question=question.question, context=question.context)
    return {"answer": answer}
