from fastapi import FastAPI
from pydantic import BaseModel
from model_logic import chatbot_json

app = FastAPI(title="Medical Chatbot API")


class UserInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Medical Chatbot API is running"}


@app.post("/diagnose")
def diagnose(data: UserInput):
    return chatbot_json(data.text)