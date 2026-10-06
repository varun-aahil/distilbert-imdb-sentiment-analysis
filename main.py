import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./distilbert_imdb_final"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

labels = ["Negative", "Positive"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return FileResponse('index.html')


class TextPayload(BaseModel):
    text: str


@app.post("/predict")
async def predict(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    return {
        "text": payload.text,
        "sentiment": labels[pred],
        "confidence": round(probs[pred].item(), 3),
    }