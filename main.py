from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import openai
import os

# Set your OpenAI API key
openai.api_key = "sk-xxx"  # Replace with your actual key or use environment variable

app = FastAPI()

# Enable CORS for all origins (you can restrict it in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face plant disease model
processor = AutoImageProcessor.from_pretrained("Diginsa/Plant-Disease-Detection-Project")
model = AutoModelForImageClassification.from_pretrained("Diginsa/Plant-Disease-Detection-Project")

# Root route to prevent 404 error on base URL
@app.get("/")
def read_root():
    return {"message": "🌿 Plant Disease Detection API is running!"}

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the uploaded image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    outputs = model(**inputs)

    # Get predicted label and confidence
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

    # GPT suggestion prompt
    prompt = f"""I have detected a plant disease called: {predicted_label}.
Can you give short prevention and treatment tips in two small paragraphs?"""

    try:
        # Correct GPT API usage with openai
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in plant disease management."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        gpt_suggestions = response.choices[0].message["content"]
    except Exception as e:
        gpt_suggestions = f"❌ OpenAI error:\n{str(e)}"

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence * 100, 2),
        "gpt_suggestions": gpt_suggestions
    }
