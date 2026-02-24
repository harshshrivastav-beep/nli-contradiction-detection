from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Correct label order
labels = ["Contradiction", "Neutral", "Entailment"]

def predict_relationship(premise, hypothesis):
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return labels[predicted_class_id]
