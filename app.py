import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F  # For softmax
# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")
model.eval()

def predict_emotion(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()

    # Map label ids to emotions
    id2label = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'suprise', 4: 'fear', 5: 'happy'}
    emotion_probs = {id2label[i]: prob for i, prob in enumerate(probabilities)}
    top_emotion = max(emotion_probs, key=emotion_probs.get)
    return f"Emotion probabilities: {emotion_probs}\n\nTop emotion: {top_emotion}"

# Gradio interface
interface = gr.Interface(fn=predict_emotion, 
                         inputs=gr.Textbox(lines=2, placeholder="Enter a sentence..."), 
                         outputs="text", 
                         title="Emotion Detection", 
                         description="Enter a sentence and get an emotion prediction.")

# Launch the interface
interface.launch()