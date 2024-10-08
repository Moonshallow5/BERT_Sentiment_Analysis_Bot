import torch

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

#df=pd.read_csv('sentiment-data/training.csv',names=['text','label'])
df2=pd.read_csv('sentiment-data-2/Emotion_final.csv',names=['Text','Emotion'])

print(df2.head())

print(df2.Emotion.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df2['Text'].tolist(), 
    df2['Emotion'].tolist(), 
    test_size=0.2, 
    stratify=df2['Emotion']
)


label2id = {label: i for i, label in enumerate(df2['Emotion'].unique())}
print(label2id)
id2label = {i: label for label, i in label2id.items()}
print(id2label)
train_labels = [label2id[label] for label in y_train]
test_labels = [label2id[label] for label in y_test]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True,num_labels=len(label2id))

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id))

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
train_dataset = EmotionDataset(train_encodings, train_labels)
test_dataset = EmotionDataset(test_encodings, test_labels)

dataloader_train = DataLoader(train_dataset, 
                              batch_size=32,shuffle=True)

dataloader_test = DataLoader(test_dataset, 
                              batch_size=32,shuffle=False)

optimizer = AdamW(model.parameters(), lr=1e-5)


print(test_dataset)


epochs = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
for epoch in tqdm(range(epochs)):
    model.train()  # Set model to training mode
    total_loss = 0
    train_acc=0
    total_samples=0

    for batch in dataloader_train:
        # Move batch to device (GPU/CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Get the loss
        
        logits=outputs.logits

        # Backward pass
        optimizer.zero_grad()  # Zero gradients from previous step
        loss.backward()        # Backpropagation

        # Optimizer step
        optimizer.step()       # Update model parameters

        total_loss += loss.item()
        
        
        y_pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item()
        total_samples += len(labels)

    # Log the loss for the epoch
    avg_loss = total_loss / len(dataloader_train)
    train_acc = train_acc / total_samples
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch+1}/{epochs} - Accuracy: {train_acc:.4f}")
    
    
     # Validation loop
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    total_val_loss = 0
    val_acc = 0
    total_val_samples = 0

    with torch.no_grad():  # Disable gradient computation (for validation)
        for batch in dataloader_test:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (no backpropagation in validation)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_val_loss += loss.item()

            # Calculate validation accuracy
            y_pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            val_acc += (y_pred_class == labels).sum().item()
            total_val_samples += len(labels)

    avg_val_loss = total_val_loss / len(dataloader_test)
    val_acc = val_acc / total_val_samples  # Validation accuracy
    print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")
    print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {val_acc:.4f}")

