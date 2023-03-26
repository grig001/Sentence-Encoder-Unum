import torch
import time
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset
from transformers import BertTokenizer
from scipy.stats import pearsonr, spearmanr
from transformers import (  BertForSequenceClassification, BertForMaskedLM, BertConfig,
                            BertTokenizer, BertForSequenceClassification, AdamW
                          )
from scipy import stats
from sklearn.metrics import mean_squared_error

device = torch.device('cuda')

batch_size = 8
learning_rate = 2e-5
num_epochs = 3

dataset = load_dataset('bookcorpus', split='train')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True), batched=True)

class MaskedLMDataset(torch.utils.data.Dataset):
    def init(self, tokenized_dataset, tokenizer):
        self.tokenized_dataset = tokenized_dataset
        self.tokenizer = tokenizer
    
    def len(self):
        return len(self.tokenized_dataset)
    
    def getitem(self, idx):
        tokens = self.tokenized_dataset[idx]['input_ids']
        labels = tokens.copy()
        for i, token in enumerate(tokens):
            if torch.rand(1) < 0.15:
                prob = torch.rand(1).item()
                if prob < 0.8:
                    tokens[i] = tokenizer.mask_token_id
                elif prob < 0.9:
                    tokens[i] = torch.randint(len(tokenizer), (1,)).item()

            if tokens[i] == tokenizer.mask_token_id:
                labels[i] = token
    
        tokens = torch.tensor(tokens)
        labels = torch.tensor(labels)
        return tokens, labels

train_dataset = MaskedLMDataset(tokenized_dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = MaskedLMDataset(tokenized_dataset, tokenizer)
valid_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

config = BertConfig()
model = BertForMaskedLM(config=config)
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, labels=labels)
            loss= outputs.loss
            total_loss+= loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {avg_loss:.4f}")


