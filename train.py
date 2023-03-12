import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import tensorflow as tf
import torch
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
import numpy as np 
from pathlib import Path
from transformers import DistilBertTokenizerFast
import torch as T
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DistilBertForSequenceClassification
from transformers import logging  # to suppress warnings
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import nlpaug.augmenter.char as nac
#import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from nlpaug.util import Action
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Set the device to use for training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set the hyperparameters for training
batch_size = 8
learning_rate = 2e-5
num_epochs = 5


dataset = load_dataset( 'bookcorpus', split='train', streaming=True)

from datasets import load_dataset
data = []
XY = []

dataset = load_dataset( 'bookcorpus', split='train', streaming=True)
train_stream = dataset
for i, item in enumerate(train_stream):
    if i >= 1000:
        break
    
    XY.append({'text': item['text']})
    # XY.append(item)
    data.append(item['text'])
    # print(item)

from datasets import Dataset
dataset = Dataset.from_list(XY)



# Tokenize the dataset using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True), batched=True)

# Prepare the data for training
class MaskedLMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, tokenizer):
        self.tokenized_dataset = tokenized_dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.tokenized_dataset)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_dataset[idx]['input_ids']
        labels = tokens.copy()
        for i, token in enumerate(tokens):
            # Randomly mask 15% of the tokens
            if torch.rand(1) < 0.15:
                prob = torch.rand(1).item()
                # 80% of the time, replace the token with [MASK]
                if prob < 0.8:
                    tokens[i] = tokenizer.mask_token_id
                # 10% of the time, replace the token with a random token
                elif prob < 0.9:
                    tokens[i] = torch.randint(len(tokenizer), (1,)).item()
                # 10% of the time, keep the original token
            # Set the labels for the masked tokens
            if tokens[i] == tokenizer.mask_token_id:
                labels[i] = token
    
        # Convert the tokens and labels to PyTorch tensors
        tokens = torch.tensor(tokens)
        labels = torch.tensor(labels)
        return tokens, labels

train_dataset = MaskedLMDataset(tokenized_dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the BERT model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Train the model
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

# Evaluate the model on the validation set
model.eval()
with torch.no_grad():
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {avg_loss:.4f}")

