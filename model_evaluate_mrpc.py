import tensorflow as tf
import torch
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np 
from pathlib import Path
from transformers import DistilBertTokenizerFast
import torch as T
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from transformers import logging  # to suppress warnings

device = T.device('cuda')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Hello I'm a [MASK] model.")


dataset = load_dataset('glue', 'mrpc')


def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)


import torch
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)



def encode_sentence_pair(s1, s2, tokenizer):
    # Tokenize the input sentences
    tokens = tokenizer(s1, s2, return_tensors="pt", padding=True, truncation=True).to('cuda')

    # Encode the input sentences as input IDs and attention masks
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    return input_ids, attention_mask



def predict_paraphrase(s1, s2, tokenizer, model):
    # Encode the input sentences as input IDs and attention masks
    input_ids, attention_mask = encode_sentence_pair(s1, s2, tokenizer)

    # Make a prediction using the pre-trained model
    with torch.no_grad():
        logits = model(input_ids, attention_mask)[0]

    # Apply the softmax function to convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Return the probability that the two sentences are paraphrases
    return probabilities[0][1].item()


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'], 
)

trainer.train()


s1 = "The cat is on the mat."
s2 = "The cat is resting on the mat."
prob = predict_paraphrase(s1, s2, tokenizer, model)
print(prob)

from sklearn.metrics import f1_score

curr = 0
y_pred = []
y_true = []
for i in range(len(dataset['test'])):
  s1 = dataset['test'][i]['sentence1']
  s2 = dataset['test'][i]['sentence2']
  prob = predict_paraphrase(s1, s2, tokenizer, model)
  if prob > 0.6:
    prob = 1
  else:
     prob = 0
  y_pred.append(prob)
  y_true.append(dataset['test'][i]['label'])
  if prob == dataset['test'][i]['label']:
    curr +=1

accuracy = curr/len(dataset['test'])
f1 = f1_score(y_true, y_pred)
print('accuracy:',accuracy,'f1: {f1}')
print('accuracy:', round(accuracy,3), 'f1:', round(f1,3))
