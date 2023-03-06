import tensorflow as tf
import torch
import pandas as pd
from transformers import pipeline
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


from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

# Load the STS-B dataset
from datasets import load_dataset
dataset = load_dataset('glue', 'stsb', split='train')

#tokenized_dataset = stsb.map(tokenize_function, batched=True)

# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#stsb = load_dataset('glue', 'stsb')


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

# Load the STS-B dataset
from datasets import load_dataset
dataset = load_dataset('glue', 'stsb', split='train')

# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def collate_fn(examples):
    # tokenize the inputs
    inputs = tokenizer([example['sentence1'] for example in examples],
                       [example['sentence2'] for example in examples],
                       padding=True,
                       truncation=True,
                       max_length=128,
                       return_tensors='pt')
    
    # convert the labels to tensors
    labels = torch.tensor([example['label'] for example in examples])
    
    # create a dictionary of tensors
    batch = {'input_ids': inputs['input_ids'],
             'attention_mask': inputs['attention_mask'],
             'token_type_ids': inputs['token_type_ids'],
             'label': labels}
    
    return batch


# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.MSELoss()
train_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
for epoch in range(3):
    for batch in train_loader:
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        labels = torch.tensor(batch['label']).to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits.squeeze(-1), labels.float())
        loss.backward()
        optimizer.step()


from scipy.stats import pearsonr, spearmanr


stsb = load_dataset('glue', 'stsb')


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
        model.to(device)
        logits = round(model(input_ids, attention_mask)[0].item(),2)
        # outputs.logits.squeeze(-1)

    # Return the probability that the two sentences are paraphrases
    return logits

s1 = "The cat is on the mat."
s2 = "The cat is not on the mat."
prob = predict_paraphrase(s1, s2, tokenizer, model)
print(prob)


curr = 0
y_pred = []
y_true = []
for i in range(len(stsb['test'])):
  s1 = stsb['test'][i]['sentence1']
  s2 = stsb['test'][i]['sentence2']
  prob = predict_paraphrase(s1, s2, tokenizer, model)
  y_pred.append(prob)
  y_true.append(stsb['validation'][i]['label'])

k , i = pearsonr(y_pred, y_true)
corr , pi = spearmanr(y_pred, y_true)
print(corr)



from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze(-1)
    mse = mean_squared_error(labels, preds)
    return {
        'mse': mse,
        'pearson_corr': np.corrcoef(labels, preds)[0,1],
        'spearman_corr': stats.spearmanr(labels, preds)[0],
    }
eval_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
model.to('cpu')
model.eval()
with torch.no_grad():
    preds = []
    for batch in eval_loader:
        input_ids = batch['input_ids'].to('cpu')
        attention_mask = batch['attention_mask'].to('cpu')
        outputs = model(input_ids, attention_mask=attention_mask)
        preds.extend(outputs.logits.squeeze(-1))
        print('Please wait :)')
    labels = dataset['label']
    pearson_corr = np.corrcoef(labels, preds)[0,1]
    spearman_corr = stats.spearmanr(labels, preds)[0]
    mse = mean_squared_error(labels, preds)
    print(f'Pearson correlation: {pearson_corr:.4f}')
    print(f'Spearman correlation: {spearman_corr:.4f}')
    print(f'Mean squared error: {mse:.4f}')
