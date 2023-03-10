import tensorflow as tf
import torch
import pandas as pd
from sklearn.metrics import f1_score
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
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import time

device = T.device('cuda')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'],
                      truncation=True, padding='max_length', max_length=128)


def encode_sentence_pair(s1, s2, tokenizer):
    # Tokenize the input sentences
    tokens = tokenizer(s1, s2, return_tensors="pt", padding=True, truncation=True).to(device)

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



class Evaluation:
    def __init__(self, model, dataset_name) -> None:

        dataset = load_dataset('glue', dataset_name)
        self.model = model
        self.data = dataset
        self.tokenized_dataset = dataset.map(tokenize_function, batched=True)

    def evaluate(self):

        curr = 0
        y_pred = []
        y_true = []
        for i in range(len(self.data['test'])):

            s1 = self.data['test'][i]['sentence1']
            s2 = self.data['test'][i]['sentence2']
            prob = predict_paraphrase(s1, s2, tokenizer, model)

            if prob > 0.6:
                prob = 1
            else:
                prob = 0

            y_pred.append(prob)
            y_true.append(self.data['test'][i]['label'])

            if prob == self.data['test'][i]['label']:
                curr += 1

        accuracy = curr / len(self.data['test'])
        f1 = f1_score(y_true, y_pred)

        return (round(accuracy,3), round(f1,3))


    def fine_tune(self):

        # finetuning

        training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=15,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'], 
        )

        trainer.train()

        self.model = model
        #print(f'fine-tuning ---  accuracy: {self.evaluate()[0]}, f1 score: {self.evaluate()[1]}')

if __name__ == "__main__":

    start = time.time()

    dataset_name = 'mrpc'
    #model = BertModel.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    a = Evaluation(model, dataset_name)
    
    # zero-shot evaluation
    accuracy_before = a.evaluate()[0]
    f1_score_before = a.evaluate()[1]

    # # For evaluation with fine-tuning 
    a.fine_tune()
    print()
    print(f'accuracy before fine-tuning: {accuracy_before}, f1 score before fine-tuning: {f1_score_before}')
    print(f'accuracy after  fine-tuning: {a.evaluate()[0]}, f1 score after  fine-tuning: {a.evaluate()[0]}')
    print()    
    # array_for_evaluate = np.empty((0,2))
    # number_epoch = 5
    # for i in range(number_epoch):

    #     a.fine_tune()
    #     array_for_evaluate = np.insert(array_for_evaluate,i,(a.evaluate()[0],a.evaluate()[1]), axis=0)
        
    # for i in range(number_epoch):
    #     print(f'Evaluation after {i+1} epoch')
    #     print(f'accuracy: {array_for_evaluate[i][0]}, f1 score:{array_for_evaluate[i][1]}')

    end = time.time()
    worked_time = end - start
    if worked_time < 60:
        print(f'The code working time is {worked_time} second')
    elif worked_time < 3600:
        print(f'The code working time is {round(worked_time / 60, 4)} minute')
    else:
        print(f'The code working time is {round(worked_time / 3600, 4)} hour')
