import torch
import time
import numpy as np

from sklearn.metrics import f1_score
from datasets import load_dataset
from transformers import BertTokenizer
from scipy.stats import pearsonr, spearmanr
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from scipy import stats
from sklearn.metrics import mean_squared_error

device = torch.device('cuda')

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
    probabilities = torch.argmax(logits, dim=1)

    # Return the probability that the two sentences are paraphrases
    return probabilities.item()

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

def STS_B(num_epoch, tokenize_function):

    # Load the BERT model
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    dataset = load_dataset('glue', 'stsb', split='train')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)

    # Fine-tune the model
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    for _ in range(num_epoch):
        for batch in train_loader:
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            labels = torch.tensor(batch['label']).to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits.squeeze(-1), labels.float())
            loss.backward()
            optimizer.step()
   

def MRPC(epoch): 

    dataset = load_dataset('glue', 'mrpc')

    # # Load the BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Fine-tune the model
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size=16, shuffle=True, collate_fn=collate_fn)

    training_acc = []
    training_loss = []

    for epoch in range(epoch):
        train_acc = 0.0
        train_loss = 0.0

        model.train()

        for batch in train_loader:

            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            labels = torch.tensor(batch['label']).to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # outputs = torch.softmax(outputs, dim=-1)[0][1].item()
            # print(outputs.logits.squeeze(-1))
            pred = torch.argmax(outputs.logits,1)
            loss = loss_fn(outputs.logits.squeeze(-1),labels)
            loss.backward()
            optimizer.step()
            train_acc += (pred == labels).sum().item()
            train_loss += loss.item()

            # len(train_loader))
            training_loss.append(train_loss/len(train_loader))
            training_acc.append(train_loss/len(train_loader))

        test_acc = 0.0
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val in test_loader:
                input_ids = torch.tensor(val['input_ids']).to(device)
                attention_mask = torch.tensor(val['attention_mask']).to(device)
                labels = torch.tensor(val['label']).to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fn(outputs.logits.squeeze(-1),labels)

                pred = torch.argmax(outputs.logits,1)

                test_acc += (pred == labels).sum().item()
                test_loss += loss.item()
                
        print("Epochs:{}, Training Accuracy:{:.2f}, Training Loss:{:.2f}, Validation Accuracy:{:.2f}, Validation Loss:{:.2f}".
                    format(epoch+1, train_acc/len(dataset['train'])*100, train_loss/len(train_loader),
                            test_acc/len(dataset['test'])*100, test_loss/len(test_loader)))



class Evaluation:
    def __init__(self, model, dataset_name, number_epoch) -> None:

        dataset = load_dataset('glue', dataset_name)
        self.dataset_name = dataset_name
        self.epoch = number_epoch
        self.model = model
        self.data = dataset
        self.tokenized_dataset = dataset.map(tokenize_function, batched=True)

    def evaluate(self):

        if self.dataset_name == 'mrpc':
            
            curr = 0
            y_pred = []
            y_true = []
            for i in range(len(self.data['test'])):
                s1 = self.data['test'][i]['sentence1']
                s2 = self.data['test'][i]['sentence2']
                prob = predict_paraphrase(s1, s2, tokenizer, self.model)

                y_pred.append(prob)
                y_true.append(self.data['test'][i]['label'])
                if prob == self.data['test'][i]['label']:
                    curr += 1

            accuracy = curr/len(self.data['test'])
            f1 = f1_score(y_true, y_pred)

            return (f'accuracy:,{round(accuracy,3)},  f1 score {round(f1,3)}')

        elif self.dataset_name == 'stsb':
            
            stsb = load_dataset('glue', 'stsb')
            #curr = 0
            y_pred = []
            y_true = []
            for i in range(len(stsb['test'])):
                s1 = stsb['test'][i]['sentence1']
                s2 = stsb['test'][i]['sentence2']
                prob = predict_paraphrase(s1, s2, tokenizer, self.model)
            
                y_pred.append(prob)
                y_true.append(stsb['validation'][i]['label'])

            k , i = pearsonr(y_pred, y_true)
            corr , pi = spearmanr(y_pred, y_true)

            print(corr)
            #print(round(corr, 3))

            def compute_metrics(pred):
                labels = pred.label_ids
                preds = pred.predictions.squeeze(-1)
                mse = mean_squared_error(labels, preds)
                return {
                    'mse': mse,
                    'pearson_corr': np.corrcoef(labels, preds)[0,1],
                    'spearman_corr': stats.spearmanr(labels, preds)[0],
                }
            
            eval_loader = torch.utils.data.DataLoader(self.tokenized_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
            model.to(device)
            model.eval()
            with torch.no_grad():
                preds = []
                for batch in eval_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds.extend(outputs.logits.squeeze(-1))
                    #print('Please wait :)')

                labels = self.data['label']
                pearson_corr = np.corrcoef(labels, preds)[0,1]
                spearman_corr = stats.spearmanr(labels, preds)[0]
                mse = mean_squared_error(labels, preds)
                # print(f'Pearson correlation: {pearson_corr:.4f}')
                # print(f'Spearman correlation: {spearman_corr:.4f}')
                # print(f'Mean squared error: {mse:.4f}')
                return (f'Pearson correlation: {pearson_corr}, Spearman correlation: {spearman_corr},Mean squared error: {mse}')
        else:
            return
       

    def fine_tune(self):

        if self.dataset_name == 'mrpc':
            MRPC(self.epoch)
        elif self.dataset_name == 'stsb':
            STS_B(self.epoch, tokenize_function)
        else:
            pass
        # training_args = TrainingArguments(
        # output_dir='./results',          # output directory
        # num_train_epochs=0.1,              # total number of training epochs
        # per_device_train_batch_size=16,  # batch size per device during training
        # per_device_eval_batch_size=64,   # batch size for evaluation
        # warmup_steps=500,                # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,               # strength of weight decay
        # logging_dir='./logs',            # directory for storing logs
        # logging_steps=10,
        # evaluation_strategy='epoch',
        # )

        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=self.tokenized_dataset['train'],
        #     eval_dataset=self.tokenized_dataset['validation'], 
        # )

        # trainer.train()

        self.model = model

if __name__ == "__main__":

    start = time.time()

    dataset_name = 'stsb'
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    a = Evaluation(model, dataset_name, number_epoch=3)
    
    if dataset_name == 'stsb':
        accuracy_before = a.evaluate()
    elif dataset_name == 'mrpc':
        accuracy_before = a.evaluate()[0]
        f1_score_before = a.evaluate()[1]
    else:
        pass
    
    a.fine_tune()
    print()
    if dataset_name == 'stsb':
        print(f'accuracy before fine-tuning: {accuracy_before}')
        print(f'accuracy after  fine-tuning: {a.evaluate()}')
    elif dataset_name == 'mrpc':
        print(f'accuracy before  fine-tuning: {accuracy_before}, f1 score before fine-tuning: {f1_score_before}')
        print(f'accuracy after  fine-tuning: {a.evaluate()[0]}, f1 score after  fine-tuning: {a.evaluate()[1]}')    
    else:
        pass
 
    print()    

    end = time.time()
    worked_time = end - start
    if worked_time < 60:
        print(f'The code working time is {worked_time} second')
    elif worked_time < 3600:
        print(f'The code working time is {round(worked_time / 60, 4)} minute')
    else:
        print(f'The code working time is {round(worked_time / 3600, 4)} hour')

