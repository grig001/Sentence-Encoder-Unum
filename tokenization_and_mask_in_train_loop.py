import sys
import torch
import wandb
import random
import numpy as np

from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertConfig

device = torch.device('cuda:0')


class Train:
    def __init__(self, model, num_epochs, tokenizer, train_loader, valid_loader, optimizer) -> None:

        self.model = model
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimazer = optimizer

    def train(self):

        for epoch in range(num_epochs):

            total_loss = 0
            model.train()

            for i, inputs in enumerate(train_loader):

                tokens = tokenizer(inputs['text'],
                                   padding=True,
                                   truncation=True,
                                   max_length=512)

                tokens = tokens['input_ids']
                labels = tokens.copy()

                for k, token in enumerate(tokens):

                    starting_token = token.copy()

                    if torch.rand(1) < 0.8:

                        prob = torch.rand(1).item()

                        if prob < 0.2:

                            x = random.sample(
                                range(len(token)), int(0.2 * len(token)))

                            for q in x:
                                token[q] = tokenizer.mask_token_id

                        elif prob < 0.4:

                            x = random.sample(
                                range(len(token)), int(0.4 * len(token)))

                            for q in x:
                                token[q] = torch.randint(
                                    len(tokenizer), (1,)).item()

                    # print(tokenizer.mask_token_id) == 103

                    labels[k] = token
                    print(f'starting token ---> {starting_token}')
                    print(f'ending token ---> {token}')
                    print(
                        f'comparing starting and finishing tokens {np.array(token) == np.array(starting_token)}')

                    print(
                        '______________________________________________________________________')
                tokens = torch.tensor(tokens).to(device)
                labels = torch.tensor(labels).to(device)

                optimizer.zero_grad()

                outputs = model(tokens, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 500 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            print(
                f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")

            wandb.log({"loss": avg_loss})

            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, inputs in enumerate(valid_loader):

                    tokens = tokenizer(inputs['text'],
                                       padding=True,
                                       truncation=True,
                                       max_length=512)

                    inputs = tokens['input_ids']
                    labels = inputs.copy()

                    inputs = torch.tensor(inputs).to(device)
                    labels = torch.tensor(labels).to(device)

                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {avg_loss:.4f}")


if __name__ == "__main__":

    batch_size = 16
    learning_rate = 2e-4
    num_epochs = 10
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    old_stdout = sys.stdout

    log_file = open("MyOutput.log", "w")

    sys.stdout = log_file

    with open('my_data.csv') as f:
        lines = f.readlines()

    lenght_dataset = len(lines)

    start = int(0.3 * lenght_dataset)

    train = lines[start:]
    valid = lines[:start]

    train_data = []
    for item in train:
        train_data.append({'text': item})

    valid_data = []
    for item in valid:
        valid_data.append({'text': item})

    train_dataset = Dataset.from_list(train_data)
    print(train_dataset)

    valid_dataset = Dataset.from_list(valid_data)
    print(valid_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True)

    config = BertConfig()
    model = BertForMaskedLM(config=config)
    model.to(device)
# _________________________________________________________________________________________________________
    wandb.init(project="Sentence-Encoder-Unum",

               config=config)

    wandb.watch(model, log_freq=100)
# ___________________________________________________________________________________________________________
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    print('----------------------------------------------------------------------------------')
    MyClass = Train(model=model, num_epochs=num_epochs, tokenizer=tokenizer,
                    train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer)

    MyClass.train()

    sys.stdout = old_stdout
    log_file.close()

    # wandb.finish()
