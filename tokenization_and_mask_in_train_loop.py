import torch

from datasets import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertConfig
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertConfig

device = torch.device('cuda')


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
                                   max_length=128)

                tokens = tokens['input_ids']
                labels = tokens.copy()
                for k, token in enumerate(tokens):
                    # Randomly mask 15% of the tokens
                    if torch.rand(1) < 0.15:
                        prob = torch.rand(1).item()
                        # 80% of the time, replace the token with [MASK]
                        if prob < 0.8:
                            tokens[k] = tokenizer.mask_token_id
                        # 10% of the time, replace the token with a random token
                        elif prob < 0.9:
                            tokens[k] = torch.randint(
                                len(tokenizer), (1,)).item()
                        # 10% of the time, keep the original token

                    if tokens[k] == tokenizer.mask_token_id:
                        labels[k] = token

                    if isinstance(tokens[k], int) == False:
                        len_token = len(tokens[k])

                for i in range(len(tokens)):

                    x = tokens[i]
                    if isinstance(x, int):
                        arr = list([x])
                        tokens[i] = len_token * arr

                tokens = torch.tensor(tokens)
                labels = torch.tensor(labels)

                tokens = tokens.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(tokens, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            print(
                f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")

            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, inputs in enumerate(valid_loader):

                    tokens = tokenizer(inputs['text'],
                                       padding=True,
                                       truncation=True,
                                       max_length=128)

                    inputs = tokens['input_ids']
                    labels = inputs.copy()

                    inputs = torch.tensor(inputs)
                    inputs = inputs.to(device)

                    labels = torch.tensor(labels)
                    labels = labels.to(device)

                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {avg_loss:.4f}")


if __name__ == "__main__":

    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 7
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_data = []
    dataset = load_dataset('bookcorpus', split='train', streaming=True)
    train_stream = dataset
    for i, item in enumerate(train_stream):
        if i >= 100:
            break

        train_data.append({'text': item['text']})

    valid_data = []
    valid_stream = dataset
    for i, item in enumerate(train_stream):
        if i <= 100:
            continue
        elif i > 120:
            break
        else:
            valid_data.append({'text': item['text']})

    dataset = Dataset.from_list(train_data)
    print(dataset)

    valid_dataset = Dataset.from_list(valid_data)
    print(valid_dataset)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True)

    config = BertConfig()
    model = BertForMaskedLM(config=config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    print('--------------------------------------------------------------------------')
    klir = Train(model=model, num_epochs=num_epochs, tokenizer=tokenizer,
                 train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer)

    klir.train()
