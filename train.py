import torch

from datasets import Dataset
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW

device = torch.device('cuda')
batch_size = 16
learning_rate = 2e-5
num_epochs = 5

data = []
XY = []

dataset = load_dataset('bookcorpus', split='train', streaming=True)
train_stream = dataset

for i, item in enumerate(train_stream):
    if i >= 1000:
        break

    XY.append({'text': item['text']})
    data.append(item['text'])

dataset = Dataset.from_list(XY)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True), batched=True)

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

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)

def train_model(num_epoch):
    for epoch in range(num_epoch):
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
    return epoch

def evaluate_model(epoch):
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


if __name__ == "__main__":

    epoch = train_model(num_epoch=num_epochs)
    evaluate_model(epoch=epoch)

