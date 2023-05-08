import sys
import torch
import wandb
import random
import numpy as np
import torch.distributed as dist


from datasets import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertConfig

device = torch.device('cuda:0')


class Train:
    def __init__(self, model, num_epochs, tokenizer, train_loader, valid_loader,
                 optimizer) -> None:

        self.model = model
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimazer = optimizer
        # self.local_world_size = local_world_size
        # self.local_rank = local_rank
        # self.ddp_model = ddp_model

    def train(self):

        for epoch in range(num_epochs):

            total_loss = 0
            model.train()

            # dist.barrier()

            for i, inputs in enumerate(train_loader):

                # batch = tuple(t.to(args.device) for t in batch)

                tokens = tokenizer(inputs['text'],
                                   padding=True,
                                   truncation=True,
                                   max_length=512)

                tokens = tokens['input_ids']
                labels = tokens.copy()

                for k, token in enumerate(tokens):

                    starting_token = token.copy()

                    x = random.sample(
                        range(len(token)), int(len(token)))

                    mask_1 = int(0.8 * len(x))
                    mask_2 = int(0.9 * len(x))

                    for ind in range(mask_1):
                        token[x[ind]] = tokenizer.mask_token_id

                    for ind in range(mask_1, mask_2):
                        token[x[ind]] = torch.randint(
                            len(tokenizer), (1,)).item()

                    labels[k] = token
                    print(f'starting token ---> {starting_token}')
                    print(f'ending token ---> {token}')
                    print(
                        f'comparing starting and finishing tokens ---> {np.array(token) == np.array(starting_token)}')

                    print(
                        '______________________________________________________________________')

                tokens = torch.tensor(tokens).to(device)
                labels = torch.tensor(labels).to(device)

                optimizer.zero_grad()

                outputs = model(tokens, labels=labels)
                # outputs = model(*batch)

                loss = outputs.loss
                # loss = outputs[0]
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 500 == 0:
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
                                       max_length=512)

                    inputs = tokens['input_ids']
                    labels = inputs.copy()

                    inputs = torch.tensor(inputs).to(device)
                    labels = torch.tensor(labels).to(device)

                    outputs = model(inputs, labels=labels)

                    loss = outputs.loss
                    total_loss += loss.item()

                val_avg_loss = total_loss / len(train_loader)
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {val_avg_loss:.4f}")

            wandb.log({"loss": avg_loss, "valid_loss": val_avg_loss})


if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open("MyOutput.log", "w")
    sys.stdout = log_file

    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, metavar='N')
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    local_world_size, local_rank = args.local_world_size, args.local_rank

    args.is_master = args.local_rank == 0
    args.device = torch.cuda.device(args.local_rank)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    SEED = 42
    # torch.cuda.manual_seed_all(SEED)

    # print(n)
    # print(device_ids)
    # print()
    # print()

    batch_size = 8
    learning_rate = 2e-4
    num_epochs = 10
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )
    ddp_model = DDP(model, device_ids)

# _________________________________________________________________________________________________________
    wandb.init(project="Sentence-Encoder-Unum",
               config=config)

    wandb.watch(model, log_freq=100)
# ___________________________________________________________________________________________________________

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print('--------------------------------------------------------------------------------------------')
    MyClass = Train(
        model=model, num_epochs=num_epochs, tokenizer=tokenizer,
        train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer,
    )

    MyClass.train()

    sys.stdout = old_stdout
    log_file.close()
