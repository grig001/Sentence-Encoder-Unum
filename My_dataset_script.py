import csv

from datasets import Dataset
from datasets import load_dataset

train_data = []
dataset = load_dataset('bookcorpus', split='train', streaming=True)

for i, item in enumerate(dataset):
    if i >= 5e2:
        break

    if i % 10**2 == 0:
        print(f'{i} iteration is done')

    train_data.append({'text': item['text']})


data = Dataset.from_list(train_data)

with open('my_data.csv', 'w') as file:
    for item in data:
        file.write(item['text'] + '\n')

file.close()
