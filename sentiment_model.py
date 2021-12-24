import pandas as pd
from transformers import RobertaTokenizer, TrainingArguments
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import DatasetDict,load_dataset,Dataset
from transformers import Trainer
from transformers import RobertaForSequenceClassification
import torch
import numpy as np
import pandas as pd
import time
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# data = pd.read_csv('senti.csv')
#
#
# positive = data[data.label== "positive"]
#
# negative = data[data.label== "negative"]
#
#
# data = positive.sample(frac=0.06, random_state=404)
# data = data.append(negative.sample(frac=0.06, random_state=404), ignore_index=True)
#
# data = data.sample(frac=1, random_state=404)
# data["label"] = data["label"].replace({"positive":1, "negative":0})
# print(data)

# data.to_pickle("imdb.pkl")
dataset = load_dataset('pandas', data_files='imdb.pkl', split="train")
dataset.remove_columns_(column_names = ['__index_level_0__'])
print(dataset)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_testvalid = dataset.train_test_split(test_size=0.1)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
all_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
print(all_dataset)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels = 2)

training_args = TrainingArguments(
    output_dir='./results'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    tokenizer=tokenizer
)
trainer.train()

trainer.save_model("trained_on_setiment_imbd")

predictions = trainer.predict(tokenized_datasets["test"])
print(predictions.predictions.shape, predictions.label_ids.shape)
y_true = tokenized_datasets["test"]["label"]
y_pred = np.argmax(predictions.predictions, axis=-1)
confusion_matrix(y_true, y_pred)

metric = load_metric("accuracy")
metric.compute(predictions=y_pred, references=y_true)
