import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def helper(text):
    noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m', '\n']

    for noise in noises:
        text = text.replace(noise, '')
    text = text.lower()

    return text


def preprocess():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")
    label_cols = df.columns[2:]
    df['text'] = df['text'].apply(lambda x: helper(x))
    df['offensive'] = df[label_cols].sum(axis=1)
    df['offensive'] = df['offensive'].apply(lambda x: 1 if x>0 else 0)

    # Check percentage of polite texts in the whole dataset, around 90%
    percentage = (1-df['offensive'].sum()/df.shape[0])*100
    print('Percentage of polite texts in the whole dataset is: %f' % percentage)

    df.to_csv(config.PROCESSED_FILE)

    return df


def preprocess2():

    data1 = pd.read_csv('data/off1.tsv', sep='\t', header=0)
    data1 = data1[['tweet','subtask_a']]

    data2 = pd.read_csv('data/off2.txt', sep='\t', header=None)
    data2.columns = ["tweet", "subtask_a", "subtask_b", "subtask_c"]
    data2 = data2[['tweet','subtask_a']]

    df = pd.concat([data1,data2])

    df['text'] = df['tweet'].apply(lambda x: helper(x))
    df['offensive'] = df['subtask_a'].apply(lambda x: 1 if x=='OFF' else 0)
    df = df[['text', 'offensive']]

    percentage = (1-df['offensive'].sum()/df.shape[0])*100
    print('Percentage of non-offensive texts in the whole dataset is: %f' % percentage)

    df.to_csv('data/processed_dataset2.csv')

    return df

def combine():
    df1 = pd.read_csv('data/processed_dataset.csv')
    df1 = df1[['text', 'offensive']]
    df2 = pd.read_csv('data/processed_dataset2.csv')

    df = pd.concat([df1,df2])
    df['text'] = df['text'].apply(lambda x: helper(x))
    df['offensive'] = df['offensive']
    df = df[['text', 'offensive']]

    percentage = (1-df['offensive'].sum()/df.shape[0])*100
    print('Percentage of non-offensive texts in the whole dataset is: %f' % percentage)

    df.to_csv('data/processed_train_data.csv')
    print(df.head())
    return df

def run():

    #df = preprocess()
    #df = pd.read_csv(config.PROCESSED_FILE)
    df = pd.read_csv('data/processed_train_data.csv')
    #print(df.columns)

    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size=0.3,
        random_state=32,
        stratify=df.offensive.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df_train.text.values,
        target=df_train.offensive.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.text.values,
        target=df_valid.offensive.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    model = BERTBaseUncased()
    model.to(config.DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)


    best_accuracy = 0
    for epoch in range(5):
        engine.train_fn(train_data_loader, model, optimizer, config.DEVICE, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, config.DEVICE)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
