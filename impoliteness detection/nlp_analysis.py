import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup

import config
import torch

from model import BERTBaseUncased
import torch.nn as nn


def clean_text(text):
    soup = BeautifulSoup(text)
    clean_text = soup.get_text()
    return clean_text


# Prepare data 'Posts.xml'
def get_posts():
    root = ET.parse('Posts.xml').getroot()
    tags = {"tags":[]}
    for elem in root:
        tag = {}
        tag["Id"] = elem.attrib['Id']

        try:
            tag["ParentId"] = elem.attrib['ParentId']
        except:
            tag["ParentId"] = 0

        tag["Body"] = elem.attrib['Body']
        tags["tags"].append(tag)

    df_posts = pd.DataFrame(tags["tags"])

    df_posts['Body'] = df_posts['Body'].apply(lambda x: clean_text(x))

    total = df_posts.shape[0]

    print('Total is:')

    print(total)

    return df_posts



def get_comments():
# Prepare data 'Comments.xml'
    root = ET.parse('Comments.xml').getroot()
    tags = {"tags":[]}
    for elem in root:
        tag = {}
        tag["Id"] = elem.attrib['Id']
        tag["PostId"] = elem.attrib['PostId']
        tag["Text"] = elem.attrib['Text']
        tags["tags"].append(tag)

    df_comments = pd.DataFrame(tags["tags"])

    total = df_comments.shape[0]

    print('Total is:')

    print(total)

    return df_comments



def text_prediction(sentence, model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(config.DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(config.DEVICE, dtype=torch.long)
    mask = mask.to(config.DEVICE, dtype=torch.long)

    outputs = MODEL(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()

    pred = outputs[0][0]
    #print(pred)
    # pred: the possibility of impolite -> column 'offensive'
    if pred>=0.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    MODEL = BERTBaseUncased()
    MODEL = nn.DataParallel(MODEL)
    #MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    # if run in on 'cpu', use below
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    MODEL.to(config.DEVICE)
    MODEL.eval()


    df_posts = get_posts()
    df_posts['offensive'] = df_posts.Body.apply(lambda x: text_prediction(x, MODEL))
    df_posts2 = df_posts[df_posts['offensive'] == 1]

    percentage = df_posts2.shape[0]/df_posts.shape[0]*100

    print('----------------')
    print('Posts.xml has been processed!')
    print('Percentage is: ' + str(percentage) + '%')


    df_comments = get_comments()
    df_comments['offensive'] = df_comments.Text.apply(lambda x: text_prediction(x, MODEL))
    df_comments2 = df_comments[df_comments['offensive'] == 1]

    percentage = df_comments2.shape[0]/df_comments.shape[0]*100

    print('----------------')
    print('Comments.xml has been processed!')
    print('Percentage is: ' + str(percentage) + '%')
