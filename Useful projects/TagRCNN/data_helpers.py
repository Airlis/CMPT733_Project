import pandas as pd
import re
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(filepath):
    posts_df = pd.read_csv(filepath, encoding='utf8', index_col=0)
    x_text = [clean_str(text) for text in posts_df['Text']]
    y = [literal_eval(item) for item in posts_df['Tags'].tolist()]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    return [x_text, y]


def reduce_padding(x_batch):
    max_length = max([len(np.nonzero(row)[0]) for row in x_batch])
    return np.array(x_batch)[:, :max_length]


def batch_iter(data, batch_size, num_epochs, shuffle=True, dynamic=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # print(num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if dynamic is True:
                x_batch, y_batch = zip(*shuffled_data[start_index:end_index])
                x_batch = reduce_padding(x_batch)
                yield list(zip(x_batch, y_batch))
            else:
                yield shuffled_data[start_index:end_index]
        yield True
