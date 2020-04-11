from flask import Flask, render_template, url_for, redirect, jsonify, request
import json

import time
import config
from config import app
import torch
import functools
import torch.nn as nn
from model_helper import BERTBaseUncased

import os, sys
import tensorflow as tf
import numpy as np
import data_helpers
import pandas as pd
from rcnn import RCNN
from tensorflow.contrib import learn
####################### Import ########################


def generate_tags(dataset, title, body):
    # Data Preparation
    # ==================================================

    path = os.path.join('model', dataset)
    text = data_helpers.preprocess(title,body)
    x_text = [data_helpers.clean_str(text)]

    # Restore vocab file
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(
        path, 'vocab'))

    x = np.array(list(vocab_processor.fit_transform(x_text)))
    tags_df = pd.read_csv(os.path.join(path,'tags_df.csv'), encoding='utf8', index_col=0)
    tag_list = tags_df['TagName'].tolist()

    # prediction
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=3,
            inter_op_parallelism_threads=3)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            rcnn = RCNN(
                num_classes=len(tag_list),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=100,
                hidden_units=100,
                context_size=50,
                max_sequence_length=x.shape[1])
                # l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(rcnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            # Loading checkpoint
            save_path = os.path.join(path, "model")
            saver.restore(sess, save_path)

            # predict
            sequence_length = [len(sample) for sample in x]
            feed_dict = {
                rcnn.X: x,
                rcnn.sequence_length: sequence_length,
                # rcnn.max_sequence_length: max_sequence_length,
                rcnn.dropout_keep_prob: 1.0
            }
            prediction = sess.run([rcnn.predictions],feed_dict)[0][0]
            idx = prediction.argsort()[-5:][::-1]
            tags = [tag_list[i] for i in idx]
    return tags

# function to detect impolite language usage, output: probabilty of impoliteness
def sentence_prediction(sentence, model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())
    print(review)
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

    outputs = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


def create_new_post(dataset):
    if request.method == 'POST':
        if request.form['action'] == 'tag':
            title = request.form['title']
            body = request.form['body']
            tags = generate_tags(dataset,title, body)
            return render_template('layout/default.html',
                        content=render_template('pages/post.html',title=title,body=body,results='; '.join(tags)) )

        elif request.form['action'] == 'polite':
            title = request.form['title']
            body = request.form['body']
            sentence = body+title

            start_time = time.time()
            negative_prediction = sentence_prediction(sentence, model)
            """
            if negative_prediction>0.5:
                res = 'Rude'
            else:
                res ='Not Rude'
            positive_prediction = 1 - negative_prediction
            response = {}
            response["response"] = {
                'positive': str(positive_prediction),
                'negative': str(negative_prediction),
                'result': res,
                'sentence': str(sentence),
                'time_taken': str(time.time() - start_time)
            }
            print(str(time.time() - start_time))
            """
            return render_template('layout/default.html',
                                content=render_template('pages/result.html',title=title,body=body, prediction=negative_prediction) )

    else:
        return render_template('layout/default.html',
                            content=render_template('pages/post.html') )

@app.errorhandler(404)
def page_not_found(e):
   return redirect(url_for('bicycles'))

@app.route('/bicycles', methods=['GET','POST'])
def bicycles():
    return create_new_post("bicycles")

@app.route('/gaming', methods=['GET','POST'])
def gaming():
    return create_new_post("gaming")

@app.route('/movies', methods=['GET','POST'])
def movies():
    return create_new_post("movies")

@app.route('/music', methods=['GET','POST'])
def music():
    return create_new_post("music")

@app.route('/stackoverflow', methods=['GET','POST'])
def programing():
    return create_new_post("stackoverflow")


if __name__ == '__main__':

    # load nlp Bert model with model2.bin weights
    model = BERTBaseUncased()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.to(config.DEVICE)
    model.eval()

    # add para debug=True here to enable refresh update
    app.run(debug=True)
