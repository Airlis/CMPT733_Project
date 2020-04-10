from flask import Flask, render_template, url_for, redirect, jsonify, request
import json
from config import *
import torch
# from bert_pytorch import *
# from transformers import BertForSequenceClassification


import os, sys
import tensorflow as tf
import numpy as np
import data_helpers
import pandas as pd
from rcnn import RCNN
from tensorflow.contrib import learn
####################### Import ########################




# function to predict politeness of post title&content
def sentence_prediction(sentence, model):
    tokenizer = TOKENIZER
    max_len = MAX_LEN
    #review = str(sentence)
    #print(review)
    #review = " ".join(review.split())
    review = [str(sentence)]
    print(len(review))



    test_examples = [InputExample(guid=i, text_a=x, labels=[]) for i, x in enumerate(review)]
    test_features = convert_examples_to_features(test_examples, 256, tokenizer)


    ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    token_type_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    '''
    print(inputs)
    inputs = tokenizer.encode(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len
    )
    print(inputs)

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
    '''

    input_ids = ids.to(DEVICE)
    segment_ids = token_type_ids.to(DEVICE)
    input_mask = mask.to(DEVICE)


    # Compute the logits
    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask)
        logits = logits.sigmoid()

    # Save the logits

    all_logits = logits.detach().cpu().numpy()

    print(all_logits)
    '''
    outputs = model(ids,token_type_ids,mask)
    logits = outputs[0]
    pred = logits.detach().cpu().squeeze().numpy()
    print('----------------')
    print(type(logits))
    print(pred)

    # Compute the logits
    # with torch.no_grad():
    #     logits = model(ids, token_type_ids, mask)
    #     logits = logits.sigmoid()
    #
    # # Save the logits
    # if all_logits is None:
    #     all_logits = logits.detach().cpu().numpy()
    # else:
    #     all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
            # Compute the logits

    with torch.no_grad():
        logits = model(ids,token_type_ids,mask)
        logits = logits[0].sigmoid()

    # Save the logits
    if all_logits is None:
        all_logits = logits.detach().cpu().numpy()
    else:
        all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
    print(all_logits)

    outputs = torch.sigmoid(torch.tensor(pred)).numpy()
    print(type(outputs))
    print(outputs)
    '''
    return all_logits


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
            body +=title
            sentence = body

            positive_prediction = sentence_prediction(sentence, model=MODEL)[0]

            output = {LABEL_LIST[i]: str(positive_prediction[i]) for i in range(6)}
            output["polite"] = 1 - sum(positive_prediction)

            return render_template('layout/default.html',
                                content=render_template('pages/post.html',title=title,body=body, json_data=output) )

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

# @app.route("/predict", methods=['GET','POST'])
# def predict():
#     #sentence = request.args.get("sentence")
#     if request.method == 'POST':
#         title = request.form['title']
#         message = request.form['message']
#         message +=title
#         #print(type(message))
#         sentence = message

#         positive_prediction = sentence_prediction(sentence, model=MODEL)[0]

#         output = {LABEL_LIST[i]: str(positive_prediction[i]) for i in range(6)}
#         output["polite"] = 1 - sum(positive_prediction)

#         return render_template('layout/default.html',
#                                 content=render_template('pages/result.html', json_data=output) )



if __name__ == '__main__':

    # model_state_dict = torch.load(BERT_MODEL, map_location='cpu')
    #MODEL = BertForSequenceClassification.from_pretrained(BERT_CONFIG,state_dict = model_state_dict, num_labels=6)

    # MODEL = BertForMultiLabelSequenceClassification.from_pretrained(BERT_PATH,
                                                                             # num_labels=6,
                                                                             # state_dict=model_state_dict)

    # MODEL.to(DEVICE)
    # MODEL.eval()

    ####################################################


    # add para debug=True here to enable refresh update
    app.run(debug=True)
