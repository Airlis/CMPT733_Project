#coding:utf-8
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import math
import os
import time
import datetime
import data_helpers
import data_helper
import sklearn.model_selection
from text_cnn import TextCNN  # 导入textCNN类
from tensorflow.contrib import learn  #从contrib导入learn包

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#Parameters
# ==================================================
#tf.flag用于处理命令行参数的解析工作
#调用flags内部的DEFINE_string/float/integer来制定解析规则 （命名，默认值，解释）
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_input", r"/home/lj/cw/cnn/data/stackoverflow.com", "Data source")

# Model Hyperparameters
#调参
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
#调参
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 40, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#显示参数集合
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load  train data
print("Loading data...")
#
#

#x_text,y=data_helper.file_iter(FLAGS.data_input, FLAGS.batch_size)

#x_text, y_test= data_helpers.load_data_and_labels(FLAGS.data_input, train=True)

# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_text])
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(x_text)))
#vocab = data_helper.build_vocab(x_text)

#print(x)
#print(type(x))
# Randomly shuffle data
#np.random.seed(10)

#print(x_train.shape,y_train.shape)
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#print("Vocabulary Size: {:d}".format(len(vocab)))
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      device_count={"CPU": 2},  # limit to num_cpu_core CPU usage
      inter_op_parallelism_threads=1,
    )
   
    #session_conf.gpu_options.per_process_gpu_memory_fraction = 0.2
    #session_conf.gpu_options.allow_growth =True
    #session_conf.gpu_options.allocator_type="BFC"
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        
       length_t=data_helper.file_size(FLAGS.data_input)
       l=math.ceil(length_t/FLAGS.batch_size)
        
       timestamp = str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
       out_dir = os.path.abspath(os.path.join(r"/home/lj/cw/cnn/data/stackoverflow.com", "runs/2017-09-02"))
       vocab_dir = os.path.join(out_dir, "vocab")
       vocab,vocablength=data_helper.open_vocab(vocab_dir)
      
       train=data_helper.file_iter(FLAGS.data_input, FLAGS.batch_size)
       
       x=[]
       y=[]
       
       for i in range(l-1):     
          data=next(train)
          
          #print(train)
          for line in data:
             #xtrain=[]
             line = line.split('\t')
             #xline.append(line[0])
             yline=line[1]
             ytt = [int(i) for i in yline[:-1]]
             #xtrain.append(line[0])     
             y.append(ytt)
             xtt=data_helper.use_vocab_dict_line(vocab,line[0])
             #print(xtt)
             x.append(xtt)
             print(len(x))
             #print(y)
       x = np.array(x)

       y = np.array(y)
       #print(xtrain)
       #x,vocab_length=data_helper.use_vocab_dict(vocab_dir,xtrain)
       shuffle_indices = np.random.permutation(np.arange(len(y)))
          
       x_shuffled=x[shuffle_indices]
       y_shuffled = y[shuffle_indices]

       # Split train/test set
       # TODO: This is very crude, should use cross-validation
       dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
       x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
       y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]  
       #print(y_train.shape[1])        
          
       cnn = TextCNN(
            sequence_length=600,
            #
            num_classes=y.shape[1],
            #vocab_size=len(vocab_processor.vocabulary_),
            vocab_size=vocablength,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,

            l2_reg_lambda=FLAGS.l2_reg_lambda)
       # Define Training procedure
       global_step = tf.Variable(0, name="global_step", trainable=False)
       optimizer = tf.train.GradientDescentOptimizer(1e-3)
       grads_and_vars = optimizer.compute_gradients(cnn.loss)
       train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

       # Output directory for models and summaries
       timestamp = str(time.strftime('%Y-%m-%d',time.localtime(time.time())))
       out_dir = os.path.abspath(os.path.join(r"/home/lj/cw/cnn/data/stackoverflow.com", "runs",timestamp))
       print("Writing to {}\n".format(out_dir))

       # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
       checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
       checkpoint_prefix = os.path.join(checkpoint_dir, "model")
       if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
       saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        

         
         # Write vocabulary
         #vocab_processor.save(os.path.join(out_dir, "vocab"))
       
       # Initialize all variables
       sess.run(tf.global_variables_initializer())

       def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              #cnn.sequence_length:np.array(x_batch).shape[1],
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss = sess.run(
                [train_op, global_step, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

       def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              # cnn.sequence_length:x_batch.shape[1],
              cnn.dropout_keep_prob: 1.0
            }
            step, loss = sess.run(
                [global_step, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            #print (feed_dict)
          # Generate batches
       batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
          # print("hello",batches)
       # Training loop. For each batch...
       for batch in batches:
            #print("hello",batch.shape)
            x_batch, y_batch = zip(*batch)
            #x_batch=datahelper.use_vocab_dict(vocab, x_batch)
            print(np.array(x_batch).shape[1])
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=None)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))   

     

      
