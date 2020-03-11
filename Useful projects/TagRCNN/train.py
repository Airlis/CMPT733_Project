import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import tensorflow as tf
import numpy as np
import data_helpers
import time
import datetime
import signal
from rcnn import RCNN
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1,
#                       "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
#                        "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
#                        "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100,
                        "Dimensionality of character embedding (default: 100)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5",
#                        "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer(
#     "num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer(
    "num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100,
                        "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5,
                        "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
def sigint_handler(signum, frame):
    print("Test:")
    dev_step(x_test, y_test, writer=dev_summary_writer, test=True)
signal.signal(signal.SIGINT, sigint_handler)
# Data Preparation
# ==================================================
# first = True
datasets = ['wordpress']
for dataset in datasets:
    path = 'transformed_data/' + dataset
    if not os.path.exists(path):
        os.makedirs(path)
        # Load data
        x_text, y = data_helpers.load_data(
            'community-data/' + dataset + '/50/posts_df.csv')
        y = np.array(y).astype(float)

        # Build vocabulary
        max_document_length = np.percentile(
            [len(x.split(" ")) for x in x_text], 50, interpolation='lower')
        vocab_processor = learn.preprocessing.VocabularyProcessor(
            max_document_length)
        # vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
        #     'runs/1501323827/vocab')
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Write vocabulary
        vocab_processor.save(path + "/vocab")

        # Split train/test set
        x_train_dev, x_test, y_train_dev, y_test = train_test_split(
            x, y, test_size=0.1, random_state=42)
        x_train, x_dev, y_train, y_dev = train_test_split(
            x_train_dev, y_train_dev, test_size=0.1, random_state=42)
        print(x_train.shape, y_train.shape)
        with open(path + '/x_train.pickle', 'wb') as output:
            pickle.dump(x_train, output)
        with open(path + '/y_train.pickle', 'wb') as output:
            pickle.dump(y_train, output)
        with open(path + '/x_dev.pickle', 'wb') as output:
            pickle.dump(x_dev, output)
        with open(path + '/y_dev.pickle', 'wb') as output:
            pickle.dump(y_dev, output)
        with open(path + '/x_test.pickle', 'wb') as output:
            pickle.dump(x_test, output)
        with open(path + '/y_test.pickle', 'wb') as output:
            pickle.dump(y_test, output)
    else:
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
            path + '/vocab')
        with open(path + '/x_train.pickle', 'rb') as output:
            x_train = pickle.load(output)
        with open(path + '/y_train.pickle', 'rb') as output:
            y_train = pickle.load(output)
        with open(path + '/x_dev.pickle', 'rb') as output:
            x_dev = pickle.load(output)
        with open(path + '/y_dev.pickle', 'rb') as output:
            y_dev = pickle.load(output)
        with open(path + '/x_test.pickle', 'rb') as output:
            x_test = pickle.load(output)
        with open(path + '/y_test.pickle', 'rb') as output:
            y_test = pickle.load(output)

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            intra_op_parallelism_threads=3,
            inter_op_parallelism_threads=3)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rcnn = RCNN(
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_units=100,
                context_size=50,
                max_sequence_length=x_train.shape[1],
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(rcnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            # grad_summaries = []
            # for g, v in grads_and_vars:
            #     if g is not None:
            #         grad_hist_summary = tf.summary.histogram(
            #             "{}/grad/hist".format(v.name), g)
            #         sparsity_summary = tf.summary.scalar(
            #             "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            #         grad_summaries.append(grad_hist_summary)
            #         grad_summaries.append(sparsity_summary)
            # grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(
                os.path.curdir, "runs", dataset, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rcnn.loss)
            recall_10_summary = tf.summary.scalar("recall_10", rcnn.recall_10)
            recall_5_summary = tf.summary.scalar("recall_5", rcnn.recall_5)
            precise_10_summary = tf.summary.scalar(
                "precise_10", rcnn.precise_10)
            precise_5_summary = tf.summary.scalar("precise_5", rcnn.precise_5)
            # Train Summaries
            train_summary_op = tf.summary.merge(
                [loss_summary, recall_10_summary, recall_5_summary, precise_10_summary, precise_5_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge(
                [loss_summary, recall_10_summary, recall_5_summary, precise_10_summary, precise_5_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(
                dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already
            # exists so we need to create it
            checkpoint_dir = os.path.abspath(
                os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # total_recall_5 = 0
            # total_recall_10 = 0

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                # global total_recall_5, total_recall_10
                sequence_length = [len(np.nonzero(sample)[0])
                                   for sample in x_batch]
                feed_dict = {
                    rcnn.X: x_batch,
                    rcnn.y: y_batch,
                    rcnn.sequence_length: sequence_length,
                    # rcnn.max_sequence_length: max_sequence_length,
                    rcnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss_1, loss_0, recall_5, recall_10 = sess.run(
                    [train_op, global_step, train_summary_op,
                        rcnn.loss_for_1, rcnn.loss_for_0, rcnn.recall_5, rcnn.recall_10],
                    feed_dict)
                # total_recall_5 += recall_5
                # avg_recall_5 = total_recall_5 / step
                # total_recall_10 += recall_10
                # avg_recall_10 = total_recall_10 / step
                # time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss_1 {:.3f}, loss_0 {:.3f}, recall_5 {:.3f}, recall_10 {:.3f}".format(
                #     time_str, step, loss_1, loss_0, recall_5, recall_10))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None, test=False):
                """
                Evaluates model on a dev set
                """
                sequence_length = [len(np.nonzero(sample)[0])
                                   for sample in x_batch]
                feed_dict = {
                    rcnn.X: x_batch,
                    rcnn.y: y_batch,
                    rcnn.sequence_length: sequence_length,
                    # rcnn.max_sequence_length: max_sequence_length,
                    rcnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss_1, loss_0, recall_5, recall_10 = sess.run(
                    [global_step, train_summary_op,
                     rcnn.loss_for_1, rcnn.loss_for_0, rcnn.recall_5, rcnn.recall_10],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss_1 {:.3f}, loss_0 {:.3f}, recall_5 {:.3f}, recall_10 {:.3f}".format(
                    time_str, step, loss_1, loss_0, recall_5, recall_10))
                if test:
                    writer.add_summary(summaries, step + 1)
                else:
                    writer.add_summary(summaries, step)

            # Generate batches
            # print('1')
            # x_train = [np.trim_zeros(row) for row in x_train]
            # print('2')
            # x_batches, y_batches = tf.train.batch([x_train, y_train], batch_size=100, dynamic_pad=True, name='batches')
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), 100, 100, dynamic=False)
            # Training loop. For each batch...
            # print('3')
            # for x_batch, y_batch in zip(x_batches, y_batches):
            epoch = 0
            for batch in batches:
                if batch is True:
                    epoch += 1
                    current_step = tf.train.global_step(sess, global_step)
                    print("Epoch:" + str(epoch))
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                    path = saver.save(sess, checkpoint_prefix,
                                      global_step=current_step)
                    # print("Saved model checkpoint to {}\n".format(path))
                else:
                    x_batch, y_batch = zip(*batch)
                    # print(len(x_batch), len(y_batch))
                    # print(x_batch, y_batch)
                    train_step(x_batch, y_batch)
                # if current_step % FLAGS.evaluate_every == 0:

                # if current_step % FLAGS.checkpoint_every == 0:
            print("Test:")
            dev_step(x_test, y_test, writer=dev_summary_writer, test=True)
