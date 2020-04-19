import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import data_helpers
import pandas as pd
from rcnn import RCNN
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100,
                        "Dimensionality of character embedding (default: 100)")

# # Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()


def main(dataset, title, body):
    # Data Preparation
    # ==================================================
    
    path = os.path.join('transformed_data', dataset)

    # body = sys.argv[1]
    text = data_helpers.preprocess(title,body)
    x_text = [data_helpers.clean_str(text)]

    # Restore vocab file
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(
        path, 'vocab'))

    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # print(x)
    # print(x.shape)

    tags_df = pd.read_csv(os.path.join('community-data', dataset,'50','tags_df.csv'), encoding='utf8', index_col=0)
    tag_list = tags_df['TagName'].tolist()

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            intra_op_parallelism_threads=3,
            inter_op_parallelism_threads=3)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            rcnn = RCNN(
                num_classes=len(tag_list),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
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

            # Checkpoint directory. 
            checkpoint_dir = os.path.abspath(
                os.path.join(os.path.curdir, "runs", dataset))
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

            # Loading checkpoint
            save_path = os.path.join(checkpoint_dir, "model")
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
            
            # print(prediction)
            idx = prediction.argsort()[-5:][::-1]
            # print(idx)
            tags = [tag_list[i] for i in idx]

    print("\n||| Text |||\n")
    print(x_text[0])
    print("\n||| Predicted tags |||\n")
    print(tags)


if __name__ == '__main__':
    
    # dataset = 'bicycles'
    # title = 'Can I use a Presta tube in a Schrader rim?'
    # body = '''&lt;p&gt;I keep losing pressure in my tires, and among other things, I'm looking at the valves in the tubes. It's an old mountain bike with 26 inch tires. Can I use &lt;a href=&quot;https://en.wikipedia.org/wiki/Presta_valve&quot;&gt;Presta tubes&lt;/a&gt; on rims drilled for &lt;a href=&quot;https://en.wikipedia.org/wiki/Schrader_valve&quot;&gt;Schrader valves&lt;/a&gt;? I know the valve will be smaller than the hole and could cause issues there, but has anyone had success (or failures) with this?&lt;/p&gt;&#xA;'''

    # dataset = 'movies'
    # title = 'What weighs toward the decision to ignore the gravity change?'
    # body = """&lt;p&gt;Melancholia is a realistic &quot;what-if&quot; science fiction, set in roughly the time we live in. The whole plot is centered around the concept of a planet approaching Earth. Being what it is, why would they choose to ignore the effects of gravity as the planet approaches?&lt;/p&gt;&#xA;&#xA;&lt;p&gt;The moon is a relatively small satellite and is quite far from the Earth, but still it makes a noticeable difference to the tide. Now, think of a big planet approaching...&lt;/p&gt;&#xA;&#xA;&lt;p&gt;If you stop for a second and really consider the effect it would have on Earth's gravity. It would probably have freed up a whole lot of Earth's mass on the side that it was approaching, such that the surface of our planet and much more beneath would have been ejected from the atmosphere and drawn to the approaching planet. Or, in the very least, gravity would have greatly decreased.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;So, how come they didn't even explore a grand finale where the characters are ejected?&lt;/p&gt;&#xA;&#xA;&lt;p&gt;I found it terribly unrealistic. Was it a choice of saving on special effects? Didn't the people involved in production even consider the physics of it? Are there interviews done with them surrounding this issue? Or, is it a non-issue as far as entertainment is concerned (i.e. it's not relevant in this movie/genre)?&lt;/p&gt;&#xA;"""
    
    dataset = 'gardening'
    title = "How can I revive a tree that's been stripped of bark?"
    body = """&lt;p&gt;I've got what I think is a cedar tree in my backyard that is missing most of its bark on one side from the ground to about 4' up. (Thanks to my puppy who is a shredding machine.) The branches next to the bare area of the trunk are slowly turning brown and dying. Is there any way to revive the tree and minimize the damage? Should I prune off branches that appear to be mostly dead?&lt;/p&gt;&#xA;&#xA;&lt;p&gt;Also, on the topic of preventative action: Are there products that successfully keep animals (dogs in my case) away from trees and shrubs?&lt;/p&gt;&#xA;"""
    main(dataset, title, body)
        
