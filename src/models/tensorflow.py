import tensorflow as tf
import numpy as np

from tqdm import tqdm
from itertools import zip_longest

from keras.preprocessing.text import Tokenizer


def model_fn(inputs, voc_size, emb_size=16, rnn_size=16):
    inp = inputs['comment_text']
    inp_len = inputs['comment_text_len']

    with tf.name_scope('embedding'):
        emb_w = tf.get_variable("emb", [voc_size, emb_size], dtype=tf.float32)

        inp = tf.nn.embedding_lookup(emb_w, inp)
        # TODO Add dropout

    with tf.name_scope('rnn'):
        if True:
            rnn = tf.contrib.rnn.GRUCell(rnn_size)
            _, inp = tf.nn.dynamic_rnn(rnn, inp, inp_len, dtype=tf.float32)
        else:
            rnn = tf.contrib.cudnn_rnn.CudnnGRU(1, rnn_size, direction='bidirectional')
            inp, _ = rnn(inp)

    # Max pooling
    #inp = tf.reduce_max(inp, axis=1)

    with tf.name_scope('out'):
        return tf.layers.dense(inp, 6)


class TfModel:

    def __init__(self, num_epochs, batch_size, model_opts={}):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.predict_batch_size = batch_size
        self.model_opts = model_opts

        self.graph = None
        self.session = None

    def fit(self, train_X, train_y):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(train_X['comment_text'])
        self.voc_size = len(self.tokenizer.word_index) + 1

        self._build_model()

        with self.session.as_default():
            self.session.run(self.global_initializer)

            for epoch in range(self.num_epochs):
                print("Epoch %d..." % epoch)

                self.session.run(self.local_initializer)

                indices = np.random.permutation(len(train_X))
                print("Indices: %d" % len(indices))

                pbar = tqdm(range(0, len(indices), self.batch_size))
                losses = []
                for ofs in pbar:
                    batch_indices = indices[ofs:ofs+self.batch_size]
                    batch = self._prepare_batch(train_X.iloc[batch_indices], train_y.iloc[batch_indices])

                    batch_loss, _ = self.session.run([self.model_batch_loss, self.optimize_op], feed_dict=batch)

                    losses.append(batch_loss)
                    pbar.set_description("Loss: %.3f" % np.mean(losses))

    def predict(self, X):
        with self.session.as_default():
            preds = []
            for ofs in range(0, len(X), self.predict_batch_size):
                batch = self._prepare_batch(X.iloc[ofs:ofs+self.predict_batch_size])
                batch_preds = self.session.run(self.model_preds, feed_dict=batch)

                preds.append(batch_preds)

            return np.concatenate(preds, axis=0)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def close(self):
        self.session.close()
        self.session = None

        self.graph.close()
        self.graph = None

    def _prepare_batch(self, X, y=None):
        seq = self.tokenizer.texts_to_sequences(X['comment_text'])

        #print(max(map(len, seq)))

        seq = [s[:100] for s in seq]

        seq_len = np.array(list(map(len, seq)), dtype=np.int32)

        # Pad batch sequences to common len
        seq = np.array(list(zip_longest(*seq, fillvalue=0)), dtype=np.int32).T

        batch = {}
        batch[self.inputs['comment_text']] = seq
        batch[self.inputs['comment_text_len']] = seq_len

        if y is not None:
            batch[self.labels] = y.values.astype(np.float32)

        return batch

    def _build_model(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.inputs = {
                'comment_text': tf.placeholder(dtype=tf.int32, shape=[None, None]),
                'comment_text_len': tf.placeholder(dtype=tf.int32, shape=[None]),
            }

            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 6])

            with tf.name_scope('forward'):
                self.model_logits = model_fn(self.inputs, self.voc_size, **self.model_opts)
                self.model_preds = tf.sigmoid(self.model_logits)

            with tf.name_scope('loss'):
                self.model_sample_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
                self.model_batch_loss = tf.reduce_mean(self.model_sample_losses)

            with tf.name_scope('backward'):
                self.optimizer = tf.train.AdamOptimizer()
                self.optimize_op = self.optimizer.minimize(self.model_batch_loss)

            print("Trainable vars: %r" % tf.trainable_variables())

            self.global_initializer = tf.global_variables_initializer()
            self.local_initializer = tf.local_variables_initializer()

        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
