import numpy as np

from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Embedding, Dense, Dropout, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, concatenate
from keras.optimizers import Adam


def load_emb_matrix(filename, word_index, voc_size, emb_size, rand_std=None):
    if rand_std is None:
        emb_matrix = np.zeros((voc_size, emb_size), dtype=np.float32)
    else:
        emb_matrix = np.random.randn(voc_size, emb_size) * rand_std

    print("Loading embeddings...")
    for line in open(filename, 'r'):
        values = line.split()

        idx = word_index.get(values[0])
        if idx is not None:
            emb_matrix[idx] = np.asarray(values[1:], dtype=np.float32)

    return emb_matrix


def cudnn_lstm_1(
    data, target_shape,
    lr=0.0003,
    rnn_dropout=None, rnn_layers=[50],
    mlp_layers=[70], mlp_dropout=0.3,
    text_emb_dropout=0.2, text_emb_size=32, text_emb_file=None
):
    if text_emb_file is not None:
        emb_weights = [load_emb_matrix(text_emb_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_size)]
    else:
        emb_weights = None

    model = Sequential()
    model.add(InputLayer(name='comment_text', input_shape=[data.max_text_len]))
    model.add(Embedding(data.text_voc_size, text_emb_size, weights=emb_weights, trainable=False))
    model.add(Dropout(text_emb_dropout))

    for layer_size in rnn_layers:
        model.add(Bidirectional(CuDNNLSTM(layer_size, return_sequences=True)))
        if rnn_dropout is not None:
            model.add(SpatialDropout1D(rnn_dropout))

    model.add(GlobalMaxPool1D())
    for layer_size in mlp_layers:
        model.add(Dense(layer_size, activation="relu"))
        model.add(Dropout(mlp_dropout))
    model.add(Dense(6, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000015))
    return model


def bigru_1(
    data, target_shape,
    lr=0.0003,
    rnn_size=64, rnn_pooling=None,
    out_dropout=0.1,
    text_emb_dropout=0.2, text_emb_size=32, text_emb_file=None, text_emb_trainable=False
):
    if text_emb_file is not None:
        emb_weights = [load_emb_matrix(text_emb_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_size)]
    else:
        emb_weights = None

    inp = Input(shape=[data.max_text_len], name='comment_text')

    emb = Embedding(data.text_voc_size, text_emb_size, weights=emb_weights, trainable=text_emb_trainable)(inp)
    emb = SpatialDropout1D(text_emb_dropout)(emb)

    rnn_seq, rnn_fwd_out, rnn_rev_out = Bidirectional(CuDNNGRU(rnn_size, return_sequences=True, return_state=True))(emb)

    if rnn_pooling is None:
        out = concatenate([rnn_fwd_out, rnn_rev_out])
    elif rnn_pooling == 'gmp':
        out = GlobalMaxPool1D()(rnn_seq)
    elif rnn_pooling == 'sterby':
        out = concatenate([rnn_fwd_out, rnn_rev_out, GlobalMaxPool1D()(rnn_seq), GlobalAveragePooling1D()(rnn_seq)])
    else:
        raise RuntimeError("Unknown pooling: %r" % rnn_pooling)

    out = Dropout(out_dropout)(out)
    out = Dense(6, activation='sigmoid')(out)

    # Model
    model = Model(inp, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000015))
    return model
