from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Embedding, Dense, Dropout, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, Conv1D, CuDNNLSTM, CuDNNGRU, TimeDistributed, Reshape, Permute, LocallyConnected1D, concatenate, ELU, Activation
from keras.optimizers import Adam
from keras import regularizers

from kgutil.models.keras.base import DefaultTrainSequence, DefaultTestSequence
from kgutil.models.keras.rnn import KerasRNN, load_emb_matrix

from copy import deepcopy
import inspect


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
    text_emb_dropout=0.2, text_emb_size=32, text_emb_file=None, text_emb_trainable=False, text_emb_rand_std=None
):
    if text_emb_file is not None:
        emb_weights = [load_emb_matrix(text_emb_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_size, rand_std=text_emb_rand_std)]
    else:
        emb_weights = None

    text_inp = Input(shape=[data.max_text_len], name='comment_text')

    inputs = [text_inp]

    emb = Embedding(data.text_voc_size, text_emb_size, weights=emb_weights, trainable=text_emb_trainable)(text_inp)
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

    if len(data.numeric_columns) > 0:
        num_inp = Input(shape=[len(data.numeric_columns)], name="numeric_columns__")
        inputs.append(num_inp)
        out = concatenate([out, num_inp])

    out = Dropout(out_dropout)(out)
    out = Dense(6, activation='sigmoid')(out)

    # Model
    model = Model(inputs, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000015))
    return model


def bigru_2(
    data, target_shape,
    lr=0.0003,
    rnn_size=64, rnn_pooling=None,
    out_dropout=0.1, num_layer_size=16,
    text_emb_dropout=0.2, text_emb_fix_size=32, text_emb_fix_file=None, text_emb_free_size=8,
):
    if text_emb_fix_file is not None:
        fix_emb_weights = [load_emb_matrix(text_emb_fix_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_fix_size)]
    else:
        fix_emb_weights = None

    text_inp = Input(shape=[data.max_text_len], name='comment_text')

    inputs = [text_inp]

    emb_fix = Embedding(data.text_voc_size, text_emb_fix_size, weights=fix_emb_weights, trainable=False)(text_inp)

    if text_emb_free_size > 0:
        emb_free = Embedding(data.text_voc_size, text_emb_free_size)(text_inp)
        emb = concatenate([emb_fix, emb_free])
    else:
        emb = emb_fix

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

    if len(data.numeric_columns) > 0:
        num_inp = Input(shape=[len(data.numeric_columns)], name="numeric_columns__")
        inputs.append(num_inp)
        num_out = Dense(num_layer_size, activation='relu')(num_inp)

        out = concatenate([out, num_out])

    out = Dropout(out_dropout)(out)
    out = Dense(6, activation='sigmoid')(out)

    # Model
    model = Model(inputs, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000015))
    return model


def activation(act, x):
    if act == 'relu':
        return Activation('relu')(x)
    elif act == 'elu':
        return ELU()(x)
    elif act is None:
        return x
    else:
        raise RuntimeError("Unknown activation %r" % act)


def bigru_cnn_1(
    data, target_shape,
    lr=1e-3,
    rnn_size=128, rnn_dropout=None, rnn_layers=1,
    conv_size=64, conv_activation=None,
    num_layers=[], num_activation='relu', num_dropout=None,
    out_dropout=None,
    text_emb_dropout=0.2, text_emb_size=32, text_emb_file=None, text_emb_trainable=False, text_emb_rand_std=None
):
    if text_emb_file is not None:
        emb_weights = [load_emb_matrix(text_emb_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_size, rand_std=text_emb_rand_std)]
    else:
        emb_weights = None

    text_inp = Input(shape=[data.max_text_len], name='comment_text')

    inputs = [text_inp]

    seq = Embedding(data.text_voc_size, text_emb_size, weights=emb_weights, trainable=text_emb_trainable)(text_inp)
    seq = SpatialDropout1D(text_emb_dropout)(seq)

    for _ in range(rnn_layers):
        seq = Bidirectional(CuDNNGRU(rnn_size, return_sequences=True))(seq)
        if rnn_dropout is not None:
            seq = SpatialDropout1D(rnn_dropout)(seq)
    seq = Conv1D(conv_size, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(seq)
    seq = activation(conv_activation, seq)
    out = concatenate([GlobalMaxPool1D()(seq), GlobalAveragePooling1D()(seq)])

    if len(data.numeric_columns) > 0:
        num_inp = Input(shape=[len(data.numeric_columns)], name="numeric_columns__")
        inputs.append(num_inp)

        num = num_inp
        for num_layer_size in num_layers:
            if num_dropout is not None:
                num = Dropout(num_dropout)(num)
            num = Dense(num_layer_size, activation=None)(num)
            num = activation(num_activation, num)

        out = concatenate([out, num])

    if out_dropout is not None:
        out = Dropout(out_dropout)(out)

    out = Dense(6, activation='sigmoid')(out)

    # Model
    model = Model(inputs, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))
    return model


def bigru_rcnn_1(
    data, target_shape,
    lr=1e-3,
    rnn_size=128, rnn_dropout=None, rnn_dense_size=64, rnn_dense_activation=None,
    mlp_layers=[], mlp_dropout=0.2, out_dropout=None,
    text_emb_dropout=0.2, text_emb_size=32, text_emb_file=None, text_emb_trainable=False, text_emb_rand_std=None
):
    if text_emb_file is not None:
        emb_weights = [load_emb_matrix(text_emb_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_size, rand_std=text_emb_rand_std)]
    else:
        emb_weights = None

    text_inp = Input(shape=[data.max_text_len], name='comment_text')

    inputs = [text_inp]

    emb = Embedding(data.text_voc_size, text_emb_size, weights=emb_weights, trainable=text_emb_trainable)(text_inp)
    emb = SpatialDropout1D(text_emb_dropout)(emb)

    seq = Bidirectional(CuDNNGRU(rnn_size, return_sequences=True))(emb)
    if rnn_dropout is not None:
        seq = SpatialDropout1D(rnn_dropout)(seq)

    seq = concatenate([emb, seq])
    seq = TimeDistributed(Dense(rnn_dense_size, activation=rnn_dense_activation))(seq)

    out = concatenate([GlobalMaxPool1D()(seq), GlobalAveragePooling1D()(seq)])

    if len(data.numeric_columns) > 0:
        num_inp = Input(shape=[len(data.numeric_columns)], name="numeric_columns__")
        inputs.append(num_inp)
        out = concatenate([out, num_inp])

    for layer_size in mlp_layers:
        if mlp_dropout is not None:
            out = Dropout(mlp_dropout)(out)
        out = Dense(layer_size)(out)

    if out_dropout is not None:
        out = Dropout(out_dropout)(out)

    out = Dense(6, activation='sigmoid')(out)

    # Model
    model = Model(inputs, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))
    return model


def bigru_rcnn_2(
    data, target_shape,
    lr=1e-3,
    rnn_size=128, rnn_dropout=None, rnn_dense_size=64,
    mlp_layers=[], mlp_dropout=0.2,
    out_dropout=None,
    text_emb_dropout=0.2, text_emb_size=32, text_emb_file=None, text_emb_trainable=False, text_emb_rand_std=None
):
    if text_emb_file is not None:
        emb_weights = [load_emb_matrix(text_emb_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_size, rand_std=text_emb_rand_std)]
    else:
        emb_weights = None

    text_inp = Input(shape=[data.max_text_len], name='comment_text')

    inputs = [text_inp]

    emb = Embedding(data.text_voc_size, text_emb_size, weights=emb_weights, trainable=text_emb_trainable)(text_inp)
    emb = SpatialDropout1D(text_emb_dropout)(emb)

    fwd_seq = Lambda(lambda x: tf.slice(tf.pad(x, [[0, 0], [1, 0], [0, 0]]), [0, 0, 0], tf.shape(x)))(emb)
    fwd_seq = CuDNNGRU(rnn_size, return_sequences=True)(fwd_seq)

    rev_seq = Lambda(lambda x: tf.slice(tf.pad(x, [[0, 0], [0, 1], [0, 0]]), [0, 1, 0], tf.shape(x)))(emb)
    rev_seq = Lambda(lambda x: tf.reverse(x, [1]))(rev_seq)
    rev_seq = CuDNNGRU(rnn_size, return_sequences=True)(rev_seq)
    rev_seq = Lambda(lambda x: tf.reverse(x, [1]))(rev_seq)

    if rnn_dropout is not None:
        fwd_seq = SpatialDropout1D(rnn_dropout)(fwd_seq)
        rev_seq = SpatialDropout1D(rnn_dropout)(rev_seq)

    seq = concatenate([fwd_seq, emb, rev_seq])
    seq = TimeDistributed(Dense(rnn_dense_size, activation='tanh'))(seq)

    out = concatenate([GlobalMaxPool1D()(seq), GlobalAveragePooling1D()(seq)])

    if len(data.numeric_columns) > 0:
        num_inp = Input(shape=[len(data.numeric_columns)], name="numeric_columns__")
        inputs.append(num_inp)
        out = concatenate([out, num_inp])

    for layer_size in mlp_layers:
        if mlp_dropout is not None:
            out = Dropout(mlp_dropout)(out)
        out = Dense(layer_size)(out)

    if out_dropout is not None:
        out = Dropout(out_dropout)(out)

    out = Dense(6, activation='sigmoid')(out)

    # Model
    model = Model(inputs, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))
    return model


def stack1(data, target_shape, l2=1e-3, shared=True):
    model = Sequential()
    model.add(InputLayer(name='numeric_columns__', input_shape=[len(data.numeric_columns)]))
    model.add(Reshape([-1, 6]))
    model.add(Permute((2, 1)))
    if shared:
        model.add(Conv1D(1, 1, kernel_regularizer=regularizers.l2(l2), kernel_initializer='zero', activation='sigmoid'))
    else:
        model.add(LocallyConnected1D(1, 1, kernel_regularizer=regularizers.l2(l2), kernel_initializer='zero', activation='sigmoid'))
    model.add(Reshape([6]))
    return model


def stack2(data, target_shape, l2=1e-3, hid_size=16, shared=True):
    model = Sequential()
    model.add(InputLayer(name='numeric_columns__', input_shape=[len(data.numeric_columns)]))
    model.add(Reshape([-1, 6]))
    model.add(Permute((2, 1)))
    if shared:
        model.add(Conv1D(hid_size, 1, kernel_regularizer=regularizers.l2(l2), activation='relu'))
        model.add(Conv1D(1, 1, kernel_regularizer=regularizers.l2(l2), kernel_initializer='zero', activation='sigmoid'))
    else:
        model.add(LocallyConnected1D(hid_size, 1, kernel_regularizer=regularizers.l2(l2), activation='relu'))
        model.add(LocallyConnected1D(1, 1, kernel_regularizer=regularizers.l2(l2), kernel_initializer='zero', activation='sigmoid'))
    model.add(Reshape([6]))
    return model


class AugTrainSequence(DefaultTrainSequence):

    def __init__(self, augmentations=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = [deepcopy(aug) for aug in augmentations]
        for aug in self.augmentations:
            if hasattr(aug, 'fit'):
                aug.fit(self.X, self.y)

    def _transform_batch(self, batch_x, batch_y):
        for aug in self.augmentations:
            if 'y' in inspect.getargspec(aug.transform).args:
                batch_x, batch_y = aug.transform(batch_x, batch_y)
            else:
                batch_x = aug.transform(batch_x)

        return super()._transform_batch(batch_x, batch_y)


class AugTestSequence(DefaultTestSequence):

    def __init__(self, augmentations=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = augmentations

    def _transform_batch(self, batch_x):
        for aug in self.augmentations:
            batch_x = aug.transform(batch_x)
        return super()._transform_batch(batch_x)


class AugmentedModel(KerasRNN):

    def __init__(
        self,
        _sentinel=None,
        train_augmentations=[], predict_augmentations=[],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.train_augmentations = train_augmentations
        self.predict_augmentations = predict_augmentations

    def _build_train_sequence(self, X, y, batch_size):
        return AugTrainSequence(
            data_transformer=self.data_transformer, target_transformer=self.target_transformer,
            X=X, y=y, batch_size=batch_size,
            augmentations=self.train_augmentations)

    def _build_test_sequence(self, X, batch_size):
        return AugTestSequence(
            data_transformer=self.data_transformer,
            X=X, batch_size=batch_size,
            augmentations=self.predict_augmentations)
