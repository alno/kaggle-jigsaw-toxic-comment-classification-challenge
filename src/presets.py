from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from src.util.estimators import MultiProba
from src.util.preprocessors import OnColumn
from src.meta import input_file

import src.models.keras as keras_models

from kgutil.models.keras import KerasRNN

from hyperopt import hp


def param_search_space(**space):
    def decorator(fn):
        fn.param_search_space = space
        return fn
    return decorator


def features(*features):
    def decorator(fn):
        fn.features = features
        return fn
    return decorator


## Presets


def basic_lr():
    return make_pipeline(
        OnColumn('comment_text', CountVectorizer(max_features=1000, min_df=5)),
        MultiProba(LogisticRegression())
    )


@features('clean1', 'num1')
def test_rnn():
    return KerasRNN(
        num_epochs=1, batch_size=3000, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=8,
            rnn_layers=[8],
            mlp_layers=[]
        )
    )


@param_search_space(
    text_emb_size=hp.quniform('text_emb_size', 8, 32, 4),
    rnn_layer_size=hp.quniform('rnn_layer_size', 4, 16, 4),
    mlp_layer_size=hp.quniform('mlp_layer_size', 4, 16, 4),
    mlp_dropout=hp.uniform('mlp_dropout', 0.0, 0.7),
)
def basic_rnn(text_emb_size=32, rnn_layer_size=32, mlp_layer_size=64, mlp_dropout=0.3):
    return KerasRNN(
        num_epochs=10, batch_size=2000, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=int(text_emb_size),
            rnn_layers=[int(rnn_layer_size)],
            mlp_dropout=mlp_dropout, mlp_layers=[int(mlp_layer_size)]
        )
    )


def rnn_pretrained():
    return KerasRNN(
        num_epochs=5, batch_size=200, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=300, text_emb_file=input_file('glove.42B.300d.txt'), text_emb_trainable=False, text_emb_dropout=0.2,
            rnn_layers=[32],
            mlp_dropout=0.3, mlp_layers=[64]
        )
    )


@param_search_space(
    text_emb_dropout=hp.uniform('text_emb_dropout', 0.1, 0.6),
    rnn_layer_size=hp.quniform('rnn_layer_size', 16, 64, 16),
    rnn_bidi=hp.choice('rnn_bidi', [True, False]),
    rnn_pooling=hp.choice('rnn_pooling', [None, 'avg', 'max', 'avgmax']),
    mlp_layer_size=hp.quniform('mlp_layer_size', 32, 128, 32),
    mlp_layer_num=hp.quniform('mlp_layer_num', 1, 2, 1),
    mlp_dropout=hp.uniform('mlp_dropout', 0.1, 0.6),
)
def rnn_pretrained_2(text_emb_dropout=0.2, rnn_layer_size=32, rnn_bidi=True, rnn_pooling='avgmax', mlp_layer_size=64, mlp_layer_num=1, mlp_dropout=0.3):
    return KerasRNN(
        num_epochs=10, batch_size=200, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=300, text_emb_file=input_file('glove.42B.300d.txt'), text_emb_trainable=False, text_emb_dropout=text_emb_dropout,
            rnn_layers=[int(rnn_layer_size)], rnn_bidi=rnn_bidi, rnn_pooling=rnn_pooling,
            mlp_dropout=mlp_dropout, mlp_layers=[int(mlp_layer_size)] * int(mlp_layer_num)
        )
    )


def rnn_pretrained_3(text_emb_dropout=0.29, rnn_layer_size=32, rnn_bidi=True, rnn_pooling='avgmax', mlp_layer_size=96, mlp_layer_num=1, mlp_dropout=0.16):
    return KerasRNN(
        num_epochs=10, batch_size=500, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=300, text_emb_file=input_file('glove.42B.300d.txt'), text_emb_trainable=False, text_emb_dropout=text_emb_dropout,
            rnn_layers=[int(rnn_layer_size)], rnn_bidi=rnn_bidi, rnn_pooling=rnn_pooling,
            mlp_dropout=mlp_dropout, mlp_layers=[int(mlp_layer_size)] * int(mlp_layer_num)
        )
    )


def cudnn_lstm_1():
    return KerasRNN(
        num_epochs=20, batch_size=500, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=None,
        model_fn=keras_models.cudnn_lstm_1,
        model_opts=dict(
            text_emb_size=25, text_emb_file=input_file('glove.twitter.27B.25d.txt'), text_emb_trainable=False
        )
    )


def cudnn_lstm_2():
    return KerasRNN(
        num_epochs=30, batch_size=800, external_metrics=dict(roc_auc=roc_auc_score),
        early_stopping_opts=dict(patience=3),
        compile_opts=None,
        model_fn=keras_models.cudnn_lstm_1,
        model_opts=dict(
            lr=1e-3,
            rnn_layers=[64, 64], rnn_dropout=0.15,
            text_emb_size=200, text_emb_file=input_file('glove.twitter.27B.200d.txt'), text_emb_dropout=0.25
        )
    )


@features('clean1', 'num1')
def rnn_pretrained_4(text_emb_dropout=0.3, rnn_layer_size=32, rnn_layer_num=2, rnn_bidi=True, rnn_pooling='avgmax', mlp_layer_size=96, mlp_layer_num=1, mlp_dropout=0.15):
    return KerasRNN(
        num_epochs=20, batch_size=500, external_metrics=dict(roc_auc=roc_auc_score),
        early_stopping_opts=dict(patience=3),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=200, text_emb_file=input_file('glove.twitter.27B.200d.txt'), text_emb_trainable=False, text_emb_dropout=text_emb_dropout,
            rnn_layers=[int(rnn_layer_size)] * int(rnn_layer_num), rnn_bidi=rnn_bidi, rnn_pooling=rnn_pooling, rnn_cell='gru', rnn_dropout=0.1, rnn_cudnn=True,
            mlp_dropout=mlp_dropout, mlp_layers=[int(mlp_layer_size)] * int(mlp_layer_num)
        )
    )


@features('clean1', 'num1')
def rnn_pretrained_5(text_emb_dropout=0.3, rnn_layer_size=48, rnn_layer_num=2, rnn_bidi=True, rnn_pooling='avgmax', rnn_dropout=0.15, mlp_layer_size=96, mlp_layer_num=2, mlp_dropout=0.2):
    return KerasRNN(
        num_epochs=30, batch_size=500, external_metrics=dict(roc_auc=roc_auc_score),
        max_text_len=200,
        early_stopping_opts=dict(patience=3),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=200, text_emb_file=input_file('glove.twitter.27B.200d.txt'), text_emb_trainable=False, text_emb_dropout=text_emb_dropout,
            rnn_layers=[int(rnn_layer_size)] * int(rnn_layer_num), rnn_bidi=rnn_bidi, rnn_pooling=rnn_pooling, rnn_cell='gru', rnn_dropout=rnn_dropout, rnn_cudnn=True,
            mlp_dropout=mlp_dropout, mlp_layers=[int(mlp_layer_size)] * int(mlp_layer_num)
        )
    )
