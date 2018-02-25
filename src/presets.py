from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from src.util.estimators import MultiProba
from src.util.preprocessors import OnColumn

from kgutil.models.keras import KerasRNN

from hyperopt import hp


def hp_search_space(**space):
    def decorator(fn):
        fn.hp_search_space = space
        return fn
    return decorator


## Presets


def basic_lr():
    return make_pipeline(
        OnColumn('comment_text', CountVectorizer(max_features=1000, min_df=5)),
        MultiProba(LogisticRegression())
    )


@hp_search_space(
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
            text_emb_size=300, text_emb_file='input/glove.42B.300d.txt', text_emb_trainable=False, text_emb_dropout=0.2,
            rnn_layers=[32],
            mlp_dropout=0.3, mlp_layers=[64]
        )
    )


@hp_search_space(
    text_emb_dropout=hp.uniform('text_emb_dropout', 0.0, 0.7),
    rnn_layer_size=hp.quniform('rnn_layer_size', 16, 128, 16),
    mlp_layer_size=hp.quniform('mlp_layer_size', 16, 128, 16),
    mlp_dropout=hp.uniform('mlp_dropout', 0.0, 0.7),
)
def rnn_pretrained_2(text_emb_dropout=0.2, rnn_layer_size=32, mlp_layer_size=64, mlp_dropout=0.3):
    return KerasRNN(
        num_epochs=5, batch_size=200, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=300, text_emb_file='input/glove.42B.300d.txt', text_emb_trainable=False, text_emb_dropout=text_emb_dropout,
            rnn_layers=[int(rnn_layer_size)], rnn_bidi=True, rnn_pooling='avgmax',
            mlp_dropout=mlp_dropout, mlp_layers=[int(mlp_layer_size)]
        )
    )
