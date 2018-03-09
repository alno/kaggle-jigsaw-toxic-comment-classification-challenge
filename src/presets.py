from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import roc_auc_score

from scipy.special import expit

from src.util.estimators import MultiProba, SimpleAverage, WeightedAverage, OnExtendedData
from src.util.preprocessors import OnColumn, DropColumns
from src.meta import input_file
from src import augmentations

import src.models.keras as keras_models
import src.models.tensorflow as tf_models

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


def submodels(*submodels):
    def decorator(fn):
        fn.submodels = submodels
        return fn
    return decorator


## Test models

@features('clean1')
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


@features('clean1', 'num2')
def test_rnn_feats():
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


@features('clean1', 'sentiment1')
def test_rnn_ext():
    return OnExtendedData(KerasRNN(
        num_epochs=1, batch_size=3000, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=8,
            rnn_layers=[8],
            mlp_layers=[]
        )
    ))


@features('clean1')
def test_tf():
    return tf_models.TfModel(
        num_epochs=1, batch_size=3000,
        model_opts=dict(emb_size=8, rnn_size=8)
    )


@features('clean1')
def test_rnn_aug():
    return keras_models.AugmentedModel(
        num_epochs=3, batch_size=3000, external_metrics=dict(roc_auc=roc_auc_score),
        compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
        train_augmentations=[augmentations.RandomCrop(min_len=0.5, max_len=100)],
        predict_augmentations=[augmentations.RandomCrop(min_len=0.5, max_len=100)],
        predict_passes=2,
        model_opts=dict(
            out_activation='sigmoid',
            text_emb_size=8,
            rnn_layers=[8],
            mlp_layers=[]
        )
    )

## L1 models


def basic_lr():
    return make_pipeline(
        OnColumn('comment_text', CountVectorizer(max_features=1000, min_df=5)),
        MultiProba(LogisticRegression())
    )


def lr2():
    return make_pipeline(
        OnColumn('comment_text', make_union(
            TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english',
                ngram_range=(1, 1),
                max_features=10000),
            TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                stop_words='english',
                ngram_range=(2, 6),
                max_features=50000)
        )),
        MultiProba(LogisticRegression())
    )


@features('clean1', 'num1')
def lr3():
    return make_pipeline(
        make_union(
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english',
                ngram_range=(1, 1),
                max_features=10000)),
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                stop_words='english',
                ngram_range=(2, 6),
                max_features=50000)),
            DropColumns(['comment_text']),
        ),
        MultiProba(LogisticRegression())
    )


@features('clean2', 'num1')
def lr3_cl2():
    return make_pipeline(
        make_union(
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english',
                ngram_range=(1, 1),
                max_features=10000)),
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                stop_words='english',
                ngram_range=(2, 6),
                max_features=50000)),
            DropColumns(['comment_text']),
        ),
        MultiProba(LogisticRegression())
    )


@features('clean1', 'num1', 'num2', 'sentiment1')
def lr3_more_feats():
    return make_pipeline(
        make_union(
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english',
                ngram_range=(1, 1),
                max_features=10000)),
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                stop_words='english',
                ngram_range=(2, 6),
                max_features=50000)),
            DropColumns(['comment_text']),
        ),
        MultiProba(LogisticRegression())
    )


@features('clean1', 'num1')
def lr3_more_ngrams():
    return make_pipeline(
        make_union(
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english',
                ngram_range=(1, 1),
                max_features=10000)),
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                stop_words='english',
                ngram_range=(2, 6),
                max_features=60000)),
            DropColumns(['comment_text']),
        ),
        MultiProba(LogisticRegression())
    )


@features('clean2', 'num1', 'num2', 'sentiment1')
def lr4():
    return make_pipeline(
        make_union(
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                stop_words='english',
                ngram_range=(1, 1),
                max_features=10000)),
            OnColumn('comment_text', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                stop_words='english',
                ngram_range=(2, 6),
                max_features=60000)),
            DropColumns(['comment_text']),
        ),
        MultiProba(LogisticRegression())
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


def cudnn_lstm_2_ext():
    return OnExtendedData(max_len=70, decay=0.7, model=KerasRNN(
        num_epochs=30, batch_size=800, external_metrics=dict(roc_auc=roc_auc_score),
        early_stopping_opts=dict(patience=3),
        compile_opts=None,
        model_fn=keras_models.cudnn_lstm_1,
        model_opts=dict(
            lr=1e-3,
            rnn_layers=[64, 64], rnn_dropout=0.15,
            text_emb_size=200, text_emb_file=input_file('glove.twitter.27B.200d.txt'), text_emb_dropout=0.25
        )
    ))


def cudnn_lstm_3_ext():
    return OnExtendedData(max_len=70, decay=0.7, n_samples=80000, model=KerasRNN(
        num_epochs=30, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post',
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.cudnn_lstm_1,
        model_opts=dict(
            lr=1e-3,
            rnn_layers=[64, 64], rnn_dropout=0.15,
            mlp_layers=[96], mlp_dropout=0.3,
            text_emb_size=200, text_emb_file=input_file('glove.twitter.27B.200d.txt'), text_emb_dropout=0.25
        )
    ))


def bigru_gmp_1():
    return KerasRNN(
        num_epochs=50, batch_size=800, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='gmp',
            out_dropout=0.3,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.4
        )
    )


@features('clean2_corrected_fasttext')
def bigru_gmp_2():
    return KerasRNN(
        num_epochs=50, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='gmp',
            out_dropout=0.3,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5
        )
    )


@features('clean2_corrected_fasttext')
def bigru_sterby_2():
    return KerasRNN(
        num_epochs=50, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.4,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5
        )
    )


@features('clean2_corrected_fasttext', 'num1')
def bigru_sterby_2_num():
    return KerasRNN(
        num_epochs=50, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5
        )
    )


@features('clean2_corrected_fasttext', 'num1', 'sentiment1')
def bigru_sterby_2_num_sent():
    return KerasRNN(
        num_epochs=50, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5
        )
    )


@features('clean2_corrected_fasttext', 'num1', 'num2', 'sentiment1')
def bigru_sterby_2_num_sent_longer():
    return KerasRNN(
        num_epochs=50, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        num_text_words=50000, max_text_len=150,
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5
        )
    )


@param_search_space(
    out_dropout=hp.uniform('out_dropout', 0.2, 0.6),
    text_emb_dropout=hp.uniform('text_emb_dropout', 0.3, 0.7),
    lr=hp.loguniform('lr', -9.2, -4.6),
    rnn_size=hp.quniform('rnn_size', 32, 128, 16),
)
@features('clean2_corrected_fasttext', 'num1', 'num2', 'sentiment1')
def bigru_sterby_2_num_sent_longer_rand(out_dropout=0.35, text_emb_dropout=0.5, lr=1e-3, rnn_size=80):
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=500), dict(num_epochs=10, batch_size=1000), dict(num_epochs=40, batch_size=2000)],
        external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        num_text_words=50000, max_text_len=150,
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=lr,
            rnn_size=rnn_size, rnn_pooling='sterby',
            out_dropout=out_dropout,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=text_emb_dropout, text_emb_rand_std=0.3,
        )
    )


@features('clean2_corrected_fasttext')
def bigru_sterby_2_ext():
    return OnExtendedData(max_len=70, decay=0.8, n_samples=40000, model=KerasRNN(
        num_epochs=50, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.4,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5
        )
    ))


@features('clean2_corrected_fasttext', 'num1')
def bigru_sterby_3():
    return KerasRNN(
        num_epochs=50, batch_size=1000, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_2,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35, num_layer_size=16,
            text_emb_fix_size=300, text_emb_fix_file=input_file('crawl-300d-2M.vec'), text_emb_free_size=8, text_emb_dropout=0.5
        )
    )


@features('clean2_bpe50k', 'num1', 'num2', 'sentiment1')
def bigru_sterby_4_bpe50k():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=500), dict(num_epochs=10, batch_size=1000), dict(num_epochs=40, batch_size=2000)],
        external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        max_text_len=150,
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35,
            text_emb_size=300, text_emb_file=input_file('en.wiki.bpe.op50000.d300.w2v.txt'), text_emb_dropout=0.5, text_emb_rand_std=0.3,
        )
    )


@features('clean2_bpe25k', 'num1')
def bigru_sterby_4_bpe25k():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=500), dict(num_epochs=10, batch_size=1000), dict(num_epochs=40, batch_size=2000)],
        external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post', text_tokenizer_opts=dict(filters='', lower=False),
        max_text_len=150,
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35,
            text_emb_size=300, text_emb_file=input_file('en.wiki.bpe.op25000.d300.w2v.txt'), text_emb_dropout=0.5, text_emb_rand_std=0.3,
        )
    )


@features('clean2_bpe10k', 'num1', 'num2', 'sentiment1')
def bigru_sterby_4_bpe10k():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=500), dict(num_epochs=10, batch_size=1000), dict(num_epochs=40, batch_size=2000)],
        external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        max_text_len=150,
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35,
            text_emb_size=300, text_emb_file=input_file('en.wiki.bpe.op10000.d300.w2v.txt'), text_emb_dropout=0.5, text_emb_rand_std=0.3,
        )
    )


def bigru_cnn_1():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=2, batch_size=512)],
        external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=150,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_cnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=128, out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('glove.42B.300d.txt'), text_emb_dropout=0.4
        )
    )


@features('clean2')
def bigru_cnn_2():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=5, batch_size=256), dict(num_epochs=40, batch_size=512)],
        external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=150,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_cnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=128, rnn_dropout=0.2, out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.4
        )
    )


@features('clean2_no_punct')
def bigru_cnn_3():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=4, batch_size=512), dict(num_epochs=10, batch_size=1024)],
        predict_batch_size=1024, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=150,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_cnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=128, rnn_dropout=0.3, out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5, text_emb_rand_std=0.3
        )
    )


@features('clean2_no_punct')
def bigru_rcnn_1():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=4, batch_size=512), dict(num_epochs=10, batch_size=1024)],
        predict_batch_size=1024, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=150,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_rcnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=96, rnn_dropout=0.3, rnn_dense_activation='relu', out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5, text_emb_rand_std=0.3
        )
    )


@features('clean2_expand_no_punct', 'num1', 'ind1')
def bigru_rcnn_2():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=4, batch_size=512), dict(num_epochs=10, batch_size=1024)],
        predict_batch_size=1024, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=150,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_rcnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=96, rnn_dropout=0.3, rnn_dense_activation='relu', out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5, text_emb_rand_std=0.3
        )
    )


@features('clean2_expand_no_punct', 'num1', 'ind1')
def bigru_rcnn_3():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=4, batch_size=512), dict(num_epochs=20, batch_size=1024)],
        predict_batch_size=1024, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=150,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_rcnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=96, rnn_dropout=0.3,
            mlp_layers=[64], mlp_dropout=0.2, out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5, text_emb_rand_std=0.3
        )
    )


@features('clean2_expand_no_punct_lemmatize', 'num1', 'ind1')
def bigru_rcnn_4():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=4, batch_size=512), dict(num_epochs=20, batch_size=1024)],
        predict_batch_size=1024, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=150,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_rcnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=96, rnn_dropout=0.3,
            mlp_layers=[64], mlp_dropout=0.2, out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5, text_emb_rand_std=0.3
        )
    )


@features('clean2_expand_no_punct_lemmatize', 'num1', 'num2', 'ind1', 'sentiment1')
def bigru_cnn_4():
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=4, batch_size=512), dict(num_epochs=4, batch_size=1024), dict(num_epochs=10, batch_size=2048)],
        predict_batch_size=1024, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        num_text_words=100000, max_text_len=200,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_cnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=128, rnn_dropout=0.3, out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5, text_emb_rand_std=0.3
        )
    )


@param_search_space(
    out_dropout=hp.uniform('out_dropout', 0.2, 0.6),
    text_emb_dropout=hp.uniform('text_emb_dropout', 0.3, 0.7),
    lr=hp.loguniform('lr', -9.2, -4.6),
    rnn_size=hp.quniform('rnn_size', 32, 128, 16),
)
@features('clean2_expand_no_punct_lemmatize', 'num1', 'num2', 'ind1', 'sentiment1')
def bigru_sterby_5(out_dropout=0.35, text_emb_dropout=0.5, lr=1e-3, rnn_size=80):
    return KerasRNN(
        train_schedule=[dict(num_epochs=3, batch_size=500), dict(num_epochs=10, batch_size=1000), dict(num_epochs=40, batch_size=2000)],
        external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        num_text_words=50000, max_text_len=200,
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=lr,
            rnn_size=rnn_size, rnn_pooling='sterby',
            out_dropout=out_dropout,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=text_emb_dropout, text_emb_rand_std=0.3,
        )
    )


@features('clean2_no_punct', 'num1')
def bigru_cnn_5_aug():
    return keras_models.AugmentedModel(
        train_schedule=[dict(num_epochs=3, batch_size=128), dict(num_epochs=4, batch_size=256), dict(num_epochs=4, batch_size=512), dict(num_epochs=4, batch_size=1024), dict(num_epochs=10, batch_size=2048)],
        predict_batch_size=1024, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='pre', text_padding='pre',
        num_text_words=100000, max_text_len=100,
        early_stopping_opts=dict(patience=5),
        compile_opts=None,
        model_fn=keras_models.bigru_cnn_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=128, rnn_dropout=0.2, out_dropout=0.2,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.45, text_emb_rand_std=0.3
        ),
        train_augmentations=[augmentations.RandomCrop(min_len=0.9, max_len=100)],
        predict_augmentations=[augmentations.RandomCrop(min_len=0.9, max_len=100)],
        predict_passes=4,
    )



@features('clean2_corrected_fasttext', 'num1')
def bigru_sterby_2_num_aug():
    return keras_models.AugmentedModel(
        num_epochs=50, batch_size=1000, predict_batch_size=2048, external_metrics=dict(roc_auc=roc_auc_score),
        text_truncating='post', text_padding='post',
        early_stopping_opts=dict(patience=6),
        compile_opts=None,
        model_fn=keras_models.bigru_1,
        model_opts=dict(
            lr=1e-3,
            rnn_size=80, rnn_pooling='sterby',
            out_dropout=0.35,
            text_emb_size=300, text_emb_file=input_file('crawl-300d-2M.vec'), text_emb_dropout=0.5, text_emb_rand_std=0.3
        ),
        train_augmentations=[augmentations.RandomCrop(min_len=0.9, max_len=100)],
        predict_augmentations=[augmentations.RandomCrop(min_len=0.9, max_len=100)],
        predict_passes=8,
    )


# L2


@submodels('cudnn_lstm_2', 'rnn_pretrained_3')
def l2_lr():
    return make_pipeline(
        DropColumns(['comment_text']),
        FunctionTransformer(expit),
        MultiProba(LogisticRegression())
    )


@submodels('cudnn_lstm_2', 'rnn_pretrained_3')
def l2_avg():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels('cudnn_lstm_2', 'rnn_pretrained_4')
def l2_avg2():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels('cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4')
def l2_wavg1():
    return make_pipeline(
        DropColumns(['comment_text']),
        WeightedAverage([0.4, 0.2, 0.4]),
    )


@submodels('lr2', 'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4')
def l2_wavg2():
    return make_pipeline(
        DropColumns(['comment_text']),
        WeightedAverage([0.15, 0.35, 0.1, 0.4]),
    )


@submodels('lr2', 'lr3', 'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4')
def l2_wavg3():
    return make_pipeline(
        DropColumns(['comment_text']),
        WeightedAverage([0.05, 0.1, 0.35, 0.1, 0.4]),
    )


@submodels('lr2', 'lr3', 'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1')
def l2_wavg4():
    return make_pipeline(
        DropColumns(['comment_text']),
        WeightedAverage([0.05, 0.1, 0.3, 0.1, 0.4, 0.4], renorm=True),
    )


@submodels('lr2', 'lr3', 'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1')
def l2_avg3():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels('lr2', 'lr3', 'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1', 'bigru_sterby_2')
def l2_avg4():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels('lr2', 'lr3', 'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1', 'bigru_sterby_2', 'bigru_sterby_2_num')
def l2_avg5():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels(
    'lr2', 'lr3',
    'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1', 'bigru_sterby_2',
    'bigru_sterby_2_num', 'bigru_sterby_2_num_sent_longer_rand', 'bigru_sterby_4_bpe50k'
)
def l2_avg6():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels(
    'lr2', 'lr3', 'lr3_cl2',
    'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1', 'bigru_sterby_2',
    'bigru_sterby_2_num', 'bigru_sterby_2_num_sent_longer_rand', 'bigru_sterby_4_bpe50k'
)
def l2_avg7():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels(
    'lr2', 'lr3', 'lr3_cl2',
    'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1', 'bigru_sterby_2',
    'bigru_sterby_2_num', 'bigru_sterby_2_num_sent_longer_rand', 'bigru_sterby_4_bpe50k',
    'bigru_cnn_3'
)
def l2_avg8():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )


@submodels(
    'lr2', 'lr3', 'lr3_cl2',
    'cudnn_lstm_2', 'rnn_pretrained_3', 'rnn_pretrained_4', 'bigru_gmp_1', 'bigru_sterby_2',
    'bigru_sterby_2_num', 'bigru_sterby_2_num_sent_longer_rand', 'bigru_sterby_4_bpe50k',
    'bigru_cnn_3', 'bigru_cnn_4', 'bigru_sterby_5'
)
def l2_avg9():
    return make_pipeline(
        DropColumns(['comment_text']),
        SimpleAverage(),
    )
