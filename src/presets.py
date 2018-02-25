from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.util.estimators import MultiProba
from src.util.preprocessors import OnColumn

from kgutil.models.keras import KerasRNN

from sklearn.metrics import roc_auc_score


basic_lr = make_pipeline(
    OnColumn('comment_text', CountVectorizer(max_features=1000, min_df=5)),
    MultiProba(LogisticRegression())
)


basic_rnn = KerasRNN(
    num_epochs=2, batch_size=200,
    compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
    model_opts=dict(
        out_activation='sigmoid',
        text_emb_size=32,
        rnn_layers=[32],
        mlp_dropout=0.3, mlp_layers=[64]
    )
)


rnn_pretrained = KerasRNN(
    num_epochs=5, batch_size=200, external_metrics=dict(roc_auc=roc_auc_score),
    compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
    model_opts=dict(
        out_activation='sigmoid',
        text_emb_size=300, text_emb_file='input/glove.42B.300d.txt', text_emb_trainable=False, text_emb_dropout=0.2,
        rnn_layers=[32],
        mlp_dropout=0.3, mlp_layers=[64]
    )
)


rnn_pretrained_2 = KerasRNN(
    num_epochs=5, batch_size=200, external_metrics=dict(roc_auc=roc_auc_score),
    compile_opts=dict(loss='binary_crossentropy', optimizer='adam'),
    model_opts=dict(
        out_activation='sigmoid',
        text_emb_size=300, text_emb_file='input/glove.42B.300d.txt', text_emb_trainable=False, text_emb_dropout=0.2,
        rnn_layers=[32], rnn_bidi=True, rnn_pooling='avgmax',
        mlp_dropout=0.3, mlp_layers=[64]
    )
)
