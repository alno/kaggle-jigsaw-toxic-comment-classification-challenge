from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.util.estimators import MultiProba


basic_lr = make_pipeline(
    CountVectorizer(max_features=1000, min_df=5),
    MultiProba(LogisticRegression())
)
