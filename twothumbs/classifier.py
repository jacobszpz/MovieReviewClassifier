from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ReviewClassifier:
    def __init__(self, X_train: list, y_train):
        self.pipe = make_pipeline(CountVectorizer(min_df=5), LogisticRegression(max_iter=1000))
        self.pipe.fit(X_train, y_train)

    def test(self, X_test, y_test) -> None:
        return self.pipe.score(X_test, y_test)
        #return cross_val_score(LogisticRegression(max_iter=700), self.bag, train_target, cv=5)
