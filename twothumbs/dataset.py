import pickle
from pathlib import Path

from sklearn.datasets import load_files


class ReviewDataset:
    """Loads the IMDB Dataset to memory, either from the raw text files or from a cache."""

    def __init__(self):
        train_pickle = Path("acl_train_pickle")
        test_pickle = Path("acl_test_pickle")

        self.train = None
        self.test = None

        # Load Train Data
        if train_pickle.is_file():
            with train_pickle.open("rb") as f:
                self.train = pickle.load(f)
        else:
            self.train = load_files("aclImdb/train/")

            with train_pickle.open("wb") as f:
                pickle.dump(self.train, f)

        # Load Test Data
        if test_pickle.is_file():
            with test_pickle.open("rb") as f:
                self.test = pickle.load(f)
        else:
            self.test = load_files("aclImdb/test/")

            with test_pickle.open("wb") as f:
                pickle.dump(self.test, f)

    @property
    def train_data(self) -> list[bytes]:
        # Lazy filtering
        return [doc.replace(b"<br />", b" ") for doc in self.train.data]

    @property
    def train_target(self):
        return self.train.target

    @property
    def test_data(self) -> list[bytes]:
        return [doc.replace(b"<br />", b" ") for doc in self.test.data]


    @property
    def test_target(self):
        return self.test.target
