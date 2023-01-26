#!/usr/bin/env python3

from typing import Sequence

import numpy as np

from .classifier import ReviewClassifier
from .dataset import ReviewDataset


def main(argv: Sequence[str] | None = None) -> int:
    print("Loading IMDB Dataset...")
    dataset = ReviewDataset()

    print("Pre-Processing...")
    classifier = ReviewClassifier(dataset.train_data, dataset.train_target)

    print(classifier.test(dataset.test_data, dataset.test_target))

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
