"""
This example demonstrates how to use `RF` model ( any ML model in general) from
`speechemotionrecognition` package
"""
import sys
sys.path.insert(1, '../main')

import pickle
import utility
from common import extract_data
from mlmodel import RF
from utility import get_feature_vector_from_mfcc


def ml_example():
    to_flatten = True
    x_train, x_test, y_train, y_test, _ = extract_data(flatten=to_flatten)
    model = RF()
    print('Starting', model.name)
    model.train(x_train, y_train)
    filename = 'rf_finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    model.evaluate(x_test, y_test)
    filename = '../AudioData/test--anger.wav'
    print('prediction', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
          'Actual: Anger')


if __name__ == "__main__":
    ml_example()
