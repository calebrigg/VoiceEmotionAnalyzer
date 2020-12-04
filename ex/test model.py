import sys
sys.path.insert(1, '../main')

import pickle
import utility
from common import extract_data
from mlmodel import NN
from utility import get_feature_vector_from_mfcc

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
x_train, x_test, y_train, y_test, _ = extract_data(flatten=True)
loaded_model.evaluate(x_test, y_test)
filename = "../AudioData/testcaleb.wav"
print(loaded_model.predict_one(get_feature_vector_from_mfcc(filename, flatten=True)), "actual:Angry")