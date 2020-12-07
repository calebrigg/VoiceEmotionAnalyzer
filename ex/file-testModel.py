import sys
sys.path.insert(1, '../main')

import pickle
import utility
import os
from os import path
from common import extract_data
from mlmodel import NN
from utility import get_feature_vector_from_mfcc

while (True):
    prompt = input("Enter the name of the file you wish to analyze (The audio must be a 16 bit, mono .wav file with a bitrate of 48kHz) or q to quit: ")

    if (prompt.endswith(".wav") and path.exists("../AudioData/{}".format(prompt))):
        filename = 'nn_finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        filename = "../AudioData/{}".format(prompt)
        print(loaded_model.predict_one(get_feature_vector_from_mfcc(filename, flatten=True)))

    elif (prompt=="q"):
        sys.exit()
    
    elif (not path.exists("../AudioData/{}".format(prompt))):
        print("Error, this file does not exist or is not in the AudioData directory.")

    else:
        print("Error, Invalid file-type. Please try another file: ")