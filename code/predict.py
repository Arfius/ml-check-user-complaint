# This class does the prediction of a product giving a text

import tensorflow as tf
from tensorflow import keras
from data_cleaning import DataCleaning
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json
import sys  
sys.path.insert(0, '../misc')
from etl import ETL

# Setting parameters. !!!These parameters could be moved in a config file.
MAX_WORDS = 120000
MAX_SEQ_LENGTH = 150
DATA_PATH = "../data"
CONFIG_PATH = "../misc/db_config.yaml"
SCHEMA_PATH = "../misc/schemas.yaml"


class Predict:
    # Load the model and the tokenizer
    # Clean the test
    # Get the new_id_product created during the training phases
    def __init__(self):
        self.model = tf.keras.models.load_model("../artifacts/model_products.h5")
        self.tokenizer = self.load_tokenizer()

    # Get the product_name and sub_product_name from the new_id_product
    # Return value need to be formatted
    def get_product_name(self,text):
        array_text = self.clean_text(text)
        new_id = self.get_new_product_id(array_text)
        return ETL(DATA_PATH, CONFIG_PATH, SCHEMA_PATH).select_product_name(new_id)

    # Text Preprocessing 
    def clean_text(self,text):
        dc = DataCleaning()
        return [dc.normalize(text)]

    # new_product_id Prediction 
    def get_new_product_id(self, array_text):
        _tokenized = self.tokenizer.texts_to_sequences(array_text)
        _testing = pad_sequences(_tokenized, maxlen=MAX_SEQ_LENGTH)
        _y_testing = self.model.predict(_testing, verbose = 1)
        return np.argmax(_y_testing)
       
    # Load the tokenizer from json file
    def load_tokenizer(self):
        with open('../artifacts/tokenizer.json') as f:
            data = json.load(f)
            return tokenizer_from_json(data)
