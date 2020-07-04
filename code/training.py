# This class trains the ML model 
# Export model (h5) and the tokenized into artifacts dicectory 
import sys  
sys.path.insert(0, '../misc')
from etl import ETL

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import io 
import json 

from embedding_glove import EmbeddingGlove
from data_cleaning import DataCleaning
from net_data_structure import NNetDS

#tf import
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Setting parameters. !!!These parameters could be moved in a config file.
DATA_PATH = "../data"
CONFIG_PATH = "../misc/db_config.yaml"
SCHEMA_PATH = "../misc/schemas.yaml"

MAX_WORDS = 120000
MAX_SEQ_LENGTH = 150
GLOV_EMBEDDING_DIM = 50

class Training:
    # Load dataset from Database.(I've manipulated the ETL class to load the dataset).
    
    # According the pruning method (pruning_method), the dataset is cleaned of some classes in order to have a good number of comment of each class.
    # pruning_method accepts mean|median|quartile|None
    # With None all the dataset is considered
    
    # Clean the text and build the model 

    def __init__(self, pruning_method=None):
        self.c_users = pd.DataFrame(data= ETL(DATA_PATH, CONFIG_PATH, SCHEMA_PATH).select_complaints_users_from_db(), columns=['COMPLAINT_ID','COMPLAINT_TEXT','PRODUCT_ID'] )
        p_id = self.pruning_product_list(pruning_method=pruning_method)
        self.dataset = self.pruning_data_set(p_id)
        self.update_mapping_new_product_id()
        texts = self.clean_text()
        self.nnds = NNetDS()
        self.prepare_tf_data_set(texts)
        self.nnds.embedding_matrix = EmbeddingGlove( MAX_WORDS = MAX_WORDS, MAX_SEQ_LENGTH = MAX_SEQ_LENGTH, GLOV_EMBEDDING_DIM = GLOV_EMBEDDING_DIM , word_index=self.nnds.word_index).get_matrix()
        self.build_tf_model()

    # Build the NN and fit the model.
    # Model is save into the artifacts folder
    # Network parameter are store here. !!!These parameters could be moved in a config file. 
    def build_tf_model(self):
        NN_OUTPUT = self.nnds.labels.shape[1]
        NN_DROPOUT = 0.2
        NN_INPUT = NN_OUTPUT * 2
        NAME_MODEL = "model_products.h5"
        EPOCHS = 2
        BATCH_SIZE = 32

        sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(self.nnds.word_index) + 1,
                           GLOV_EMBEDDING_DIM,
                           weights = [self.nnds.embedding_matrix],
                           input_length = MAX_SEQ_LENGTH,
                           trainable=False,
                           name = 'embeddings')
        embedded_sequences = embedding_layer(sequence_input)

        x = LSTM(NN_INPUT,return_sequences=True,name='lstm_complaint_layer')(embedded_sequences)
        x = GlobalMaxPool1D()(x)
        x = Dropout(NN_DROPOUT)(x)
        x = Dense(NN_INPUT, activation="relu")(x)
        x = Dropout(NN_DROPOUT)(x)
        preds = Dense(NN_OUTPUT, activation="sigmoid")(x)
        model = Model(sequence_input, preds)
        model.compile(loss = 'binary_crossentropy',
                    optimizer='adam',
                    metrics = ['accuracy'])
        model.fit(self.nnds.x_train, self.nnds.y_train, epochs = EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.nnds.x_val, self.nnds.y_val))
        model.save("../artifacts/"+NAME_MODEL)
        logging.info("Model H5 saved with name "+ NAME_MODEL)

    # Perare the data ready for tensorflow environemt
    # Create a tokenized and save it into the artifact folder as json file
    # Sreate the sequences from the text and do the padding
    # Split the dataset in trainingset and testset (75%,25%)
    def prepare_tf_data_set(self,texts):
        logging.info("Preprare data for Tensorflow")
        tokenizer = Tokenizer(num_words=MAX_WORDS)
        tokenizer.fit_on_texts(texts)
        self.nnds.sequences = tokenizer.texts_to_sequences(texts)
        type(self.nnds.sequences)
        logging.info("Create sequences")
        self.nnds.word_index = tokenizer.word_index
        x_seq = pad_sequences(self.nnds.sequences, padding = 'post', maxlen = MAX_SEQ_LENGTH)
        logging.info("Padding sequences")
        self.nnds.labels = to_categorical(self.dataset["NEW_PRODUCT_ID"].values)
        logging.info("Create hot vector")
        self.nnds.x_train, self.nnds.x_val, self.nnds.y_train, self.nnds.y_val = train_test_split(x_seq, self.nnds.labels, test_size=0.25, random_state=1)
        logging.info("Create Training-set and Test-set")
        self.save_tokenizer(tokenizer)

    # Text Preprocessing 
    def clean_text(self):
        logging.info("Text Cleaning")
        dc = DataCleaning()
        texts = [] 
        for line in tqdm(self.dataset["COMPLAINT_TEXT"], total=self.dataset.shape[0]): 
            texts.append(dc.normalize(line))
        return texts

    # Create a map between the product_id with a new_product_id
    # new_product_id facilitates the creation of the hot-vector
    # These new value are store into database. (I've manipulated the ETL class to load the dataset).
    def update_mapping_new_product_id(self):
        logging.info("Create new product id after pruning dataset")
        _r_map = self.dataset[['PRODUCT_ID', 'NEW_PRODUCT_ID']]
        mm = [_r_map['PRODUCT_ID'].unique(),_r_map['NEW_PRODUCT_ID'].unique()]
        mapping = pd.DataFrame(data=np.array(mm).T,columns=["product_id", "new_product_id"])
        logging.info("Save new product id into db")
        ETL(DATA_PATH, CONFIG_PATH, SCHEMA_PATH).insert_new_product_id_table(mapping)

    # Create a new_product_id and associate it to the existing product_id
    def pruning_data_set(self,p_id):
        logging.info("Removing entry from dataset")
        dataset = self.c_users[self.c_users["PRODUCT_ID"].isin(p_id)]
        dataset.loc[:,"NEW_PRODUCT_ID"] = None
                                        
        for p_id_i in range(len(p_id)):
            indexes = dataset["PRODUCT_ID"] == p_id[p_id_i]
            dataset["NEW_PRODUCT_ID"][indexes] = p_id_i

        return dataset
        
    # Manage the pruning method
    def pruning_product_list(self,pruning_method=None):
        unique_products = self.c_users.groupby('PRODUCT_ID')["COMPLAINT_ID"].nunique().sort_values(ascending=False).reset_index()

        if pruning_method == "mean":
            mean_products = unique_products["COMPLAINT_ID"].mean()
            p_id = unique_products[unique_products["COMPLAINT_ID"]>mean_products]["PRODUCT_ID"]

        elif pruning_method == "median":
            median_products = unique_products["COMPLAINT_ID"].median()
            p_id = unique_products[unique_products["COMPLAINT_ID"]>median_products]["PRODUCT_ID"]


        elif pruning_method == "quartile":
            quartile_products = unique_products["COMPLAINT_ID"].median()/2 # approx value!!!!
            p_id = unique_products[unique_products["COMPLAINT_ID"]>quartile_products]["PRODUCT_ID"]

        else:
            min_products = unique_products["COMPLAINT_ID"].min()
            p_id = unique_products[unique_products["COMPLAINT_ID"]>min_products]["PRODUCT_ID"]

        return p_id

    # Save the tokenized into artifacts folder
    def save_tokenizer(self,tokenizer):
        logging.info("Save Tokenized")
        tokenizer_json = tokenizer.to_json()
        with io.open('../artifacts/tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False)) 