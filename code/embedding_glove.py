# This class creates the embedding matrix using Glove.

import numpy as np

class EmbeddingGlove:
    # Extract the embedding value from Glove file. For default, each word is represented by 50 feautures. To increate this
    # number is needed to dowload other resources from https://nlp.stanford.edu/projects/glove/

    def __init__ (self,  MAX_WORDS = 120000, MAX_SEQ_LENGTH = 150, GLOV_EMBEDDING_DIM = 50 , word_index=None):
        GLOVE_DIR = "../data/glove.6B/glove.6B."+str(GLOV_EMBEDDING_DIM)+"d.txt"
        self.GLOV_EMBEDDING_DIM = GLOV_EMBEDDING_DIM
        self.embeddings_index = {}
        self.word_index=word_index

        f = open(GLOVE_DIR)
        for line in f:
            values = line.split()
            word = values[0]
            self.embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        f.close()

    # Return the embedding_matrix. Convert each word  in the dictionary (from dataset) in float vector.
    def get_matrix(self):
        embedding_matrix = np.random.random((len(self.word_index) + 1, self.GLOV_EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix



    # def create_embedding(self):
    #     MAX_WORDS = 120000 # max words for tokenizer
    #     MAX_SEQ_LENGTH = 150 # max length of each entry comment ( num of words)
    #     GLOV_EMBEDDING_DIM = 50 # Glove dimensions 
        