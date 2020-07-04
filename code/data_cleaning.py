# Resource from Internet (partially)
# This class implements functions for text cleaning

import nltk
import contractions
import inflect
import re
import unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

class DataCleaning:
    
    def replace_contractions(self,text):
        return contractions.fix(text)

    def remove_non_ascii(self,words):
        # Remove non-ASCII characters from list of tokenized words
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    def money_value(self,text):
        # Convert money value like {$0.10} to word value  
        text = re.sub(r'\{[$£€]\d+.\d+\}', "value", text)
        return text
    
   
    def obfuscate_date(self,text):
        # Convert obfuscate_date (like XX/XX/XXXX) to word date  
        text = re.sub(r'XX/XX/XXXX', "date", text)
        return text
         

    def to_lowercase(self,words):
        # Convert all characters to lowercase from list of tokenized words
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self,words):
        # Remove punctuation from list of tokenized words
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    

    def replace_numbers(self,words):
        # Replace all interger occurrences in list of tokenized words with textual representation
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self,words):
        # Remove stop words from list of tokenized words
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
        
    def normalize(self,text):
        sample = self.money_value(text)
        sample = self.obfuscate_date(sample)
        sample = self.replace_contractions(sample)
        words = nltk.word_tokenize(sample)
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
        words = self.replace_numbers(words)

        return re.sub(r' \.', '.', " ".join(words))

