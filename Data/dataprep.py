# Imports
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def readdata():
    # Read data
    train_en_path = './data/Data (WMT 2016)/training/train.en'
    train_de_path = './data/Data (WMT 2016)/training/train.de'

    test_en_path = './data/Data (WMT 2016)/test/test.en'
    test_de_path = './data/Data (WMT 2016)/test/test.de'

    val_en_path = './data/Data (WMT 2016)/validation/val.en'
    val_de_path = './data/Data (WMT 2016)/validation/val.de'

    f_train_en = open(train_en_path, 'r', encoding='utf-8')
    train_en = f_train_en.read()
    f_train_en.close()

    f_train_de = open(train_de_path, 'r', encoding='utf-8')
    train_de = f_train_de.read()
    f_train_de.close()

    f_test_en = open(test_en_path, 'r', encoding='utf-8')
    test_en = f_test_en.read()
    f_test_en.close()

    f_test_de = open(test_de_path, 'r', encoding='utf-8')
    test_de = f_test_de.read()
    f_test_de.close()

    f_val_en = open(val_en_path, 'r', encoding='utf-8')
    val_en = f_val_en.read()
    f_val_en.close()

    f_val_de = open(val_de_path, 'r', encoding='utf-8')
    val_de = f_val_de.read()
    f_val_de.close()

    return train_en, train_de, test_en, test_de, val_en, val_de


def clean(data=None):
    data = re.sub('[0-9]+p*', 'n', data)  # replace all numbers with n
    data = re.sub('  ', ' ', data)  # remove double spaces
    data = re.sub("'", '', data)  # remove apostrophe
    data = data.split('\n')
    return list(map(lambda sentence: '<BOS> ' + sentence + ' <EOS> ', data))


def get_sen_len(data):
    data_sen_len = []
    for sentence in data:
        data_sen_len.append(len(sentence))
    return data_sen_len


# Tokenize text and tokenize
def tokenize(train=None, val=None, test=None, max_words=None,
             max_length=None, min_frequency=1):
    # Clean up the data
    train = clean(train)
    val = clean(val)
    test = clean(test)

    # Tokenize
    tokenizer = Tokenizer(num_words=max_words, lower=True, oov_token='<UNK>')
    tokenizer.fit_on_texts(np.concatenate((train, val, test), axis=0))
    vocab = {k: v for k, v in tokenizer.word_counts.items() if v >= min_frequency}
    max_words = len(vocab)

    # Tokenize with appropriate max_word length
    tokenizer = Tokenizer(num_words=max_words, lower=True, oov_token='<UNK>')
    tokenizer.fit_on_texts(np.concatenate((train, val, test), axis=0))
    train = tokenizer.texts_to_sequences(train)
    val = tokenizer.texts_to_sequences(val)
    test = tokenizer.texts_to_sequences(test)

    # Record the length of each sentence
    train_sen_len = get_sen_len(train)
    val_sen_len = get_sen_len(val)
    test_sen_len = get_sen_len(test)

    # Pad
    train = pad_sequences(train, maxlen=max_length, truncating='post',
                          padding='post', value=0)
    val = pad_sequences(val, maxlen=max_length, truncating='post',
                        padding='post', value=0)
    test = pad_sequences(test, maxlen=max_length, truncating='post',
                         padding='post', value=0)

    # Get inverted dictionary
    i2w = {v: k for k, v in tokenizer.word_index.items()}

    return train, val, test, train_sen_len, val_sen_len, test_sen_len, \
           tokenizer.word_index, i2w, max_words


# Create Glove Embedding dictionary
def create_embedding_indexmatrix(max_words=None, embedding_dim=300, dict_en=None):
    # read in embedding matrix
    embeddings_index = {}
    f = open('./data/glove/glove.6B.300d.txt', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # compute glove embedding dictionary for words
    glove_embedding_matrix = np.zeros((max_words, embedding_dim))
    glove_embedding_matrix[0] = embeddings_index.get('unk')
    for j in range(max_words):
        word = dict_en.get(j)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            glove_embedding_matrix[j] = embedding_vector
        else:
            # words not found in embedding index will the word embedding of unk
            glove_embedding_matrix[j] = embeddings_index.get('unk')

    del embeddings_index
    return glove_embedding_matrix
