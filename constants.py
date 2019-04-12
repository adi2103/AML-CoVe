# Imports
from __future__ import absolute_import, division, print_function

import gc
import numpy as np
import os
import re
import tensorflow as tf
import time
import warnings

from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Bidirectional, Dropout, Embedding, LSTM, Multiply, Lambda, Permute, \
    Reshape, Masking
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant, RandomUniform

from onmt import Dataset
from dataprep import *

# Initialize hyperparameters
MAX_INPUT_SIZE = 35
BATCH_SIZE = 128
EPOCHS = 24
LEARNING_RATE = 0.001

DROP_OUT = 0.2
R_DROP_OUT = 0.0
EMBEDDING_DIM = 300

# CNN hyper-parameters
CNN_FILTERS = 128
KERNEL_SIZE = 4

# Parameters for annealing, follow from https://arxiv.org/pdf/1611.02344.pdf
USE_SGD = False
SGD_LEARNING_RATE = 0.01
MIN_ANNEALING_RATE = 0.00001
CLIP_MIN = -5
CLIP_MAX = 5

# Hyperparameters for ATTN
ATTN_DROP_OUT = 0.1
D_K = 100

# Hyperparameters for BCN -> From paper
BCN_DROPOUT = 0.1
BCN_R_DROPOUT = 0.0
BCN_MAX_LENGTH = 35
N_TARGET = None
BCN_BATCH_SIZE = 128