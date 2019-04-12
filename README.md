# AML-CoVe
University of Oxford - Advanced Machine Learning Reproducibility Challenge. Contextualized Word Vectors (CoVe) - https://arxiv.org/pdf/1708.00107.pdf

Tensorflow 2.0-alpha is used

To train the models, first train the NMT. We used the WMT ('16) data to train, which can be found at http://www.statmt.org/wmt16/. 
Next, the CoVe encodings can be generated using the GetEncodings script. 
Lastly, the weights are loaded into the BCN or CNN classification network. We use the SST data set, which can be found at https://nlp.stanford.edu/sentiment/ 
