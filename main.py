import json
import os
import re
import tarfile
import tempfile
import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, LSTM, Dropout, TimeDistributed , Reshape, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils

#Config
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300
BATCH_SIZE = 128
MAX_EPOCHS = 30
MAX_LEN = 42
DP = 0.25
el = 0.1
USE_GLOVE = True
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'

#Get Data from json file
def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))

  return left, right, Y

#dataset be in shekle ke dar sotoone premisis har 3 jomle daghighan yeksan hastand. va dar sotoone dovvom hyposis
#3 jomle mokhtalef ba 3 label mokhtalef zakhire mishe.
#dar zir training be soorate [3][500k] ast. (training[0][0] == training[1][0] == training[2][0])
#training[0][0] -> premisis
#training[1][0] -> hyposis
#training[2][0] -> label ( [0 , 1, 0] -> neutral)
training = get_data('snli_1.0_train.jsonl')
validation = get_data('snli_1.0_dev.jsonl')
test = get_data('snli_1.0_test.jsonl')

#har kalame ro dare ye adad behesh nesbat mide
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

#tedade loghate monhaser be fard dar jomle
VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

#har jomle ro be arraye 42 tayi tabdil mikone
#jomle ro akhare arraye gharar mide
#be har kalame yek adad ekhtesas mide
#tokenizer.texts_to_sequences(X) -> bar asase rank har kalame, adade rankesh ro bejaye kalame minevesie. masaln (.) az hame
#bishtare be jash minevise 1, ya masalan woman -> 24
to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

#zakhire sazie dade ha dar 3 araye zir
training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)
#yek jomle englise -> yek araye 42 tayi adadi

print('Build model...')
print('Vocab size =', VOCAB)

#embed hidden siez = 300. yani har jomle be 42 bordare 300 tayi tabdil mishe (adad beine 0 ta 1)
#baraye estefade blstm
#har jomle hade aksar 42 kalame mitoone dashte bashe
GLOVE_STORE = 'precomputed_glove.weights'
if USE_GLOVE:
  if not os.path.exists(GLOVE_STORE + '.npy'):
    print('Computing GloVe')
  
    embeddings_index = {}
    f = open('glove.840B.300d.txt')
    for line in f:
      values = line.split(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
    #tokenizer dar zaman tekonize kardan liste kalamat ro dar embeddings_index zakhire mikone 
    for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
      else:
        print('Missing from GloVe: {}'.format(word))
  
    np.save(GLOVE_STORE, embedding_matrix)

  print('Loading GloVe')
  embedding_matrix = np.load(GLOVE_STORE + '.npy')

  print('Total number of null word embeddings:')
  print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

  #ye arraye sparse discrete ro be ye fazaye chand boedi continius bedoone sparse tabdil mikone
  #embeding be khodie khod similar ha ro kenare ham nemizare, bekhtare hamin az dataset hayi mesle
  #word2vec ya glove (dar inja glove) estefade kardim.
  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)
else:
  #inja ba farze naboode glove, serfan ye arraye 300 tayi be har kalame ekhtesas dade mishe
  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)


#build model

#laye voroodi -> (42)
premise = keras.layers.Input(shape=(MAX_LEN,))
hypothesis = keras.layers.Input(shape=(MAX_LEN,))

#emale embedding : (42) -> (42,300)
prem = embed(premise)
hypo = embed(hypothesis)


#normalize kardane beine 0 ta 1
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

#emale biLSTM bar rooye prem va hypo: (42,300) -> (600) 
bi_prem = Bidirectional(LSTM(input_dim=MAX_LEN,output_dim=300,return_sequences=False))(prem)
bi_prem = BatchNormalization()(bi_prem)
bi_hypo = Bidirectional(LSTM(input_dim=MAX_LEN,output_dim=300,return_sequences=False))(hypo)
bi_hypo = BatchNormalization()(bi_hypo)


########## piade sazie ghasmate inner attention
#amade size baraye emale meanPooling. reshape be in ellat ast ke tabe AveragePooling1D tensor3d be onvane voroodi mipazirad
# (600) -> (600,1)
bi_prem_reshape = keras.layers.Reshape((600,1))(bi_prem)
bi_hypo_reshape = keras.layers.Reshape((600,1))(bi_hypo)

# (600,1) -> (300,1)
meanPrem = keras.layers.AveragePooling1D(pool_size=2)(bi_prem_reshape)
meanHypo = keras.layers.AveragePooling1D(pool_size=2)(bi_hypo_reshape)

# (300,1) -> (300)
meanPrem = keras.layers.Reshape((300,))(meanPrem)
meanHypo = keras.layers.Reshape((300,))(meanHypo)

#kenare ham gozashtane 2 mean : (300) -> (600)
mean = keras.layers.concatenate([meanPrem,meanHypo])

# mmm = W(y)*Y + W(h)*R(avg)*el
mPrem = keras.layers.Add()([bi_prem,mean*el])
mHypo = keras.layers.Add()([bi_hypo,mean*el])
mPrem.trainable = False
mHypo.trainable = False
# M = tanh(mmm)
# chon dar laye Add natoonestam activation ro tanh konam inja in karo kardam
mPrem = keras.layers.Dense(600,activation='tanh')(mPrem)
mHypo = keras.layers.Dense(600,activation='tanh')(mHypo)

#alpha = softmax(W(t) * M)
alphaPrem = keras.layers.Dense(600,activation='softmax')(mPrem)
alphaHypo = keras.layers.Dense(600,activation='softmax')(mHypo)


#ijade sentence vector
################ if inner attention work!!! #################
# (600) * (600)
#sentence_prem = keras.layers.Multiply()([alphaPrem,bi_prem])
#sentence_hypo = keras.layers.Multiply()([alphaHypo,bi_hypo])

############### without inner attention!!! ##################
#khorooje meanPooling (ke rooye bilstm emal shod)
sentence_prem = bi_prem
sentence_hypo = bi_hypo

#zarbe sentence vector ha
multiply = keras.layers.Multiply()([sentence_prem, sentence_hypo])
#tafrighe sentence vector ha
subtracted = keras.layers.Subtract()([sentence_prem, sentence_hypo])

#normalize kardan
multiply = BatchNormalization()(multiply)
subtracted = BatchNormalization()(subtracted)


#kenare ham gozashtane laye ha : (42) + (600) + (600) + (42) -> (1284)
concatenateLayer = keras.layers.concatenate([premise,multiply,subtracted,hypothesis],axis=-1)
concatenateLayer = BatchNormalization()(concatenateLayer)

#emale 25 darsad dro out baraye laye akhar
concatenateLayer = Dropout(DP)(concatenateLayer)

#laye khorooje ba activation softmax
pred = Dense(len(LABELS), activation='softmax')(concatenateLayer)

#tarife model (elame laye voroodi va khorooji)
model = keras.models.Model(inputs=[premise, hypothesis], outputs=pred)

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

#print vaziate laye ha
model.summary()

#fit kardane model
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]))

#arzyabi model
loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
