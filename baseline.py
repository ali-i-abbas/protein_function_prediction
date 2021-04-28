import math
import itertools
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Layer, InputSpec, Dropout
from tensorflow.keras.utils import Sequence, to_categorical
from collections import deque
import errno
import os

try:
    os.makedirs("models")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

sys.setrecursionlimit(100000)

DATA_PATH = 'data/'
MAXLEN = 1002
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

# encoding: 'oh' - one hot encoding, 'ad' - ad-hoc trainable embeddings
def main(encoding='oh', subontology='mf', nb_epoch=15, gram_len=1, embedding_size=20, batch_size=64, run=1):

    tf.keras.backend.clear_session()

    if encoding == 'oh':
        embedding_size = 0

    model_path = 'models/baseline_' + encoding + '_' + subontology + '_e' + str(nb_epoch) \
                 + '_b' + str(batch_size) + '_n' + str(gram_len) + '_v' + str(embedding_size) + '_r' + str(run)

    logging.basicConfig(filename=model_path + '.log', format='%(message)s', filemode='w', level=logging.INFO)
    logging.info('Model: %s' % model_path)

    # this part that deals with GO loading is from https://github.com/bio-ontology-research-group/deepgo/blob/master/nn_hierarchical_seq.py
    # START
    subontology_id = FUNC_DICT[subontology]
    gene_ontology = get_gene_ontology('go.obo')
    subontology_df = pd.read_pickle(DATA_PATH + subontology + '.pkl')
    subontologies = subontology_df['functions'].values
    subontology_set = set(subontologies)
    all_subontology_set = get_go_set(gene_ontology, subontology_id)
    logging.info('Subontologies: %s %d' % (subontology, len(subontologies)))
    # END



    nb_classes = len(subontologies)

    train_df, valid_df, test_df = load_data(subontology)
    test_gos = test_df['gos'].values
    train_data, train_labels = get_data(train_df)
    val_data, val_labels = get_data(valid_df)
    test_data, test_labels = get_data(test_df)
    logging.info("Training data size: %d" % len(train_data))
    logging.info("Validation data size: %d" % len(val_data))
    logging.info("Test data size: %d" % len(test_data))

    codes = 'ACDEFGHIKLMNPQRSTVWY'
    vocab = {}
    for index, gram in enumerate(itertools.product(codes, repeat=gram_len)):
        vocab[''.join(gram)] = index + 1
    print(('Gram length:', gram_len))
    print(('Vocabulary size:', len(vocab)))

    train_generator = DataGenerator(encoding, train_data, train_labels, batch_size, nb_classes, vocab, gram_len)
    valid_generator = DataGenerator(encoding, val_data, val_labels, batch_size, nb_classes, vocab, gram_len)
    test_generator = DataGenerator(encoding, test_data, test_labels, batch_size, nb_classes, vocab, gram_len)


    start_time = time.time()

    train_model(model_path=model_path, nb_classes=nb_classes, gram_len=gram_len, embedding_size=embedding_size,
                vocab_len=len(vocab), train_generator=train_generator, valid_generator=valid_generator, nb_epoch=nb_epoch)

    run_time = time.time() - start_time
    print()
    print("--- %s seconds ---" % run_time)
    print()


    predictions = predict(model_path, test_generator)


    logging.info('Evaluation on test data')
    print('Evaluation on test data')
    f, p, r, t, preds_max = compute_performance(predictions, test_labels, test_gos, all_subontology_set, gene_ontology, subontology_id, subontology_set)
    logging.info('Fmax\t\tPrecision\tRecall\t\tThreshold\n%f\t%f\t%f\t%f' % (f, p, r, t))
    print('Fmax\t\tPrecision\tRecall\t\tThreshold\n%f\t%f\t%f\t%f' % (f, p, r, t))

    return f, p, r, t, run_time


class DataGenerator(Sequence):

    def __init__(self, encoding, inputs, targets, batch_size, num_outputs, vocab, gram_len, shuffle=False):
        self.start = 0
        self.encoding = encoding
        self.inputs = inputs
        self.targets = targets
        self.size = len(self.inputs)
        if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
            self.size = len(self.inputs[0])
        self.has_targets = targets is not None
        self.batch_size = batch_size
        self.num_outputs = num_outputs
        self.vocab = vocab
        self.gram_len = gram_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.inputs[k] for k in index]

        if self.has_targets:
            labels = np.asarray([self.targets[k] for k in index])

        ngrams = list()

        for seq in batch:
            grams = np.zeros((len(seq) - self.gram_len + 1,), dtype='int32')
            for i in range(len(seq) - self.gram_len + 1):
                grams[i] = self.vocab[seq[i: (i + self.gram_len)]]
            ngrams.append(grams)

        maxLength = MAXLEN - self.gram_len + 1

        ngrams = sequence.pad_sequences(ngrams, maxlen=maxLength)

        if self.encoding == 'ad':
            res_inputs = ngrams
        else:
            res_inputs = to_categorical(ngrams, num_classes=len(self.vocab) + 1)

        if self.has_targets:
            return res_inputs, labels, [None]
        return res_inputs

    def on_epoch_end(self):
        self.index = np.arange(len(self.inputs))
        if self.shuffle == True:
            np.random.shuffle(self.index)

# code from https://github.com/chojc408/Dynamic-CNN-using-Global_k_MaxPooling1D/blob/cbedb157663b874df070619bb87dc058df1d3ae8/DCNN.py#L12
@tf.keras.utils.register_keras_serializable()
class KMaxPooling(Layer):

    def __init__(self, k=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[1] * self.k))

    def call(self, top_k):
        top_k = tf.transpose(top_k, [0, 2, 1])
        top_k = tf.nn.top_k(top_k, k=self.k, sorted=True, name=None)[0]
        top_k = tf.transpose(top_k, [0, 2, 1])
        return top_k

    def get_config(self):
        config = super(KMaxPooling, self).get_config()
        config.update({"k": self.k})
        return config

def load_data(subontology):

    df = pd.read_pickle(DATA_PATH + 'train' + '-' + subontology + '.pkl')
    df = df.sample(frac=1).reset_index(drop=True)
    n = len(df)
    index = df.index.values

    valid_n = int(n * 0.875)
    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:]]

    test_df = pd.read_pickle(DATA_PATH + 'test' + '-' + subontology + '.pkl')

    return train_df, valid_df, test_df


def get_data(data_frame):
    print((data_frame['labels'].values.shape))
    data = data_frame['sequences'].values
    labels = (lambda v: np.hstack(v).reshape(len(v), len(v[0])))(data_frame['labels'].values)
    return data, labels

def train_model(model_path, nb_classes, gram_len, embedding_size, vocab_len, train_generator, valid_generator, nb_epoch):



    callbacks_list = [tf.keras.callbacks.ModelCheckpoint(model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True),
                      tf.keras.callbacks.CSVLogger(filename=model_path + '_train.log', append=True)
                      ]

    model = tf.keras.Sequential()


    if embedding_size == 0:
        model.add(tf.keras.Input(shape=(MAXLEN - gram_len + 1, vocab_len + 1)))
    else:
        model.add(tf.keras.layers.Embedding(
            MAXLEN - gram_len + 1,
            embedding_size,
            input_length=MAXLEN))
    model.add(tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=9,
        activation='relu',
        strides=1))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3))

    model.add(tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=7,
        activation='relu',
        strides=1))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3))

    model.add(tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=7,
        activation='relu',
        strides=1))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3))

    model.add(KMaxPooling(k=10))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=32, activation='relu'))

    model.add(tf.keras.layers.Dense(units=nb_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    model.summary()

    logging.info('Training the model')

    # train the model
    history = model.fit(
        x=train_generator,
        epochs=nb_epoch,
        validation_data=valid_generator,
        verbose=2,
        callbacks=callbacks_list)

    np.savez_compressed(model_path + '_history.npz', tr_acc=history.history['accuracy'], tr_loss=history.history['loss'],
                        val_acc=history.history['val_accuracy'], val_loss=history.history['val_loss'])

    loss_train = min(history.history['loss'])
    accuracy_train = max(history.history['accuracy'])


    logging.info("Training Loss: %f" % loss_train)
    logging.info("Training Accuracy: %f" % accuracy_train)
    print('\nLog Loss and Accuracy on Train Dataset:')
    print("Loss: {}".format(loss_train))
    print("Accuracy: {}".format(accuracy_train))
    print()

    loss_val = min(history.history['val_loss'])
    accuracy_val = max(history.history['val_accuracy'])

    logging.info("Validation Loss: %f" % loss_val)
    logging.info("Validation Accuracy: %f" % accuracy_val)
    print('\nLog Loss and Accuracy on Val Dataset:')
    print("Loss: {}".format(loss_val))
    print("Accuracy: {}".format(accuracy_val))
    print()

    plt.clf()
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.ylim(ymin=0.6, ymax=1.1)
    plt.savefig(model_path + "_accuracy.png", type="png", dpi=300)

    plt.clf()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(model_path + "_loss.png", type="png", dpi=300)

    plt.clf()



def predict(model_path, test_generator):
    print('Loading the best model from %s' % model_path + '.h5')
    model = tf.keras.models.load_model(model_path + '.h5', custom_objects={'KMaxPooling': KMaxPooling})
    print('Predicting test data')
    preds = model.predict(test_generator)
    return preds

# code from https://github.com/bio-ontology-research-group/deepgo/blob/d97447a05c108127fee97982fd2c57929b2cf7eb/nn_hierarchical_seq.py#L517
def compute_performance(preds, labels, gos, all_subontology_set, gene_ontology, subontology_id, subontology_set):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_subontology_set:
                    all_gos |= get_anchestors(gene_ontology, go_id)
            all_gos.discard(subontology_id)
            all_gos -= subontology_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max

# code from https://github.com/bio-ontology-research-group/deepgo/blob/d97447a05c108127fee97982fd2c57929b2cf7eb/utils.py#L56
def get_gene_ontology(filename='go.obo'):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open('data/' + filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go

# code from https://github.com/bio-ontology-research-group/deepgo/blob/d97447a05c108127fee97982fd2c57929b2cf7eb/utils.py#L104
def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set

# code from https://github.com/bio-ontology-research-group/deepgo/blob/d97447a05c108127fee97982fd2c57929b2cf7eb/utils.py#L125
def get_go_set(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set


if __name__ == '__main__':
    main()