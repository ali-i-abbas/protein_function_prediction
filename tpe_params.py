import math
import itertools
import shutil
import sys
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Layer, InputSpec, Dropout
from tensorflow.keras.utils import Sequence, to_categorical
from collections import deque
from hyperopt import Trials, fmin, hp, atpe, space_eval
from pprint import pprint
from pathlib import Path
import errno
import os

try:
    os.makedirs("params")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
CODES = 'ACDEFGHIKLMNPQRSTVWY'

subontology = 'mf'
max_evals = 300
nb_epoch = 10
batch_size = 64



df = pd.DataFrame(
    columns=['c1', 'c2', 'c3', 'c4', 'encoding', 'k', 'layer1', 'layer2', 'last_fc_units', 'accuracy', 'loss'])
model_path = 'params/'
trials = None
min_loss = math.inf
encoding = 'oh'
gram_len = 1


def main():
    start_time = time.time()

    parameter_space = {
        'c1': get_conv_param(1),
        'c2': get_conv_param(2),
        'c3': get_conv_param(3),
        'c4': get_conv_param(4),
        'k': hp.choice('k', [1, 3, 6, 8, 10, 15, 20, 40, 80]),
        'layer1': get_layer_param(1),
        'last_fc_units': hp.choice('last_fc_units', [32, 64, 128, 256, 512, 1024, 2048])
    }

    trial_count = 0
    global df
    global min_loss
    global trials
    if Path("hyperopt_trials.pkl").is_file():
        trials = joblib.load('hyperopt_trials.pkl')

        df = pd.read_pickle("parameters.pkl")

        trial_count = len(trials)
        best_params = df.loc[df['loss'].idxmin()].to_dict()
        min_loss = best_params['loss']
    else:
        trials = Trials()

    while trial_count < max_evals:
        try:
            flat_best_params = fmin(
                train_model,
                trials=trials,
                space=parameter_space,
                algo=atpe.suggest,
                max_evals=max_evals,
                show_progressbar=True
            )
            trial_count = max_evals
        except:
            if Path("hyperopt_trials.pkl").is_file():
                del trials._dynamic_trials[-1]
                del trials._trials[-1]
                trials._ids.remove(max(trials._ids))
                joblib.dump(trials, 'hyperopt_trials.pkl')
                trial_count = len(trials)
            else:
                trials = Trials()
                trial_count = 0

    joblib.dump(trials, 'hyperopt_trials.pkl')

    print('\nFmin Best Parameters:')
    pprint(space_eval(parameter_space, flat_best_params))

    best_params = df.loc[df['loss'].idxmin()].to_dict()

    print('\nBest Parameters:')
    pprint(best_params)

    with open("best_param.log", "w") as best_param_log:
        pprint(best_params, best_param_log)

    print('\n-------------------------------------\n')

    print("--- %s seconds ---" % (time.time() - start_time))

    vocab = {}
    for index, gram in enumerate(itertools.product(CODES, repeat=gram_len)):
        vocab[''.join(gram)] = index + 1

    test_generator = DataGenerator(encoding, test_data, test_labels, batch_size, nb_classes, vocab, gram_len)

    predictions = predict(model_path + 'best_model', test_generator)

    predict_time = time.time() - start_time
    print()
    print("--- predict_time %s seconds ---" % predict_time)
    print()

    print('Evaluation on test data')
    f, p, r, t, preds_max = compute_performance(predictions, test_labels, test_gos, all_subontology_set,
                                                gene_ontology, subontology_id, subontology_set)
    print('Fmax\t\tPrecision\tRecall\t\tThreshold\n%f\t%f\t%f\t%f' % (f, p, r, t))

    # main_plot_history(trials)
    # main_plot_histogram(trials)


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

    # code for encoding amino acids in less dimensions by functional and structural properties
    # is from encode_x.py and seq_util.py from https://pitgroup.org/seclaf/

    # Gets the expected charge of an amino acid (1: positive, -1: negative, 0: neutral).
    # Since "H" is positive only 10% of the time, 0.1 is returned for "H".
    def getAminoAcidCharge(self, x):
        if x in "KR":
            return 1.0
        if x == "H":
            return 0.1
        if x in "DE":
            return -1.0
        return 0.0

    # Returns the hydrophobicity of an amino acid (identified by its letter).
    def getAminoAcidHydrophobicity(self, x):
        AminoAcids = "ACDEFGHIKLMNPQRSTVWY"
        _hydro = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2,
                  -0.9, -1.3]
        return _hydro[AminoAcids.find(x)]

    # Returns whether the amino acid (identified by its letter) is polar.
    def isAminoAcidPolar(self, x):
        return x in "DEHKNQRSTY"

    # Returns whether the amino acid (identified by its letter) is an aromatic compound.
    def isAminoAcidAromatic(self, x):
        return x in "FWY"

    # Returns whether the amino acid (identified by its letter) has a hydroxyl group.
    def hasAminoAcidHydroxyl(self, x):
        return x in "ST"

    # Returns whether the amino acid (identified by its letter) has a sulfur atom.
    def hasAminoAcidSulfur(self, x):
        return x in "CM"

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.inputs[k] for k in index]

        if self.has_targets:
            labels = np.asarray([self.targets[k] for k in index])

        ngrams = list()

        np_prots = list()
        for seq in batch:
            grams = np.zeros((len(seq) - self.gram_len + 1,), dtype='int32')
            np_prot = list()
            for i in range(len(seq) - self.gram_len + 1):
                a = seq[i]
                descArray = [float(x) for x in
                             [self.getAminoAcidCharge(a), self.getAminoAcidHydrophobicity(a), self.isAminoAcidPolar(a),
                              self.isAminoAcidAromatic(a), self.hasAminoAcidHydroxyl(a), self.hasAminoAcidSulfur(a)]]
                np_prot.append(descArray)
                grams[i] = self.vocab[seq[i: (i + self.gram_len)]]
            np_prots.append(np_prot)
            ngrams.append(grams)

        maxLength = MAXLEN - self.gram_len + 1

        np_prots = sequence.pad_sequences(np_prots, maxlen=maxLength)
        ngrams = sequence.pad_sequences(ngrams, maxlen=maxLength)

        res_inputs = to_categorical(ngrams, num_classes=len(self.vocab) + 1)

        res_inputs = np.concatenate((res_inputs, np_prots), 2)

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


def get_conv_param(layer_no):
    layer_name = 'c' + str(layer_no)
    c = hp.choice(layer_name, [
        {
            layer_name: 0
        },
        {
            layer_name: 1,
            'filters' + str(layer_no): hp.choice('filters' + str(layer_no), [32, 64, 128, 256, 512]),
            'kernel_size' + str(layer_no): hp.choice('kernel_size' + str(layer_no), [3, 5, 7, 9, 11, 13]),
            'pool_size' + str(layer_no): hp.choice('pool_size' + str(layer_no), [2, 3, 4, 5, 7, 9, 11]),
            'strides' + str(layer_no): hp.choice('strides' + str(layer_no), [1, 2, 3, 4, 5]),
            'dropout' + str(layer_no): hp.choice('dropout' + str(layer_no), [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        }
    ])
    return c


def get_layer_param(layer_no):
    layer_name = 'layer' + str(layer_no)
    layer = hp.choice(layer_name, [
        {
            layer_name: 0
        },
        {
            layer_name: 1,
            'units' + str(layer_no): hp.choice('units' + str(layer_no), [32, 64, 128, 256, 512, 1024, 2048]),
            'layer_dropout' + str(layer_no): hp.choice('layer_dropout' + str(layer_no), [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        }
    ])
    return layer


def load_data(subontology):
    df = pd.read_pickle(DATA_PATH + 'train' + '-' + subontology + '.pkl')
    n = len(df)
    index = df.index.values

    valid_n = int(n * 0.875)
    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:]]

    test_df = pd.read_pickle(DATA_PATH + 'test' + '-' + subontology + '.pkl')

    return train_df, valid_df, test_df


def get_data(data_frame):
    data = data_frame['sequences'].values
    labels = (lambda v: np.hstack(v).reshape(len(v), len(v[0])))(data_frame['labels'].values)
    return data, labels


def train_model(parameters):
    try:
        vocab = {}
        for index, gram in enumerate(itertools.product(CODES, repeat=gram_len)):
            vocab[''.join(gram)] = index + 1
        vocab_len = len(vocab)

        train_generator = DataGenerator(encoding, train_data, train_labels, batch_size, nb_classes, vocab, gram_len)
        valid_generator = DataGenerator(encoding, val_data, val_labels, batch_size, nb_classes, vocab, gram_len)

        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(model_path + 'model.h5', monitor='val_loss', verbose=1,
                                               save_best_only=True)
        ]

        model = tf.keras.Sequential()

        model.add(tf.keras.Input(shape=(MAXLEN - gram_len + 1, vocab_len + 7)))

        clayers_count = 4
        for clayer in ['1', '2', '3', '4']:
            if parameters['c' + clayer]['c' + clayer] == 1:
                model.add(tf.keras.layers.Conv1D(
                    filters=parameters['c' + clayer]['filters' + clayer],
                    kernel_size=parameters['c' + clayer]['kernel_size' + clayer],
                    activation='relu',
                    strides=1))
                model.add(Dropout(parameters['c' + clayer]['dropout' + clayer]))
                model.add(tf.keras.layers.MaxPooling1D(pool_size=parameters['c' + clayer]['pool_size' + clayer],
                                                       strides=parameters['c' + clayer]['strides' + clayer]))
            else:
                clayers_count -= 1

        # we need at least one conv layer, otherwise return
        if clayers_count == 0:
            tf.keras.backend.clear_session()
            return np.Inf

        model.add(KMaxPooling(k=parameters['k']))

        model.add(tf.keras.layers.Flatten())

        for layer in ['1']:
            if parameters['layer' + layer]['layer' + layer] == 1:
                if parameters['last_fc_units'] >= 1024 and parameters['layer' + layer]['units' + layer] > 1024:
                    # too many fully connected units so we abort
                    return np.Inf
                model.add(tf.keras.layers.Dense(units=parameters['layer' + layer]['units' + layer], activation='relu'))
                model.add(Dropout(parameters['layer' + layer]['layer_dropout' + layer]))

        model.add(tf.keras.layers.Dense(units=parameters['last_fc_units'], activation='relu'))

        model.add(tf.keras.layers.Dense(units=nb_classes, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        model.summary()

        # train the model
        history = model.fit(
            x=train_generator,
            epochs=nb_epoch,
            validation_data=valid_generator,
            verbose=2,
            callbacks=callbacks_list)

        accuracy = max(history.history['val_accuracy'])
        loss = min(history.history['val_loss'])

        global min_loss
        if loss < min_loss:
            min_loss = loss
            shutil.copy(model_path + 'model.h5', model_path + 'best_model.h5')

        parameters['accuracy'] = accuracy
        parameters['loss'] = loss

        if Path("hyperopt_trials.pkl").is_file():
            shutil.copy('hyperopt_trials.pkl', 'hyperopt_trials_backup.pkl')
            shutil.copy('parameters.pkl', 'parameters_backup.pkl')
            shutil.copy('parameters.csv', 'parameters_backup.csv')

        global df
        df = df.append(parameters, ignore_index=True)
        df.to_pickle("parameters.pkl")
        df.to_csv('parameters.csv')
        joblib.dump(trials, 'hyperopt_trials.pkl')
    except:
        loss = np.inf

    tf.keras.backend.clear_session()

    return loss


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


# this part that deals with GO loading is from https://github.com/bio-ontology-research-group/deepgo/blob/master/nn_hierarchical_seq.py
# START
subontology_id = FUNC_DICT[subontology]
gene_ontology = get_gene_ontology('go.obo')
subontology_df = pd.read_pickle(DATA_PATH + subontology + '.pkl')
subontologies = subontology_df['functions'].values
subontology_set = set(subontologies)
all_subontology_set = get_go_set(gene_ontology, subontology_id)
# END

nb_classes = len(subontologies)

train_df, valid_df, test_df = load_data(subontology)
test_gos = test_df['gos'].values
train_data, train_labels = get_data(train_df)
val_data, val_labels = get_data(valid_df)
test_data, test_labels = get_data(test_df)

if __name__ == '__main__':
    main()
