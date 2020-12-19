import pickle
from sklearn.model_selection import train_test_split
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Embedding
from keras.utils import plot_model
from keras.models import load_model, save_model
from IPython.display import SVG, display
#from keras.utils.vis_utils import model_to_dot
import keras.backend as K
from sys import getsizeof
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from collections import Counter
from numpy import argmax
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt

#matplotlib.rcParams['text.usetex'] = False
# matplotlib.use('Agg')
import time
import sys
from utils import *
import inspect

manager = Manager()

all_results = manager.dict()  # this is for sharing data across processes and also store the features data that will be used for clustering

modeling_scenarios = OrderedDict()


data_save_path = "./data_combine/"
P_model_name = "Pretrain_conc"
P_trace = data_save_path + "train.out"
P_test = data_save_path + "test.out"
P_set_total_number=200000*(int(sys.argv[1]))
P_epoch = int(sys.argv[2])#2
model_file_name ="./model/Pretrain_conc.h5"
B_use_exist = False

My_pickle=data_save_path

def convert_to_binary(data=None, bit_size=16, multidimensional_input=False):
    """
    Input: an array of integers
    Returns a numpy array of arrays where each number is represented with a list of its binary digits
    IMPORTANT: Currently 16 bit conversion is implemented below. For difference sizes, change 16 to something else.
    """
    if multidimensional_input:
        if bit_size == 16:
            # [[int(d) for d in str('{0:016b}'.format(i)) for i in y] for y in x]
            dataset = np.array([[int(d) for i in j for d in str('{0:016b}'.format(i))] for j in list(data)])
        elif bit_size == 32 or bit_size not in [16, 32]:
            if bit_size != 32:
                print("Using 32bits delta representation")
            dataset = np.array([[int(d) for i in j for d in str('{0:032b}'.format(i))] for j in list(data)])
    else:
        if bit_size == 16:
            dataset = np.array([[int(d) for d in str('{0:016b}'.format(x))] for x in list(data)])
        elif bit_size == 32 or bit_size not in [16, 32]:
            if bit_size != 32:
                print("Using 32bits delta representation")
            dataset = np.array([[int(d) for d in str('{0:032b}'.format(x))] for x in list(data)])
    return dataset

def plot_train_test_model_performance(train_history=None, app_name="", scenario_name=None):
    # TODO: optimize function and remove key accesses from the dictionaries (pass them as params)
    print(inspect.getouterframes(inspect.currentframe())[0].function)
    if train_history is not None:
        print('Final Training accuracy: %s' % (train_history.history['accuracy'][-1]))
        scenario_name = scenario_name.replace("_", "-").replace(" ", "-")
        data = train_history.history['accuracy']
        set_plot_size(6, 6)
        _ = plt.figure()
        _ = plt.plot(range(1, len(data) + 1), data, linestyle='-', marker='^', markersize=8, linewidth=2)
        _ = plt.xticks(range(1, len(data) + 1))
        _ = plt.title('Training Accuracy Per Epoch\nTrace %s' % (app_name))
        _ = plt.ylabel('Accuracy')
        _ = plt.xlabel('Epoch No.')
        _ = plt.grid(True)
        # _ = plt.legend(['Train'], loc='upper left')
        _ = plt.savefig(
            NOTEBOOK_PLOTS_DIRECTORY + "train_and_test_accuracy_for_%s_%s.png" % (app_name, scenario_name),
            bbox_inches='tight')
        _ = plt.show(block=False)

def difference(dataset, interval=1):
    """
    Calculates the difference between a time-series and a lagged version of it
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def difference16(data=None, lag=1, prune_lsb=False, prune_length=None):
    """
    Calculates the difference between a time-series and a lagged version of it that are represented in HEX format.
    This can be used to convert memory addresses to integers.
    """
    diff = list()
    for i in range(lag, len(data)):
        if prune_lsb:
            value = int(data[i][:-prune_length] + '0' * prune_length, 16) - int(
                data[i - lag][:-prune_length] + '0' * prune_length, 16)
        else:
            value = int(data[i], 16) - int(data[i - lag], 16)
        diff.append(value)
    return diff

def inverse_difference(last_ob, value):
    """
    Reconstructs the next value of a differenced time series.
    """
    return value + last_ob

params = {
    'axes.labelsize': 28,
    'font.size': 24,
    'legend.fontsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'text.usetex': False,  # True,
    # 'figure.figsize': [4.5, 4.5],
    'figure.facecolor': 'w',
    'figure.edgecolor': 'w',
    'axes.facecolor': 'w',
    'axes.edgecolor': 'gray',
    'savefig.facecolor': 'w',
    'savefig.edgecolor': 'g',
    'savefig.pad_inches': 0.1,
    'savefig.transparent': True,
    'axes.titlepad': 24,
    'axes.titlesize': 32
}
rcParams.update(params)

def precision(y_true, y_pred):
    """Precision metric. Only computes a batch-wise average of precision.
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
-    Only computes a batch-wise average of recall.
-    Computes the recall, a metric for multi-label classification of
-    how many relevant items are selected.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
-    The F score is the weighted harmonic mean of precision and recall.
-    Here it is only computed as a batch-wise average, not globally.
-    This is useful for multi-label classification, where input samples can be
-    classified as sets of labels. By only using accuracy (precision) a model
-    would achieve a perfect score by simply assigning every class to every
-    input. In order to avoid this, a metric should penalize incorrect class
-    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
-    computes this, as a weighted mean of the proportion of correct class
-    assignments vs. the proportion of incorrect class assignments.
-    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
-    correct classes becomes more important, and with beta > 1 the metric is
-    instead weighted towards penalizing incorrect class assignments.
-    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def encode_mem_accesses(data):
    """
    It implements a mapping between a set of strings to integers.
    It does not have a limit in the max_dictionary_size supported.

    data: data input should be in Pandas DF format
    """
    tmp = defaultdict(int)
    i = 1
    encoded = []
    for el in list(data):
        if not tmp[el]:
            # if el not in tmp.keys():
            tmp[el] = i
            i += 1
        encoded.append(tmp[el])
    return encoded

def create_windowed_dataset(data, look_back):
    """
    Create the dataset by grouping windows of memory accesses together (using the look_back parameter)

    data: it should be a list of integers
    """
    sequences = list()
    for i in range(look_back, len(data)):
        sequence = data[i - look_back:i + 1]
        sequences.append(sequence)
    return sequences

def get_timeseries_decomposition(data=None, frequency=None, plot=False, report_name_prefix=None,
                                 file_name_suffix=None, app_name=None):
    # TODO: reimplement this
    """
    data is a list of integers which is converted to address - date dataframe
    """
    dta = pd.DataFrame(data)
    dta['date'] = dta.index
    dta.columns = ["address", "date"]

    dta['date'] = pd.DatetimeIndex(dta.date)  # ,  format="%Y-%m-%d %H:%M:%S")
    # df.index = pd.to_datetime(df.index, unit='s')
    # dta.address.interpolate(inplace=True)
    dta.set_index('date', inplace=True)

    # dta = dta.groupby(pd.TimeGrouper('%ss' % 1), as_index=True)['total'].sum()
    res = sm.tsa.seasonal_decompose(dta, freq=frequency, model='additive')
    if plot:
        _ = plt.figure()
        _ = res.plot()
        _ = plt.title(
            "Time Series Decomposition Of %s Tokenized Sequence For Trace %s" % (report_name_prefix, app_name))
        _ = plt.grid(True)
        _ = plt.savefig(NOTEBOOK_PLOTS_DIRECTORY + "hist_%s" % (file_name_suffix), bbox_inches='tight')
        _ = plt.show(block=False)

    return list(res.trend['address']), list(res.seasonal['address']), list(res.resid['address'])

def plot_memory_trace(train_data=None, test_data=None, rows=2000, app_name="", scenario_name=None):
    set_plot_size(30, 4)
    scenario_name = scenario_name.replace("_", "-").replace(" ", "-")

    _ = plt.figure()
    _ = plt.plot(train_data[:rows])  # , linestyle='-', marker='.', markersize=6, linewidth=1)
    _ = plt.xlabel("Time")
    _ = plt.ylabel("Memory Address")
    _ = plt.title(
        "Memory Access Timeseries For Trace %s - Scenario %s - Training Phase" % (app_name, scenario_name))
    _ = plt.grid(True)
    _ = plt.savefig(NOTEBOOK_PLOTS_DIRECTORY + "memory_accesses_training_%s_%s.eps" % (app_name, scenario_name),
                    bbox_inches='tight')
    _ = plt.show(block=False)

    _ = plt.figure()
    _ = plt.plot(test_data[:rows])  # , linestyle='-', marker='.', markersize=6, linewidth=1)
    _ = plt.xlabel("Time")
    _ = plt.ylabel("Memory Address")
    _ = plt.title("Memory Access Timeseries For Trace %s - Scenario %s - Testing Phase" % (app_name, scenario_name))
    _ = plt.grid(True)
    _ = plt.savefig(NOTEBOOK_PLOTS_DIRECTORY + "memory_accesses_testing_%s_%s.eps" % (app_name, scenario_name),
                    bbox_inches='tight')
    _ = plt.show(block=False)

def plot_on_the_fly_model_performance(history=None, app_name="", scenario_name=None):
    # TODO: optimize function and remove key accesses from the dictionaries (pass them as params)
    # Plot training & validation accuracy values
    scenario_name = scenario_name.replace("_", "-").replace(" ", "-")

    set_plot_size(6, 6)
    _ = plt.figure()
    _ = plt.plot(range(1, len(history.history['acc']) + 1), history.history['acc'], linestyle='-', marker='^',
                 markersize=8, linewidth=2)
    _ = plt.plot(range(1, len(history.history['val_acc']) + 1), history.history['val_acc'], linestyle='-',
                 marker='s', markersize=8, linewidth=2)
    _ = plt.title('Model Accuracy For Trace %s\nScenario %s' % (app_name, scenario_name))
    _ = plt.ylabel('Accuracy')
    _ = plt.xlabel('Epoch No.')
    _ = plt.grid(True)
    _ = plt.legend(['Train', 'Test'], loc='upper left')
    _ = plt.savefig(NOTEBOOK_PLOTS_DIRECTORY + "train_test_accuracy_for_%s_%s.eps" % (app_name, scenario_name),
                    bbox_inches='tight')
    _ = plt.show(block=False)

    # Plot training & validation loss values
    _ = plt.figure()
    _ = plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], linestyle='-', marker='^',
                 markersize=8, linewidth=2)
    _ = plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], linestyle='-',
                 marker='s', markersize=8, linewidth=2)
    _ = plt.title('Model Loss For Trace %s\nScenario %s' % (app_name, scenario_name))
    _ = plt.ylabel('Loss')
    _ = plt.xlabel('Epoch No.')
    _ = plt.grid(True)
    _ = plt.legend(['Train', 'Test'], loc='upper left')
    _ = plt.savefig(NOTEBOOK_PLOTS_DIRECTORY + "train_test_loss_for_%s_%s.eps" % (app_name, scenario_name),
                    bbox_inches='tight')
    _ = plt.show(block=False)

def approx_entropy(U, m, r):
    try:
        U = np.array(U)

        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
            return (N - m + 1.0) ** (-1) * sum(np.log(C))

        N = len(U)
        return abs(_phi(m + 1) - _phi(m))
    except:
        print(U)
        raise

def sample_entropy(data):
    p_data = pd.Series(data).value_counts() / len(data)  # calculates the probabilities
    entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy

def dataset_creator(scenario):
    with open(My_pickle+'pretrain.pkl',
              'rb') as f:  # Python 3: open(..., 'rb')
        encoded_final, sequences, final_vocab_size, tokenizer, tokenizer2, max_test_accuracy, max_length, dummy_word, dummy_word_index, dummy_index, vocab_size_raw, dataset = pickle.load(
            f)
    return encoded_final, final_vocab_size, tokenizer, tokenizer2, max_test_accuracy, max_length, dummy_word, dummy_word_index, dummy_index, vocab_size_raw


if platform.system() == "Linux":
    PROJECT_ROOT_DIRECTORY = "/home/pengmiao/Project/MEMSYS/Pem/"
    TRACE_DIRECTORY = "/home/pengmiao/Project/MEMSYS/data/"
    sys.path.append(PROJECT_ROOT_DIRECTORY)
    os.environ["TMP"] = "/tmp"
    USE_GPU = False  # True
else:
    PROJECT_ROOT_DIRECTORY = data_save_path
    TRACE_DIRECTORY = data_save_path
    sys.path.append(PROJECT_ROOT_DIRECTORY)
    USE_GPU = True#False

# File Settings



DELETE_OLD_RESULTS = False

NUMBER_OF_PROCESSES = 2

PERFORM_EDA = False

NOTEBOOK_ID = "Pem_DC_LSTM_Pretrain"

README_TXT = """# README
## Ensemble modeling of memory access timeseries (using shapelets, LSTMs and more)
"""
###

# Inputs:
# TRACE_DIRECTORY = PROJECT_ROOT_DIRECTORY + "data/input/"

TRACE_FILE_NAMES = [
    'train.out'
]  # more to be added here

# Size of files in number of rows. This should be implemented as a single dict for both
TRACE_FILE_NAME_SIZES = {
    'swaptions_1_1M.out': 1000000,
    'blackscholes_1_1M.out': 1000000,
    'bodytrack_1_1M.out': 1000000,
    'canneal_1_1M.out': 1000000,
    'ferret_1_1M.out': 1000000,
    'fluidanimate_1_1M.out': 1000000,
}
###

# Outputs:
NOTEBOOK_ROOT_DIRECTORY = PROJECT_ROOT_DIRECTORY + "data/output/notebooks/%s/" % NOTEBOOK_ID
NOTEBOOK_PLOTS_DIRECTORY = NOTEBOOK_ROOT_DIRECTORY + "figs/"
NOTEBOOK_DATA_DIRECTORY = NOTEBOOK_ROOT_DIRECTORY + "data/"
NOTEBOOK_PICKLES_DIRECTORY = NOTEBOOK_ROOT_DIRECTORY + "pickles/"
NOTEBOOK_REPORT_DIRECTORY = NOTEBOOK_ROOT_DIRECTORY + "reports/"

###


CURRENT_TIMESTAMP = get_current_timestamp()
setup_report(data_dir=NOTEBOOK_ROOT_DIRECTORY, readme_text=README_TXT, delete_old_results=DELETE_OLD_RESULTS)
setup_report(data_dir=NOTEBOOK_PLOTS_DIRECTORY, delete_old_results=DELETE_OLD_RESULTS)
setup_report(data_dir=NOTEBOOK_DATA_DIRECTORY, delete_old_results=DELETE_OLD_RESULTS)
setup_report(data_dir=NOTEBOOK_PICKLES_DIRECTORY, delete_old_results=DELETE_OLD_RESULTS)
setup_report(data_dir=NOTEBOOK_REPORT_DIRECTORY, delete_old_results=DELETE_OLD_RESULTS)
set_plot_style_for_paper()

print("Destination Folder: %s" % NOTEBOOK_ROOT_DIRECTORY)
InteractiveShell.ast_node_interactivity = "all"

TRACE_FILE_NAMES_AND_VARIATIONS = []
for trace in TRACE_FILE_NAMES:
    if "old" not in trace:
        TRACE_FILE_NAMES_AND_VARIATIONS.append(trace)
        TRACE_FILE_NAMES_AND_VARIATIONS.append(trace.replace("1", "2"))
        TRACE_FILE_NAMES_AND_VARIATIONS.append(trace.replace("1.out", "1_repeat.out"))
    else:
        TRACE_FILE_NAMES_AND_VARIATIONS.append(trace)



def generator(My_pickle, batch_size):

    X_train = np.load(My_pickle + "x_train.npy", mmap_mode='r')
    y_train = np.load(My_pickle + "y_train.npy", mmap_mode='r')
    X_test = np.load(My_pickle + "X_test.npy", mmap_mode='r')
    y_test = np.load(My_pickle + "y_test.npy", mmap_mode='r')
    print("load X,y")

    samples_per_epoch = X_train.shape[0]
    number_of_batches = samples_per_epoch / batch_size
    counter = 0

    while 1:
        X_batch = X_train[batch_size * counter:batch_size * (counter + 1)]
        y_batch = y_train[batch_size * counter:batch_size * (counter + 1)]
        counter += 1
        yield X_batch, y_batch

        # restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0

def run_dc_lstm(P_trace, P_test, P_epoch, P_model_name, B_use_exist,scenario=None):
    """
    Encode the sequence of memory addresses to a sequence of integers
    We can do it either using Keras, or by implementing our own convertion function (see above "encode_mem_accesses")
    For the Keras approach, see documentation in the source code here: https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
    """

    use_manual_encoding = scenario['use_manual_encoding']
    app_name = scenario['app_name']
    decompose_timeseries = scenario['decompose_timeseries']
    decomposition_frequency = scenario['decomposition_frequency']
    test_ratio = scenario['test_ratio']
    on_the_fly_testing = scenario['on_the_fly_testing']
    plot_timeseries = scenario['plot_timeseries']
    look_back = scenario['look_back_window']
    scenario_name = scenario['scenario_name']
    vocabulary_maximum_size = scenario['vocabulary_maximum_size']
    vocabulary_mimimum_word_frequency_quantile = scenario['vocabulary_mimimum_word_frequency_quantile']
    model_diffs = scenario['model_diffs']
    lstm_batch_size = scenario['lstm_batch_size']
    lstm_epochs = scenario['lstm_epochs']
    verbosity = scenario['verbosity']
    dropout_ratio = scenario['dropout_ratio']
    lstm_size = scenario['lstm_size']
    embedding_size = scenario['embedding_size']
    prediction_batch_size = scenario['prediction_batch_size']
    online_retraining = scenario['online_retraining']
    online_learning_accuracy_threshold = scenario['online_learning_accuracy_threshold']
    online_retraining_periods = scenario['online_retraining_periods']
    online_retraining_period_size = scenario['online_retraining_period_size']
    number_of_rows_to_model = scenario['number_of_rows_to_model']
    model_type = scenario['model_type']
    loss_function = scenario['loss_function']
    activation_fuction = scenario['activation_function']
    # output compress
    convert_output_to_binary = scenario['convert_output_to_binary']  # this is used for FPGA implementation.
    # bit_size
    bit_size = scenario['bit_size']
    load_existing_pickles = scenario['load_existing_pickles']
    # ！！！new updata: input encode and compress
    encode_inputs = scenario['encode_inputs']
    # new, about cache
    CACHE_SIZES = scenario['CACHE_SIZES']
    CACHE_BLOCK_SIZES = scenario['CACHE_BLOCK_SIZES']
    CACHE_REPLACEMENT_ALGOS = scenario['CACHE_REPLACEMENT_ALGOS']
    # unique_key = abs(hash(frozenset(scenario.items()))), move to later function
    unique_key = scenario['id']
    misc_stats = {}  # miscellaneous statistics

    start_time = time.time()

    if online_retraining:
        return "online_retraining no function yet"
    else:
        # =====================================================================================================
        # Data preparation
        # =====================================================================================================
        encoded_final, final_vocab_size, tokenizer, tokenizer2, max_test_accuracy, max_length, dummy_word, \
        dummy_word_index, dummy_index, vocab_size_raw = dataset_creator(scenario)
       # sequences=[]
       # dataset=[]
        # =====================================================================================================

        # =====================================================================================================
        # Neural Network Configuration
        # =====================================================================================================
        if convert_output_to_binary:
            model = Sequential()
            model.add(Embedding(bit_size if encode_inputs else final_vocab_size, embedding_size,
                                input_length=(max_length - 1) * (bit_size if encode_inputs else 1)))
            '''keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', 
            embeddings_regularizer=None, activity_regularizer=None, 
            embeddings_constraint=None, mask_zero=False, input_length=None)'''
            # model.add(Embedding(final_vocab_size, embedding_size, input_length=max_length - 1))
            if USE_GPU:
                print("uG2")
                model.add(CuDNNLSTM(lstm_size))
            else:
                print("nG2")
                model.add(LSTM(lstm_size))
            model.add(Dropout(dropout_ratio))
            model.add(Dense(bit_size,
                            activation=activation_fuction))  # the size of this layer should align with the size of the bit representation of the output.
            model.compile(loss=loss_function, optimizer='adam', metrics=[
                'accuracy'])  # top_k_categorical_accuracy is still wrong but we have it for illustration purposes.
        else:
            model = Sequential()
            model.add(Embedding(final_vocab_size, embedding_size, input_length=max_length - 1))

            if USE_GPU:
                model.add(CuDNNLSTM(lstm_size))
            else:
                model.add(LSTM(lstm_size))
            model.add(Dropout(dropout_ratio))
            model.add(Dense(final_vocab_size, activation=activation_fuction))
            model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

        if verbosity > 0:
            print(model.summary())
       # SVG(model_to_dot(model, show_shapes=True, show_layer_names=False).create(prog='dot', format='svg'))
       # plot_model(model, to_file=NOTEBOOK_PLOTS_DIRECTORY + 'model_for_%s.png' % scenario_name, show_shapes=True,
       #            show_layer_names=False)

        if load_existing_pickles and os.path.isfile(model_file_name):
            model = load_model(model_file_name)
            train_history = None
            train_accuracy = -1  # train_history.history['acc'][-1]
        else:
            '''train_history = model.fit(X_train,
                                      y_train,
                                      epochs=lstm_epochs,
                                      verbose=verbosity,
                                      shuffle=False,
                                      batch_size=lstm_batch_size)'''
            train_history = model.fit_generator(
                 generator(My_pickle, lstm_batch_size),
                 epochs=lstm_epochs,
                 steps_per_epoch=P_set_total_number // lstm_batch_size)
            model.save(model_file_name)
            train_accuracy = train_history.history['accuracy'][-1]
          #  plot_train_test_model_performance(train_history, app_name=app_name, scenario_name=scenario_name)
            return 0, 0, 0
        # =====================================================================================================

trace_offsets = {
    'swaptions_1_1M.out': 100000,
    'bodytrack_1_1M.out': 100000,
    'canneal_1_1M.out': 100000,
    'ferret_1_1M.out': 100000,
    'fluidanimate_1_1M.out': 100000,
    'blackscholes_1_1M.out': 100000,
    'swaptions_old_1_1M.out': 100000,
    'blackscholes_old_1_1M.out': 100000,
    'fluidanimate_old_1_1M.out': 100000,
}

CACHE_SIZES = [16, 32, 64, 128]
CACHE_BLOCK_SIZES = [2, 4, 8, 16, 32, 64]
CACHE_REPLACEMENT_ALGOS = ["LRU"]

manager = Manager()

all_results = manager.dict()  # this is for sharing data across processes and also store the features data that will be used for clustering

modeling_scenarios = OrderedDict()

cache_performance_stats = {}

def lstm_modeling_worker(scenario):
    global all_results
    import json

    print("Executing Scenario %s" % scenario['scenario_name'])

    report_file_name = '%s/all_stats_%s_%s.json' % (NOTEBOOK_REPORT_DIRECTORY, scenario_name, scenario['id'])

    if not scenario['skip_run_if_previous_stats_found'] or not os.path.isfile(report_file_name):
        train_history, test_history, misc_stats = run_dc_lstm(P_trace, P_test, P_epoch, P_model_name, B_use_exist,scenario)
        all_results[scenario['scenario_name']] = [scenario, train_history, test_history, misc_stats]

        json = json.dumps(all_results[scenario['scenario_name']], indent=4)

        f = open(report_file_name, "w")
        f.write(json)
        f.close()
    else:
        print("Skipping run since past results found for the same configuration...")
    return all_results

# # 50/50 Split Offline Analysis
# WARNING: Some of the parameters have been removed from the implementation since the scope of this research (prefetching) is limited.
if True and trace in P_trace:
    for model_type in [
        "double_fpga"]:  # , "lsb_fpga", "vanilla"]:  # "vanilla",  "custom_loss_fpga" , "fpga", "lsb_fpga",
        scenario_counter = 1
        trace_short = trace.split(".")[0].replace("_mem", "").capitalize()
        for i in range(5, 6):
            # we store the scenario name as key, but also add it in the scenario configuration for availability in each function
            modeling_scenarios[
                'LSTM_%s_Offline_Prefetching_%s_%s' % (model_type.upper(), scenario_counter, trace_short)] = {
                # trace params
                'scenario_name': 'LSTM_%s_Offline_Prefetching_%s_%s' % (
                    model_type.upper(), scenario_counter, trace_short),
                'app_name': trace.split(".")[0].split("_")[0].capitalize(),
                'trace_file_name': trace,
                'load_existing_pickles': B_use_exist,
                'skip_run_if_previous_stats_found': True,

                # dataset params
                'keep_read_access_only': False,
                'number_of_rows_to_model': P_set_total_number,
                'number_of_rows_to_skip': 0,  # int(TRACE_FILE_NAME_SIZES[trace]*3.0/5.0),
                'pretrain_type': None,

                'model_diffs': True,
                # set to True if you want instead of the actual time series to model memory location differences.
                'vocabulary_maximum_size': 50000 if "vanilla" in model_type else 0,
                # this is to further reduce the dictionary size
                'vocabulary_mimimum_word_frequency_quantile': 0.95 if not "fpga" in model_type else 1,
                "prune_lsb": True if "lsb" in model_type else False,
                "prune_length": 1,
                # this corresponds to how many "letters" to be pruned from the HEX address (1 letter = 4 bits)
                "bit_size": 16,

                # Not Used
                'decompose_timeseries': False,  # not used #TODO: remove
                'decomposition_frequency': 10,  # not used #TODO: remove
                'use_manual_encoding': False,
                # not used #TODO: remove # True applies only to non-diff time series. --TODO: remove completely.

                # model params
                'model_type': model_type,  # None, #'fpga',
                'look_back_window': 3,
                'lstm_epochs': P_epoch,  # 20,
                'lstm_batch_size': 4096,# if "fpga" in model_type else 256,
                'lstm_size': 50,
                'dropout_ratio': 0.1,
                'embedding_size': 10,
                'verbosity': 1,
                'test_ratio': 0.1 * float(i),  # this does not apply to online learning
                'prediction_batch_size': 4096,
                # 4096,  # this has an impact only if on_the_fly_testing is disabled
                'loss_function': 'categorical_crossentropy' if not "fpga" in model_type else "binary_crossentropy" if model_type in [
                    "fpga", "lsb_fpga", "double_fpga"] else "",
                # 'categorical_crossentropy', 'binary_crossentropy', # custom_crossentropy,  # binary_cross_entropy is needed for multi-label classification, otherwise we need categorical_crossentropy
                'activation_function': "softmax" if not "fpga" in model_type else "sigmoid",
                'convert_output_to_binary': True if "fpga" in model_type else False,
                'encode_inputs': True if "fpga" in model_type else False,

                # run-time params
                'on_the_fly_testing': False,
                # if True, it will run testing on the whole data for each epoch (good for plotting performance). Not used if online_retraining is enabled.
                'plot_timeseries': False,

                'online_retraining': False,
                # if this is True, then we model number_of_rows_to_model samples and then generate predictions for until the accuracy becomes smaller than online_learning_accuracy_threshold, and then we retrain etc.
                'online_learning_accuracy_threshold': 0.6,
                # It is used only if online_retraining is set to True.
                'online_retraining_periods': 5,  # It is used only if online_retraining is set to True.
                'online_retraining_period_size': 10000,
                # How many predictions to run before measuring cummulative accuracy for the given period. It is used only if online_retraining is set to True.

                'CACHE_SIZES': CACHE_SIZES,
                'CACHE_BLOCK_SIZES': CACHE_BLOCK_SIZES,
                'CACHE_REPLACEMENT_ALGOS': CACHE_REPLACEMENT_ALGOS
            }
            scenario_counter += 1

# calculate a unique ID for each scenario based on its values
tmp_scenarios = modeling_scenarios.copy()
for scenario_name, scenario in tmp_scenarios.items():
    modeling_scenarios[scenario_name]['id'] = abs(
        hash(frozenset([(k, ''.join(str(v))) for k, v in modeling_scenarios.items()])))

if not USE_GPU:
    print("nG")
    # with ProcessPoolExecutor(max_workers=NUMBER_OF_PROCESSES) as e:
    for scenario_name, scenario in modeling_scenarios.items():
        # e.submit(lstm_modeling_worker, scenario)
        lstm_modeling_worker(scenario)
else:
    print("uG")
    for scenario_name, scenario in modeling_scenarios.items():
        lstm_modeling_worker(scenario)