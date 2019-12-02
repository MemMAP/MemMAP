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
from keras.utils.vis_utils import model_to_dot
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

matplotlib.rcParams['text.usetex'] = False
# matplotlib.use('Agg')
import time
import sys
from utils import *
import inspect

manager = Manager()

all_results = manager.dict()  # this is for sharing data across processes and also store the features data that will be used for clustering

modeling_scenarios = OrderedDict()

TRACE_FILE_NAMES = [
    'swaptions_1_1M.out',
    'bodytrack_1_1M.out',
    'canneal_1_1M.out',
    'ferret_1_1M.out',
    'fluidanimate_1_1M.out',
    'blackscholes_1_1M.out',
    'swaptions_old_1_1M.out',
    'blackscholes_old_1_1M.out',
    'fluidanimate_old_1_1M.out',
    'dedup_1_1M.out',
    'facesim_1_1M.out',
    'freqmine_1_1M.out',
    'raytrace_1_1M.out',
    'streamcluster_1_1M.out',
    'vips_1_1M.out',
    'x264_1_1M.out'
]  # more to be added here

def run_dc_lstm(P_trace, P_epoch):
    # def run_scenarios_diff(traces, traces_test, fpga, new_re):

    # fpga:["fpga", "lsb_fpga", "vanilla"]
    # new_re:["new","rerun"];rerun: same input, new: different input
    # traces:['swaptions_1_1M.out','bodytrack_1_1M.out','canneal_1_1M.out','ferret_1_1M.out','fluidanimate_1_1M.out','blackscholes_1_1M.out']

    # Model settings
    # # 50/50 Split Analysis
    # pretrain_type

    if platform.system() == "Linux":
        PROJECT_ROOT_DIRECTORY = "/home/MemMAP/"
        TRACE_DIRECTORY = "/home/MemMAP/data/"
        sys.path.append(PROJECT_ROOT_DIRECTORY)
        os.environ["TMP"] = "/tmp"
        USE_GPU = False  # True
    else:
        PROJECT_ROOT_DIRECTORY = "E:/home/MemMAP/"
        TRACE_DIRECTORY = "E:/home/MemMAP/data/"
        sys.path.append(PROJECT_ROOT_DIRECTORY)
        USE_GPU = False

    # File Settings

    DELETE_OLD_RESULTS = False

    NUMBER_OF_PROCESSES = 2

    PERFORM_EDA = False

    NOTEBOOK_ID = "Pem_DC_LSTM"

    README_TXT = """# README
    ## Ensemble modeling of memory access timeseries (using shapelets, LSTMs and more)
    """
    ###

    # Inputs:
    # TRACE_DIRECTORY = PROJECT_ROOT_DIRECTORY + "data/input/"

    # Size of files in number of rows. This should be implemented as a single dict for both
    TRACE_FILE_NAME_SIZES = {
        'swaptions_1_1M.out': 1000000,
        'blackscholes_1_1M.out': 1000000,
        'bodytrack_1_1M.out': 1000000,
        'canneal_1_1M.out': 1000000,
        'ferret_1_1M.out': 1000000,
        'fluidanimate_1_1M.out': 1000000,
        'dedup_1_1M.out': 1000000,
        'facesim_1_1M.out': 1000000,
        'freqmine_1_1M.out': 1000000,
        'raytrace_1_1M.out': 1000000,
        'streamcluster_1_1M.out': 1000000,
        'vips_1_1M.out': 1000000,
        'x264_1_1M.out': 1000000
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

    def plot_train_test_model_performance(train_history=None, test_history=None, app_name="", scenario_name=None):
        # TODO: optimize function and remove key accesses from the dictionaries (pass them as params)
        print(inspect.getouterframes(inspect.currentframe())[0].function)
        if train_history is not None:
            print('Final Training accuracy: %s' % (train_history.history['acc'][-1]))
            print('Test score: %s' % test_history[0])
            print('Test accuracy: %s' % test_history[1])
            scenario_name = scenario_name.replace("_", "-").replace(" ", "-")
            data = train_history.history['acc']
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
                NOTEBOOK_PLOTS_DIRECTORY + "train_and_test_accuracy_for_%s_%s.eps" % (app_name, scenario_name),
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

    def plot_train_test_model_performance(train_history=None, test_history=None, app_name="", scenario_name=None):
        # TODO: optimize function and remove key accesses from the dictionaries (pass them as params)
        if train_history is not None:
            print('Final Training accuracy: %s' % (train_history.history['acc'][-1]))
            print('Test score: %s' % test_history[0])
            print('Test accuracy: %s' % test_history[1])
            scenario_name = scenario_name.replace("_", "-").replace(" ", "-")
            data = train_history.history['acc']
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
                NOTEBOOK_PLOTS_DIRECTORY + "train_and_test_accuracy_for_%s_%s.eps" % (app_name, scenario_name),
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
        print(inspect.getouterframes(inspect.currentframe())[0].function)
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
        number_of_rows_to_skip = scenario['number_of_rows_to_skip']
        keep_read_access_only = scenario['keep_read_access_only']
        prune_lsb = scenario['prune_lsb']
        prune_length = scenario['prune_length']
        pretrain_type = scenario["pretrain_type"]
        bit_size = scenario["bit_size"]
        convert_output_to_binary = scenario['convert_output_to_binary']  # this is used for FPGA implementation.

        max_test_accuracy = 1
        # used to reduce accuracy appropriatelly in case of rounding/approximations in the prediction address.

        tokenizer = None
        tokenizer2 = None

        # This is used to represent rare words and not get encoded individually, which then will be
        # flagged as false positives (or negatives) and use them to reduce the model accuracy
        # All the rare words will be encoded with the following value.
        dummy_word = "0xffffffff"
        dummy_word_index = -1  # this is the index of the dummy word (to be set later)

        # this is to be used to spoof the index of the dummy word and thus force false positive
        # determination during testing (since these words cannot be predicted)
        dummy_index = -1

        # Set total rows to None to load all the rows for online learning. Else keep as many as we need.
        total_rows = scenario['number_of_rows_to_model'] if not online_retraining else \
            scenario['number_of_rows_to_model'] + online_retraining_period_size * (online_retraining_periods + 2)
        print("Running for %d" % total_rows)

        if pretrain_type is None:
            dataset_verbose = pd.read_csv(TRACE_DIRECTORY + scenario['trace_file_name'], sep=" ", nrows=total_rows,
                                          skiprows=scenario['number_of_rows_to_skip'])
        else:
            ### This implements testing with a pretrained model
            # TODO: put this info about the rerun files at the begining of the script as a dictionary.
            comparison_file_name = scenario['trace_file_name']
            if pretrain_type == "rerun":
                comparison_file_name = comparison_file_name.replace("_1M.out", "") + "_repeat_1M.out"
            if pretrain_type == "new":
                comparison_file_name = comparison_file_name.replace("1_", "2_")
            if pretrain_type == "diff":
                return "diff"
                # Pem1020
                # comparison_file_name = traces_test[0]
            ###
            dataset_verbose1 = pd.read_csv(TRACE_DIRECTORY + scenario['trace_file_name'], sep=" ",
                                           nrows=int(math.floor(total_rows / 2.0)),
                                           skiprows=scenario['number_of_rows_to_skip'])
            dataset_verbose1.columns = ["instruction", "type", "address"]

            dataset_verbose2 = pd.read_csv(TRACE_DIRECTORY + comparison_file_name, sep=" ",
                                           nrows=int(math.ceil(total_rows / 2.0)),
                                           skiprows=scenario['number_of_rows_to_skip'] + int(
                                               math.floor(total_rows / 2.0)))
            dataset_verbose2.columns = ["instruction", "type", "address"]

            dataset_verbose = dataset_verbose1.append(dataset_verbose2, ignore_index=True)

        dataset_verbose.columns = ["instruction", "type", "address"]
        if keep_read_access_only:
            dataset_verbose = dataset_verbose[dataset_verbose["type"] == "R"]
        # print dataset_verbose.head()
        # print dataset_verbose.describe()

        dataset = dataset_verbose['address']
        del dataset_verbose

        if not use_manual_encoding:  # use keras
            if model_diffs:
                print("Tokenizing ...")
                # Tokenize raw memory address to convert them to integers
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts(list(dataset))
                concat_dataset = [' '.join(list(dataset))]
                # This is used only for plotting purposes
                encoded_raw = tokenizer.texts_to_sequences(concat_dataset)[0]

                vocab_size_raw = len(tokenizer.word_index) + 1
                print('Raw Vocabulary Size: %d' % vocab_size_raw)

                # calculate diffs of integer memory addresses
                # print dataset

                encoded_raw_diff = difference16(data=list(dataset), lag=1, prune_lsb=prune_lsb,
                                                prune_length=prune_length)

                encoded_raw_diff_str = ["%s%d" % ("1x" if x < 0 else "0x", abs(x)) for x in encoded_raw_diff]
                df = pd.DataFrame(encoded_raw_diff_str)
                # print df

                df.columns = ['delta']

                df2 = pd.DataFrame(pd.Series(encoded_raw_diff_str).value_counts())
                # print "Length2 %s" % len(df2.index)
                df2.columns = ['total']
                df2['delta'] = df2.index
                # print "xxxxx", df2
                df2 = df2.reset_index(drop=True)
                df2.columns = ['total', 'delta']
                # print "V2", df2
                bit_size_offset = 3

                df2_2 = df2[['total']].copy()
                df2_2['cumsum'] = df2_2.sort_values(by="total", ascending=False)[['total']].cumsum(axis=0)

                tmp_total_rows = df2['total'].sum()

                # Get the row index where the cumulative quantity reaches half the total.
                # print "AAA44", tmp_total_rows, df2_2.head()
                df2_3 = df2_2[
                    df2_2['cumsum'] < vocabulary_mimimum_word_frequency_quantile * tmp_total_rows]  # .idxmax()
                # print "AAA34", myindex, tmp_total_rows
                # print df2_2.head()

                # Get the price at that index
                vocabulary_mimimum_word_frequency = df2_3.loc[
                    df2_3['cumsum'].idxmax(), 'total']  # int(list(df2_2['total'].iloc[myindex])[0])

                if vocabulary_mimimum_word_frequency == 1:
                    vocabulary_mimimum_word_frequency = 0  # since 1 is going to prune a lot of words
                print("Quantile based Minimum Frequency for %s is %s" % (
                    vocabulary_mimimum_word_frequency_quantile, vocabulary_mimimum_word_frequency))
                # df2 = df2[df2['total'] > vocabulary_mimimum_word_frequency]
                # print "xxxxxxx", df2.head()

                # print "xxxxx", df2.head()

                if vocabulary_maximum_size and not convert_output_to_binary:
                    df2 = df2[(df2.index > vocabulary_maximum_size) | (
                            df2['total'] < vocabulary_mimimum_word_frequency)]  # TODO: make it <=
                else:
                    df2 = df2[(df2.index > math.pow(2, bit_size) - bit_size_offset) | (df2[
                                                                                           'total'] < vocabulary_mimimum_word_frequency)]  # we subtract words to allow for the dummy word to be also stored.

                # print "xxxxx", df2.head()

                # vocabulary_mimimum_word_frequency = np.quantile(encode_mem_accesses(list(encoded_raw_diff_str)), vocabulary_mimimum_word_frequency_quantile) # angelos version  , vocabulary_mimimum_word_frequency_quantile)

                # Set a dummy value to represent the ignored deltas. This will be converted later to a unique integer using a second tokenizer.
                # print "Length2 %s" % len(df2.index)
                df.loc[df.delta.isin(df2.delta), ['delta']] = dummy_word

                # print "xxxxxx", df[df['delta'] == dummy_word]
                # print "Length1 %s" % len(df.index)

                # print df.head()
                # print df[df['delta'] == dummy_word]
                # df.describe()
                encoded_raw_diff_pruned = df['delta']
                # print "AAA3", encoded_raw_diff_pruned.head()
                # print "V3"
                # print encoded_raw_diff_pruned.head()

                # print "pruned", encoded_raw_diff_pruned[:300]
                del df, df2

                # Calclulate accuracy reduction due to vocabulary pruning.
                tmp_train, tmp_test = train_test_split(encoded_raw_diff_pruned, test_size=test_ratio, shuffle=False)
                total_removals = Counter(encoded_raw_diff_pruned)[dummy_word]
                total_rows = len(encoded_raw_diff_pruned)
                train_removals = Counter(tmp_train)[dummy_word]
                train_total = len(tmp_train)
                test_removals = Counter(tmp_test)[dummy_word]
                test_total = len(tmp_test)
                print(total_removals, total_rows, train_removals, train_total, test_removals, test_total)
                max_test_accuracy = 1 - test_removals / test_total
                print("Max Accuracy: %s" % max_test_accuracy)
                print("Total Removals: %s" % total_removals)

                # Tokenize again the pruned differentials to produce unique vocabulary (classes)
                encoded_raw_diff_pruned_str = [str(x) for x in list(encoded_raw_diff_pruned)]
                tokenizer2 = Tokenizer()
                tokenizer2.fit_on_texts(encoded_raw_diff_pruned_str)
                encoded_final = tokenizer2.texts_to_sequences([' '.join(encoded_raw_diff_pruned_str)])[0]
                final_vocab_size = len(tokenizer2.word_index) + 1
                print('Pruned Vocabulary Size: %d' % final_vocab_size)

                for word, index in tokenizer2.word_index.items():
                    if word == dummy_word:
                        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", word, index)
                        dummy_word_index = index
                        break
            #             set_plot_size(6,6)
            #             _ = plt.hist(encoded_final, bins=100)
            #             _ = plt.grid(True)
            #             _ = plt.title("Histogram Of All Memory Deltas After Prunning\nFor The %s App" % app_name)
            #             _ = plt.show()

            else:
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts(list(dataset))
                encoded_final = tokenizer.texts_to_sequences([' '.join(list(dataset))])[0]
                final_vocab_size = len(tokenizer.word_index) + 1
        else:
            # TODO: move this in the diff section or remove completely.
            return 0

        if decompose_timeseries:
            return 1

        # The series below are for visulaization purposes only.
        if plot_timeseries:
            if model_diffs:
                # encoded_raw_train, encoded_raw_test = train_test_split(encoded_raw, test_size=test_ratio, shuffle=False)
                encoded_raw_diff_train, encoded_raw_diff_test = train_test_split(encoded_raw_diff, test_size=test_ratio,
                                                                                 shuffle=False)
                # plot_memory_trace(train_data=encoded_raw_train, test_data=encoded_raw_test, rows=200000, app_name=app_name, scenario_name=scenario_name+" Raw")
                plot_memory_trace(train_data=encoded_raw_diff_train, test_data=encoded_raw_diff_test, rows=200000,
                                  app_name=app_name, scenario_name=scenario_name + " Raw Diff")

            encoded_train, encoded_test = train_test_split(encoded_final, test_size=test_ratio, shuffle=False)
            plot_memory_trace(train_data=encoded_train, test_data=encoded_test, rows=2000, app_name=app_name,
                              scenario_name=scenario_name + " Final")

        sequences = create_windowed_dataset(encoded_final, look_back)

        print('Final Vocabulary Size: %d' % final_vocab_size)
        print('Total Sequences: %d' % len(sequences))

        # Pad the sequences to the same length (is not really needed here since we have fixed input windows).
        # See documentation in the source code here: https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py
        max_length = max([len(seq) for seq in sequences])
        sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

        # print encoded_final, sequences, final_vocab_size

        return encoded_final, sequences, final_vocab_size, tokenizer, tokenizer2, max_test_accuracy, max_length, dummy_word, dummy_word_index, dummy_index, vocab_size_raw, dataset

    def run_lstm_model(scenario=None):
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
            encoded_final, sequences, final_vocab_size, tokenizer, tokenizer2, max_test_accuracy, max_length, dummy_word, \
            dummy_word_index, dummy_index, vocab_size_raw, dataset = dataset_creator(scenario)
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
            SVG(model_to_dot(model, show_shapes=True, show_layer_names=False).create(prog='dot', format='svg'))
            plot_model(model, to_file=NOTEBOOK_PLOTS_DIRECTORY + 'model_for_%s.png' % scenario_name, show_shapes=True,
                       show_layer_names=False)
            # =====================================================================================================

            # =====================================================================================================
            # Model training/testing
            # =====================================================================================================
            X, y = sequences[:, :-1], sequences[:, -1]
            # print X.shape, y.shape
            # print "A11", y
            # y = y.reshape((16, len(y)))

            # print "A7", sequences
            # Vectorize the output y (one hot encoding)
            # encode_inputs
            if convert_output_to_binary:
                # print y
                y = convert_to_binary(data=y, bit_size=bit_size)  # converts diffs to 16 bit representation
                # print X[:5], y
                if encode_inputs:
                    X = convert_to_binary(data=X, bit_size=bit_size,
                                          multidimensional_input=True)  # converts diffs to 16 bit representation
                # print X[:5], y

            else:
                y = to_categorical(y, num_classes=final_vocab_size)
            # print y
            # y = np.array([np.array(tmp_y) for tmp_y in y])
            # y = y.reshape((y.shape[0], 16))
            # print X.shape, y.shape
            # print "A8", y.reshape(1, -1)

            if on_the_fly_testing:
                return 0
            else:
                X_train, X_test = train_test_split(X, test_size=test_ratio, shuffle=False)
                y_train, y_test = train_test_split(y, test_size=test_ratio, shuffle=False)
                # new@05.2, useful?
                y_train_raw, y_test_raw = train_test_split(dataset, test_size=test_ratio, shuffle=False)

                # print "A2", y_train, y_test, y
                # print X, X_train, X_test
                # print y, y_train, y_test

                # =====================================================================================================
                # IMPORTANT: The code below modifies the dummy word mappings to be forcing a false positive to be counted.
                if max_test_accuracy < 1:
                    print("Overwritting Ignored Words...")
                    print(dummy_word_index)
                    if convert_output_to_binary:
                        # print "AA2", y_test
                        # for el in y_test:
                        # print "AA3", el
                        # print convert_to_binary(data=[dummy_word_index], bit_size=bit_size)
                        # break
                        # print pd.DataFrame(y_test).describe()
                        # y_test = np.array([[0 for tmp2 in tmp1] if all(tmp1 == convert_to_binary(data=[dummy_word_index], bit_size=bit_size)[0]) else tmp1 for tmp1 in y_test])
                        y_test = np.array([[0 for tmp2 in tmp1] if np.array_equal(tmp1,
                                                                                  convert_to_binary(
                                                                                      data=[dummy_word_index],
                                                                                      bit_size=bit_size)[0]) else tmp1
                                           for tmp1 in y_test])

                    else:
                        y_test = np.array(
                            [[0 for tmp2 in tmp1] if argmax(tmp1) == dummy_word_index else tmp1 for tmp1 in y_test])
                    print("Overwritting Ignored Words Completted")
                # =====================================================================================================
                # print X_train, y_train
                model_file_name = NOTEBOOK_PICKLES_DIRECTORY + "%s_train_test_split_%s.h5" % (scenario_name, unique_key)

                if load_existing_pickles and os.path.isfile(model_file_name):
                    model = load_model(model_file_name)
                    train_history = None
                    train_accuracy = -1  # train_history.history['acc'][-1]
                else:
                    train_history = model.fit(X_train,
                                              y_train,
                                              epochs=lstm_epochs,
                                              verbose=verbosity,
                                              shuffle=False,
                                              batch_size=lstm_batch_size)

                    model.save(model_file_name)
                    train_accuracy = train_history.history['acc'][-1]

                if convert_output_to_binary:
                    y_pred = model.predict(X_test)

                    # np.savetxt('y_test1.txt', y_test, delimiter=',')
                    # np.savetxt('y_pred1.txt', y_pred, delimiter=',')

                    y_pred[y_pred >= 0.5] = 1
                    y_pred[y_pred < 0.5] = 0

                    # np.savetxt('y_test2.txt', y_test, delimiter=',')
                    # np.savetxt('y_pred2.txt', y_pred, delimiter=',')

                    aaaaa = np.packbits(np.array(y_test, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
                    bbbbb = np.packbits(np.array(y_pred, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
                    # print "ANGELOS", scenario_name
                    np.savetxt('%s/y_test_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name), aaaaa, delimiter=',',
                               fmt='%10.5f')
                    np.savetxt('%s/y_pred_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name), bbbbb, delimiter=',',
                               fmt='%10.5f')

                    # ================================================================================================================
                    # Reverse transforms
                    reverse_word_map = dict(map(reversed, tokenizer2.word_index.items()))

                    # Function takes a tokenized sentence and returns the words
                    def sequence_to_text(list_of_indices):
                        # Looking up words in dictionary
                        words = [reverse_word_map.get(letter) for letter in list_of_indices]
                        return (words)

                    # Creating texts
                    original_testing_diffs = list(map(sequence_to_text, [aaaaa]))
                    original_predictions_diffs = list(map(sequence_to_text, [bbbbb]))
                    # print original_testing_diffs[0][:100]

                    np.savetxt('%s/y_test_actual_mem2_%s_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name, unique_key),
                               np.array(original_testing_diffs), delimiter=',\n', fmt='%s')
                    np.savetxt(
                        '%s/y_test_predicted_mem2_%s_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name, unique_key),
                        np.array(original_predictions_diffs), delimiter=',\n', fmt='%s')

                    # print(sum(xxx is not None for xxx in original_testing_diffs))
                    # return 1

                    # original_testing_diffs = [(-1 if int(k[0]) == 1 else 1)*int(k[2:]) for k in original_testing_diffs[0] if k is not None]
                    # original_predictions_diffs = [(-1 if int(k[0]) == 1 else 1)*int(k[2:]) for k in original_predictions_diffs[0] if k is not None]

                    # a = [((-1 if int(k[0]) == 1 else 1)*int(k[2:]), (-1 if int(l[0]) == 1 else 1)*int(l[2:])) for k,l in zip(original_testing_diffs, original_predictions_diffs) if l is not None and k is not None]
                    # print a[:100]
                    # return 1
                    tmp = [((-1 if int(k[0]) == 1 else 1) * int(k[2:]), (-1 if int(l[0]) == 1 else 1) * int(
                        l[2:])) if l is not None and k is not None and l != dummy_word and k != dummy_word else (
                    None, None) for k, l in zip(original_testing_diffs[0], original_predictions_diffs[0])]
                    original_testing_diffs, original_predictions_diffs = zip(*tmp)

                    # print original_testing_diffs[:100]
                    # print list(y_test_raw)[:101]

                    # print difference16(data=list(y_test_raw), lag=1, prune_lsb=False, prune_length=0)[:101]
                    i = 0
                    actual_memory_address = []
                    predicted_memory_address = []
                    tmp = list(y_test_raw)
                    for act, pred in zip(original_testing_diffs, original_predictions_diffs):
                        # if i%10000 == 0:
                        #    print i
                        # if i < 5:
                        #    #print i, int(list(y_test_raw)[i+1], 16), act, int(list(y_test_raw)[i+1], 16) + act
                        #    print hex(int(list(y_test_raw)[i+1], 16) + act), hex(int(list(y_test_raw)[i+1], 16) + pred)
                        actual_memory_address.append(hex(int(tmp[i + 1], 16) + act) if act is not None else "-1")
                        predicted_memory_address.append(hex(int(tmp[i + 1], 16) + pred) if pred is not None else "-1")
                        i += 1

                    np.savetxt('%s/y_test_actual_mem_%s_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name, unique_key),
                               np.array(actual_memory_address), delimiter=',', fmt='%s')
                    np.savetxt(
                        '%s/y_test_predicted_mem_%s_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name, unique_key),
                        np.array(predicted_memory_address), delimiter=',', fmt='%s')

                    # print actual_memory_address[:100]
                    # print predicted_memory_address[:100]
                    # ================================================================================================================

                    # accuracy = accuracy_score(np.array(actual_memory_address), np.array(predicted_memory_address))
                    accuracy = accuracy_score(np.array(y_test), np.array(y_pred))

                    test_history = [0, accuracy]  # for backwards compatibility

                    misc_stats['execution_time'] = time.time() - start_time
                    misc_stats['vocab_size_raw'] = vocab_size_raw
                    misc_stats['final_vocab_size'] = final_vocab_size
                    misc_stats['params'] = model.count_params()
                    misc_stats['max_test_accuracy'] = max_test_accuracy

                    np.savetxt('%s/accuracy_%s_%s.txt' % (NOTEBOOK_REPORT_DIRECTORY, scenario_name, unique_key),
                               np.array([accuracy]), delimiter=',', fmt='%10.5f')
                    plot_train_test_model_performance(train_history, test_history, app_name=app_name,
                                                      scenario_name=scenario_name)

                    print("Train Accuracy %f, Test Accuracy %f" % (train_accuracy, accuracy))

                    '''
                    #================================================================================================================
                    ### PREFETCHER IMPLEMENTATION
                    misc_stats['cache_stats'] = []

                    actual_memory_address_bin = [str('{0:048b}'.format(int(str(d), 16))) for d in actual_memory_address]
                    predicted_memory_address_bin = [str('{0:048b}'.format(int(str(d), 16))) for d in predicted_memory_address]

                    np.savetxt('%s/y_test_actual_mem_bin_%s_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name, unique_key), np.array(actual_memory_address_bin), delimiter=',', fmt='%s')
                    np.savetxt('%s/y_test_prediction_mem_bin_%s_%s.txt' % (NOTEBOOK_DATA_DIRECTORY, scenario_name, unique_key), np.array(predicted_memory_address_bin), delimiter=',', fmt='%s')

                    # about cache
                    for cache_size in CACHE_SIZES:
                        for cache_block_size in CACHE_BLOCK_SIZES:
                            for cache_replacement_algo in CACHE_REPLACEMENT_ALGOS:
                                print ("Running Prefetcher With Size %s, block sizs %s and replacement %s" 
                                       % (cache_size, cache_block_size, cache_replacement_algo))

                                regular_cache = OrderedDict()    
                                cache_with_prefetching = OrderedDict()    

                                # cache hits counter 
                                regular_cache_hits = 0
                                cache_with_prefetching_hits = 0
                                useful_prefetches = 0

                                next_access_pruned = predicted_memory_address_bin[0][:-int(np.log2(cache_block_size))]  # first time prefetch
                                total_accesses = len(actual_memory_address_bin[:-1])

                                for index, datum in enumerate(actual_memory_address_bin[:-1]):
                                    if datum == -1:
                                        continue
                                    else:
                                        if index % 50000 == 0:
                                            print ("Progress %.2f %%" % (100*index/len(actual_memory_address_bin)))

                                        # The pruning below implements the locality Block of the cache, such that we don't store explicitly all the variations. 
                                        if cache_block_size > 1:
                                            datum_pruned = datum[:-int(np.log2(cache_block_size))]
                                        else:
                                            datum_pruned = datum

                                        if index % 50000 == 0:
                                            print ("Data %s - %s, %s" % (datum, datum_pruned, -int(np.log2(cache_block_size))))
                                        #print datum, datum_pruned
                                        ## REGULAR CACHE
                                        #Cache hit case
                                        if datum_pruned in regular_cache:
                                            regular_cache_hits += 1
                                            del regular_cache[datum_pruned]
                                            regular_cache[datum_pruned] = datum_pruned 
                                        else:
                                            # Cache miss case
                                            if len(regular_cache) >= cache_size:
                                                # LRU replacement policy
                                                if cache_replacement_algo == "LRU":
                                                    to_be_replaced = regular_cache.keys()[0]  # remove the first element of the ordered dict, which is the one that appears to be the oldest
                                                else: # random
                                                    # Random replacement policy
                                                    to_be_replaced = regular_cache.keys()[random.randint(0, len(regular_cache)-1)]  # pick a random element among all the keys. 

                                                # By deleting and adding back again an element, we refresh it's "last used" time, thus implementing LRU. We don't need it strictly speaking for "Random" policy but we leave it here for simplicity. 
                                                #if datum_pruned in regular_cache:
                                                #    del regular_cache[datum_pruned]
                                                #regular_cache[datum_pruned] = datum_pruned                            
                                                del regular_cache[to_be_replaced]
                                                regular_cache[datum_pruned] = datum_pruned  
                                            else:
                                                # if the cache is not full yet, simply add the last used element. 
                                                regular_cache[datum_pruned] = datum_pruned

                                        ## PREFETCHING CACHE

                                        #Cache hit case
                                        if datum_pruned in cache_with_prefetching:
                                            # cache hit
                                            cache_with_prefetching_hits += 1
                                            del cache_with_prefetching[datum_pruned]
                                            cache_with_prefetching[datum_pruned] = datum_pruned 

                                            if datum_pruned == next_access_pruned: 
                                                # if the fetched data are already in the cache, give credit too. 
                                                useful_prefetches += 1
                                        else:
                                            # Cache miss case
                                            if len(cache_with_prefetching) >= cache_size:
                                                # LRU replacement policy
                                                if cache_replacement_algo == "LRU":
                                                    to_be_replaced = cache_with_prefetching.keys()[0]  # remove the first element of the ordered dict, which is the one that appears to be the oldest
                                                else: # random
                                                    # Random replacement policy
                                                    to_be_replaced = cache_with_prefetching.keys()[random.randint(0, len(cache_with_prefetching)-1)]  # pick a random element among all the keys. 

                                                # By deleting and adding back again an element, we refresh it's "last used" time, thus implementing LRU. We don't need it strictly speaking for "Random" policy but we leave it here for simplicity. 
                                                #if datum_pruned in cache_with_prefetching:
                                                #    del cache_with_prefetching[datum_pruned]
                                                #cache_with_prefetching[datum_pruned] = datum_pruned                            
                                                del cache_with_prefetching[to_be_replaced]
                                                cache_with_prefetching[datum_pruned] = datum_pruned
                                            else:
                                                # if the cache is not full yet, simply add the last used element. 
                                                cache_with_prefetching[datum_pruned] = datum_pruned

                                            if datum_pruned == next_access_pruned:
                                                # if we had the entry in the prefetch buffer, then we count it as a hit
                                                # the entry has been stored already in the cache from the else statement above. 
                                                cache_with_prefetching_hits += 1
                                                useful_prefetches += 1
                                    # prefetch next access
                                    try:
                                        if cache_block_size > 1:
                                            next_access_pruned = predicted_memory_address_bin[index+1][:-int(np.log2(cache_block_size))]
                                        else: 
                                            next_access_pruned = predicted_memory_address_bin[index+1]
                                    except:
                                        print (index, predicted_memory_address_bin[:100])
                                        raise
                                regular_cache_misses = total_accesses - regular_cache_hits
                                cache_with_prefetching_misses = total_accesses - cache_with_prefetching_hits

                                misc_stats['cache_stats'].append({
                                    'cache_size': cache_size,
                                    "cache_block_size": cache_block_size,
                                    "cache_replacement_algo": cache_replacement_algo,
                                    "regular_cache_hit_rate": regular_cache_hits/total_accesses,
                                    "cache_with_prefetching_hit_rate": cache_with_prefetching_hits/total_accesses,
                                    "prefetching_coverage": ((regular_cache_misses-cache_with_prefetching_misses)/regular_cache_misses),
                                    "prefetching_accuracy": ((regular_cache_misses-cache_with_prefetching_misses)/((regular_cache_misses-cache_with_prefetching_misses) + (total_accesses - useful_prefetches))),
                                    #"prefetching_accuracy_conservative": ((regular_cache_misses-cache_with_prefetching_misses)/(cache_with_prefetching_misses)
                                })


                                print ("STATS\n------")
                                print ("Cache Size: %s" % cache_size)
                                print ("Block Size: %s" % cache_block_size)
                                print ("Replacement Algo: %s" % cache_replacement_algo)
                                print ("Hits", regular_cache_hits, cache_with_prefetching_hits)
                                print ("Regular Cache Hit Rate: %f" % (regular_cache_hits/total_accesses))
                                print ("Prefetching Cache Hit Rate: %f" % (cache_with_prefetching_hits/total_accesses))
                                print ("Coverage %.3f" % ((regular_cache_misses-cache_with_prefetching_misses)/regular_cache_misses))
                                print ("Accuracy %.3f" % ((regular_cache_misses-cache_with_prefetching_misses)/((regular_cache_misses-cache_with_prefetching_misses) + (total_accesses - useful_prefetches))))
                                #print "Accuracy2 %.3f" % ((regular_cache_misses-cache_with_prefetching_misses)/(total_accesses))



                    ####
                    '''
                    return train_accuracy, accuracy, misc_stats
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
        'dedup_1_1M.out': 100000,
        'facesim_1_1M.out': 100000,
        'freqmine_1_1M.out': 100000,
        'raytrace_1_1M.out': 100000,
        'streamcluster_1_1M.out': 100000,
        'vips_1_1M.out': 100000,
        'x264_1_1M.out': 100000
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
            train_history, test_history, misc_stats = run_lstm_model(scenario)
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
    for trace in TRACE_FILE_NAMES:
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
                        'load_existing_pickles': True,
                        'skip_run_if_previous_stats_found': True,

                        # dataset params
                        'keep_read_access_only': False,
                        'number_of_rows_to_model': 400000,
                        'number_of_rows_to_skip': trace_offsets[trace],  # int(TRACE_FILE_NAME_SIZES[trace]*3.0/5.0),
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
                        'lstm_batch_size': 256 if "fpga" in model_type else 256,
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

for trace in TRACE_FILE_NAMES:
    run_dc_lstm([trace],20)