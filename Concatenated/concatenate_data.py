from utils import *

PROJECT_ROOT_DIRECTORY = "./"
TRACE_DIRECTORY = "../data/"
sys.path.append(PROJECT_ROOT_DIRECTORY)
USE_GPU = False

# File Settings

DELETE_OLD_RESULTS = False

NUMBER_OF_PROCESSES = 2

PERFORM_EDA = False

NOTEBOOK_ID = "DC_LSTM_combine_data"

README_TXT = """# README
## Ensemble modeling of memory access timeseries (using shapelets, LSTMs and more)
"""
###

# Inputs:
# TRACE_DIRECTORY = PROJECT_ROOT_DIRECTORY + "data/input/"

TRACE_FILE_NAMES = [
    'blackscholes_1_1M.out',
    #'bodytrack_1_1M.out',
    'canneal_1_1M.out',
    #'dedup_1_1M.out',
    #'facesim_1_1M.out',
    #'ferret_1_1M.out',
    #'fluidanimate_1_1M.out',
    #'freqmine_1_1M.out',
    #'raytrace_1_1M.out',
    #'streamcluster_1_1M.out',
    #'swaptions_1_1M.out',
    #'vips_1_1M.out',
    #'x264_1_1M.out'
]  # more to be added here


total_rows = 400000
skip_rows = 100000


def combine_train_data(dataset_total,file2):
    dataset_verbose2 = pd.read_csv(TRACE_DIRECTORY + file2, sep=" ",
                                   nrows=int(math.floor(total_rows / 2.0)),
                                   skiprows=skip_rows)
    dataset_verbose2.columns = ["instruction", "type", "address"]
    dataset_verbose = dataset_total.append(dataset_verbose2, ignore_index=True)
    print(file2)
    return dataset_verbose


def combine_test_data(dataset_total,file2):
    dataset_verbose2 = pd.read_csv(TRACE_DIRECTORY + file2, sep=" ",
                                   nrows=int(math.floor(total_rows / 2.0)),
                                   skiprows=skip_rows+ int(
                                       math.floor(total_rows / 2.0)))
    dataset_verbose2.columns = ["instruction", "type", "address"]
    dataset_verbose = dataset_total.append(dataset_verbose2, ignore_index=True)
    print(file2)
    return dataset_verbose


def run_combine_train(num):
    file1=TRACE_FILE_NAMES[0]
    dataset_total = pd.read_csv(TRACE_DIRECTORY + file1, sep=" ",
                                nrows=int(math.floor(total_rows / 2.0)),
                                skiprows=skip_rows)
    dataset_total.columns = ["instruction", "type", "address"]
    for i in range(num-1):#12
        file2=TRACE_FILE_NAMES[i+1]
        dataset_total=combine_train_data(dataset_total,file2)
    dataset_total.to_csv(PROJECT_ROOT_DIRECTORY+"data_combine/train.out")


def run_combine_test(num):
    file1=TRACE_FILE_NAMES[0]
    dataset_total_test = pd.read_csv(TRACE_DIRECTORY + file1, sep=" ",
                                   nrows=int(math.floor(total_rows / 2.0)),
                                   skiprows=skip_rows + int(
                                           math.floor(total_rows / 2.0)))
    dataset_total_test.columns = ["instruction", "type", "address"]
    for i in range(num-1):#12
        file2=TRACE_FILE_NAMES[i+1]
        dataset_total_test=combine_train_data(dataset_total_test,file2)
    dataset_total_test.to_csv(PROJECT_ROOT_DIRECTORY+"data_combine/test.out")


def save_test_part(num):
    for i in range(num):#13
        file = TRACE_FILE_NAMES[i]
        dataset_total_test = pd.read_csv(TRACE_DIRECTORY + file, sep=" ",
                                         nrows=int(math.floor(total_rows / 2.0)),
                                         skiprows=skip_rows + int(
                                             math.floor(total_rows / 2.0)))
        dataset_total_test.columns = ["instruction", "type", "address"]
        dataset_total_test.to_csv(PROJECT_ROOT_DIRECTORY+"data_combine/test_part/"+file[0:-4]+"_test.csv")

if __name__ == "__main__":
    num=int(sys.argv[1])
    run_combine_test(num)
    run_combine_train(num)
    save_test_part(num)