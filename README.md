# MemMAP
This repo contains code for the paper, [MemMAP: Compact and Generalizable Meta-LSTM Models for Memory Access Prediction. It includes code for experiments on Specialized, Concatenated, MAML-DCLSTM, and C-MAML-DCLSTM](https://doi.org/10.1007/978-3-030-47436-2_5), cited as:
```
@inproceedings{srivastava2020memmap,
  title={MemMAP: Compact and Generalizable Meta-LSTM Models for Memory Access Prediction},
  author={Srivastava, Ajitesh and Wang, Ta-Yang and Zhang, Pengmiao and De Rose, Cesar Augusto F and Kannan, Rajgopal and Prasanna, Viktor K},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={57--68},
  year={2020},
  organization={Springer}
}
```
## Dataset 
The trace uses the PARSEC benchmark(https://parsec.cs.princeton.edu/), generated using Pin tool, see example *Memory Reference Trace* (https://software.intel.com/sites/landingpage/pintool/docs/97503/Pin/html/)

In `data` folder, two sample raw traces are given to run the scripts.

## Dependencies
* python: 3.x
* TensorFlow v1.0+
* Keras v1.0+
* Pytorch: 0.4+
* NVIDIA GPU

## Speclialized Model

The specialized model uses doubly compressed LSTM discribed in paper: 

```
@inproceedings{srivastava2019predicting,
  title={Predicting memory accesses: the road to compact ML-driven prefetcher},
  author={Srivastava, Ajitesh and Lazaris, Angelos and Brooks, Benjamin and Kannan, Rajgopal and Prasanna, Viktor K},
  booktitle={Proceedings of the International Symposium on Memory Systems},
  pages={461--470},
  year={2019}
}
```
### Train and test

First, `cd Specialized`,

Then run the script use `python3 Specialized.py bodytrack_1_1M.out 20`, where the first argument is the trace file name in folder `../data/`, the second argument is the training epochs. The length of training and testing sequences are both hardcoded as 200k.

## Cacatenated Model

`cd ./Concatenated_Rerun`

### Preprocessing

First, run `python3 concatenate_data.py [num]` to concatenate the traces, where the argument *num* is  the number of traces. 

* To run the sample traces directly, set *num=2*.

* To run all 13 applications, set *num=13*, the trace list variable *TRACE_FILE_NAMES* in "Concatenate_data.py" should be modified accordingly.

* The processed training and testing data is saved in folder `./data_combine`.

Then, run `python3 ./gen_conc_np.py [num]` to generate training set and testing set in numpy files, and tokenization in pickles.

### Model Training

```python3 ./conc_model_train.py [num] [epoch]```, where argv[1] is the number of traces and argv[2] is the training epochs.

## Meta-DCLSTM
