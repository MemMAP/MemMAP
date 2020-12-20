# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:56:18 2020

@author: pmzha
"""

import csv
import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle
import glob
'''
def convert_to_dt(source_path):
    dataset = pd.read_csv(source_path, header=None, index_col=None)

    #conver to deltas
    dataset_dt=dataset.diff()[1:]
    dataset_dt.to_csv("dt.csv", index=None)
'''
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
    
def convert_to_dt(source_path, dest_path):
    dataset_verbose = pd.read_csv(source_path, header=None, index_col=None,sep=" ")

    '''conver to deltas'''
    dataset_verbose.columns = ["instruction", "type", "address"]
    dataset = dataset_verbose['address'][:-1]
    del dataset_verbose
    encoded_raw_diff = difference16(data=list(dataset), lag=1)

    encoded_raw_diff_str = ["%s%d" % ("1x" if x < 0 else "0x", abs(x)) for x in encoded_raw_diff]
    df = pd.DataFrame(encoded_raw_diff_str)
    df.columns = ['delta']
    
    file = os.path.basename(source_path)
    filename = file.replace(".out", "_dt.csv")

    df.to_csv(dest_path+filename, index=None)
   
def create_windowed_dataset(data, look_back):
    """
    Create the dataset by grouping windows of memory accesses together (using the look_back parameter)

    data: it should be a list of integers
    """
    sequences = list()
    for i in range(look_back, len(data)):
        sequence = data[i - look_back:i + 1]
        sequences.append(sequence)
    return np.array(sequences)

def difference(dataset, interval=1):
    """
    Calculates the difference between a time-series and a lagged version of it
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def convert_to_binary(data,bit_size=16):
    #data_binary = np.array([[int(d) for i in j for d in str('{0:016b}'.format(i))] for j in list(data)])
    data_binary=np.array([[create_binary(x,bit_size) for x in line] for line in data]).reshape(-1,16*data.shape[1])
    return data_binary    


def create_binary(int_val, o_dim):
    output = [0]*o_dim
    for i in range(o_dim):
        if int_val >= 2**(o_dim-i):
            output[i] = 1
            int_val -= 2**(o_dim-i)
    return output
    
def convert_binary_to_dec(data,bit_size=16):
    #return np.reshape(np.array(list([int("".join(str(i) for i in List),2) for List in data])),(-1,1))
    dec = np.packbits(np.array(data, dtype=np.bool).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
    return dec
    # return data_dec 

def token_back(y_pred_dec,y_test_dec,tokenizer)  :
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return(words)
    dummy_word = "0xffffffff"    
    original_testing_diffs = list(map(sequence_to_text, [y_pred_dec]))
    original_predictions_diffs = list(map(sequence_to_text, [y_test_dec]))
    tmp = [((-1 if int(k[0]) == 1 else 1)*int(k[2:]), (-1 if int(l[0]) == 1 else 1)*int(l[2:])) if l is not None and k is not None and l != dummy_word and k != dummy_word else (None, None) for k,l in zip(original_testing_diffs[0], original_predictions_diffs[0])]
    original_testing_diffs, original_predictions_diffs = zip(*tmp)   
    return list(original_testing_diffs), list(original_predictions_diffs)

def Tokenize_and_Binarize(dataset,file_name,tok_save_path,np_save_path,data_len,look_back):
    '''
    function: save tokenizer and npz file of dataset X, y
    '''
    file_name = file_name.replace(".csv", "")
    
    dataset_dt_ls_x=list(dataset[0])[0:][0:data_len] #400000
    #print(dataset_dt_ls_x[0:100])
    
    '''tokenize'''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataset_dt_ls_x)
    encoded_final = tokenizer.texts_to_sequences([' '.join(dataset_dt_ls_x)])[0]
   # final_vocab_size = len(tokenizer.word_index) + 1
    # saving
    with open(tok_save_path+file_name+'_tok.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("token saved")
    
    # loading
    with open(tok_save_path+file_name+'_tok.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    '''input sequence window'''
    #look_back = 3
    sequences = create_windowed_dataset(encoded_final, look_back)
    
    '''Training data preprocessing'''
    X, y = sequences[:, :-1], sequences[:, -1]
    y = y.reshape(len(y), 1)
    print("shape before binary:", X.shape,y.shape)
    '''binay'''
    X_binary =convert_to_binary(data=X)
    y_binay=convert_to_binary(data=y)
    print("shape after binary:",X_binary.shape,y_binay.shape)
    X=X_binary
    
  #  np.savez(np_save_path+file_name+'_np.npz', X=X, y=y)
   # print("np saved")
    

    '''new data, no split'''
    #ctrl+4 block comment, +5 cancel
    #'''split'''
    #'''test'''
    test_ratio=0.34
    X_train, X_test = train_test_split(X, test_size=test_ratio, shuffle=False)
    y_train, y_test = train_test_split(y_binay, test_size=test_ratio, shuffle=False)
    
    print("X shape:",X_train.shape,X_test.shape)
    print("y shape:",y_train.shape,y_test.shape)
    np.savez(np_save_path+file_name+'_np.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("np saved")
    
    return X_train, y_train, X_test, y_test
    
    #'''train, validation'''
   # test_ratio=0.2
   # X_train, X_val = train_test_split(X_train0, test_size=test_ratio, shuffle=False)
   # y_train, y_val = train_test_split(y_train0, test_size=test_ratio, shuffle=False)
    
    #print("X shape:",X_train.shape,X_val.shape,X_test.shape)
    #print("y shape:",y_train.shape,y_val.shape,y_test.shape)

    
def Concatenate_files(source_path_list,data_len_each):
    '''
    including both train and test, for tokenizing
    input: list[files path]
    output: dataframe
    '''
    for i in range(len(source_path_list)):
        if i==0:
            dataset=pd.read_csv(source_path_list[i], header=None, index_col=None,sep=" ")[:data_len_each]
        else:
            ds=pd.read_csv(source_path_list[i], header=None, index_col=None,sep=" ")[:data_len_each]
            dataset=dataset.append(ds)
    return dataset
