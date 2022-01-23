# -*- coding: utf-8 -*-
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from itertools import compress
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

class Dataset:
    def __init__(self, name=None):
        if not name is None:
            self.name = name
    def setup_data(self, X, y, split_test=0.3, seed=1, scaler=None, stratify=None):
        # scaler is assumed to be one of the 'sklearn.preprocessing'-scalers
        # consistency checks
        tx = type(X)
        ty = type(y)
        if tx != pd.core.frame.DataFrame or ty != pd.core.frame.DataFrame:
            raise Exception('X and y must be of type pandas.core.frame.DataFrame')
        if split_test > 0:
        # train test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = split_test, shuffle=True, random_state=seed, stratify=stratify)
            nms = self.X_train.columns.copy(deep=True)
            if not (nms == self.X_test.columns).all():
                raise Exception('column names of train and test data disagree')
        else:
            self.X_train = X
            self.y_train = y
        if not scaler is None:
            self.scaler = scaler

    def encode_categories(self, variables=None, onehot=False):
        if hasattr(self, 'encoder'):
            print('replacing existing encoder ...')
            
        len_train = len(self.X_train)
        dat = self.X_train.copy()
        dat = dat.join(self.y_train)
        if hasattr(self, 'X_test'):
            dat = dat.append(self.X_test.join(self.y_test))
        if onehot:
            enc=OneHotEncoder(dtype=int, sparse=False) #returns a sparse matrix if sparse is True
        else:
            enc = OrdinalEncoder(dtype=int)
        if variables is None:
            # if column names (variables) are not supplied, encode everything that has a string at first position
            tmp = [type(dat.iloc[0,er]) is str for er in range(dat.shape[1])] # get types of features
            tmp = dat.columns[tmp] # for consistency, tmp is converted to column index
        elif variables=='all_x': # select all X-columns for encoding
            tmp = self.X_train.columns
        elif variables=='all_y': # select all y-columns for encoding
            tmp = self.y_train.columns
        elif variables=='all': # select all columns for encoding
            tmp = dat.columns
        else:
            tmp = variables
        enc.fit(dat.loc[:,tmp]) # fit encoder
        if onehot: # transform X or y directly
            if variables=='all_x':
                X = enc.transform(dat.loc[:,self.X_train.columns])
                y = dat.loc[:,self.y_train.columns]
            elif variables=='all_y':
                X = dat.loc[:,self.X_train.columns]
                y = enc.transform(dat.loc[:,self.y_train.columns])
            elif variables=='all':
                enc.fit(dat.loc[:,self.X_train.columns])
                X = enc.transform(dat.loc[:,self.X_train.columns])
                enc.fit(dat.loc[:,self.y_train.columns])
                y = enc.transform(dat.loc[:,self.y_train.columns])
            else: # encode some individual features / responses
                xs = list(compress(tmp, [x in self.X_train.columns for x in tmp]))
                ys = list(compress(tmp, [x in self.y_train.columns for x in tmp]))
                if any(xs):
                    enc.fit(dat.loc[:,xs])
                    X = pd.DataFrame(enc.transform(dat.loc[:, xs]))
                    X.columns = enc.get_feature_names(input_features=xs)
                    X = dat.drop(columns=xs).drop(columns=self.y_train.columns).join(X)
                else:
                    X = dat.loc[:,self.X_train.columns]
                if any(ys):
                    enc.fit(dat.loc[:,ys])
                    y = pd.DataFrame(enc.transform(dat.loc[:, ys]))
                    y.columns = enc.get_feature_names(input_features=ys)
                    y = dat.drop(columns=ys).drop(columns=self.X_train.columns).join(y)
                else:
                    y = dat.loc[:,self.y_train.columns]
            # make sure that the output is still a pandas data frame
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)
        else: # transform requested columns
            dat.loc[:,tmp] = enc.transform(dat.loc[:,tmp])
            # reconstruct X and y
            X = dat.drop(columns=self.y_train.columns)
            y = dat.loc[:,self.y_train.columns]            
        self.X_train = X[:len_train]
        self.y_train = y[:len_train]
        if hasattr(self, 'X_test'):
            self.X_test  = X[len_train:]
            self.y_test  = y[len_train:]
        self.encoder = enc
    def scale(self):
        if not hasattr(self, 'scaler'):
            raise Exception('cannot scale data since no scaler is defined')
        X_num = self.X_train.select_dtypes(include='number')
        nms = X_num.columns.copy(deep=True)
        indx = X_num.index.copy(deep=True)
        self.scaler.fit(self.X_train.loc[:,nms])
        self.X_train.loc[:,nms] = self.scaler.transform(self.X_train.loc[:,nms])
        #X_num = pd.DataFrame(self.scaler.transform(X_num))
        #X_num.index = indx
        #X_num.columns = nms
        #self.X_train = X_num.join(self.X_train.select_dtypes(exclude='number'))
        
        #indx = self.X_test.index.copy(deep=True)
        self.X_test.loc[:,nms] = self.scaler.transform(self.X_test.loc[:,nms])
        #X_num_test.columns = nms
        #X_num_test.index = indx
        #self.X_test = X_num_test.join(self.X_test.select_dtypes(exclude='number'))
    def add_model(self, model, modID=None):
        # adds a (e.g. fitted) model instance to the data set
        if not hasattr(self, 'model_collection'):
            self.model_collection = {}
            if modID is None: modID = '0'
        if modID is None:
            modID = str(np.int32(max(self.model_collection.keys()))+1)
        self.model_collection[modID] = model
    def permutation_importance(self, metric, modIDs = 'all', n_repeats=3, num_to_str=False):
        if hasattr(self, 'X_test'):
            X_perm = self.X_test
            y_perm = self.y_test
        else:
            X_perm = self.X_train
            y_perm = self.y_train
        X_tmp = X_perm.copy(deep=True)
        if not hasattr(self, 'model_collection'):
            raise Exception('cannot calculate permutation importance since no\
                            models are defined')
        result = {}
        for ky in self.model_collection.keys():
            if modIDs == 'all' or ky in modIDs:
                y_curr = self.model_collection[ky].predict(X_perm)
                metric.update_state(y_perm, y_curr)
                m_base = metric.result().numpy()
                if num_to_str: m_base = str(m_base)
                result[ky] = {'m_base':m_base}
                result[ky]['perm'] = pd.DataFrame(index=range(n_repeats), 
                                                  columns=X_perm.columns)
                for col in range(X_perm.shape[1]):
                    for rep in range(n_repeats):
                        X_tmp.iloc[:,col] = np.random.permutation(X_perm.iloc[:,col])
                        y_curr = self.model_collection[ky].predict(X_tmp)
                        metric.reset_states()
                        metric.update_state(y_perm, y_curr)
                        rstmp = metric.result().numpy()
                        if num_to_str: rstmp = str(rstmp)
                        result[ky]['perm'].iloc[rep,col] = rstmp
                    X_tmp.iloc[:,col] = X_perm.iloc[:,col]
                result[ky] = result[ky]['perm'] - result[ky]['m_base']
        self.perm_imp = result

        
class StandardScalerOverall:
    def __init__(self, name=None):
        if not name is None:
            self.name = name
    def fit(self, data):
        if type(data) != pd.core.frame.DataFrame or type(data) != pd.core.frame.DataFrame:
            raise Exception('data must be of type pandas.core.frame.DataFrame')
        self.mn = data.to_numpy().flatten().mean()
        self.sd = data.to_numpy().flatten().std()
        
    def transform(self, data):
        if type(data) != pd.core.frame.DataFrame or type(data) != pd.core.frame.DataFrame:
            raise Exception('data must be of type pandas.core.frame.DataFrame')
        data_new = (data - self.mn) / self.sd
        return data_new

        
def plot_history(history, file=None, logy=False):
    # This function was copied from https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search (28.07.2020)
    metrc = [i for i in history.history.keys()][1]
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if metrc in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if metrc in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    fig, (ax1, ax2) = plt.subplots(2)
    for l in loss_list:
        ax1.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        ax1.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    ax1.set(xlabel='Epochs')
    ax1.set(ylabel='Loss')
    ax1.set_xscale('log')
    if logy:
        ax1.set_yscale('log')
    ax1.legend()
        
    ## Accuracy
    for l in acc_list:
        ax2.plot(epochs, history.history[l], 'b', label='Training ' + metrc + ' (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        ax2.plot(epochs, history.history[l], 'g', label='Validation ' + metrc + ' (' + str(format(history.history[l][-1],'.5f'))+')')

    ax2.set(xlabel='Epochs')
    ax2.set(ylabel=metrc)
    ax2.set_xscale('log')
    if logy:
        ax2.set_yscale('log')
    ax2.legend()

    if file is None:
        plt.show()
    else:
        plt.savefig(file)
    plt.close()

def splt_str(x, len_na=1):
    '''Split a string into a list of characters
    Inputs:
        - x (string): String to be split
        - len_na (int): if x is NA, a list of NAs of length len_na is returned
    Outputs:
        - A list of characters'''
    if pd.isna(x):
        return ['NA'] * len_na
    else:
        return [cr for cr in x]

def make_dir(path):
    # helper function to create folders if non-existent, courtesy of Stefania Russo
    try: 
        os.mkdir(path)
    except OSError:
        print("Directory already exsists {}".format(path))
        return False
    else:
        print("Successfully created the directory {}".format(path))
        return True

def shuffle_weights(model, weights=None):
    # This code is taken from: https://gist.github.com/jkleint (06.08.2020)
    if weights is None:
        weights = model.get_weights()
    weights_new = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights_new)
    return model