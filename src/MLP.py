# -*- coding: utf-8 -*-
#%%

import helpers.DataSciPy as dsp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score, recall_score
import pandas as pd
import numpy as np
from helpers.helper_model import build_models, build_hypermodel, build_hypermodel_multiclass
from helpers.helper_dataprocessing import prepare_data

import tensorflow.keras as krs
import os
import kerastuner as kt
from kerastuner.tuners import Hyperband
from sklearn.model_selection import StratifiedKFold
#%%
# Control parameters
multiclass=False # use multiclass toxicity target variable or binary classification?

cv_fixed_models = True # fit models with fixed hyperparamters and perform cross-validation?
logreg_sklearn = False # fit a simple logistic regression model with sklearn
logreg_keras = True # fit a simple logistic regression model with keras
fit_crafted_model = True # fit a certain MLP with hand-selected hyperparameters
fit_tuned_model = True # fit the MLPs resulting from hyperparameter optimization

final_performance = True # if true, models are fitted to the
#trainval data and evaluated on the test data

hypertuning = False # perform hyperparameter tuning?
fit_base_model = False # fit single perceptron as reference?

print_best_hyper = True

setup = 'rainbow'
metrics = ['accuracy']
if multiclass:
    activation='softmax'
    loss='sparse_categorical_crossentropy'
    nout=5
    hypmod = build_hypermodel_multiclass
else:
    activation='sigmoid'
    loss='binary_crossentropy'
    nout=1
    hypmod = build_hypermodel


# Folder for results of hyperparameter tuning. In order to stay below the threshold
# path length of the windows version of the kerastuner, we us a high-level path here
fldr_hyper = 'C:/hypr_tox/'
# Folder for the rest of the output data
fldr = '../output/mlp/'

# Which features should be treated as numerical data?
numerical = ['atom_number', 'bonds_number','Mol', 'MorganDensity', 'LogP',
            'alone_atom_number', 'doubleBond', 'tripleBond', 'ring_number', 'oh_count', 'MeltingPoint', 'WaterSolubility']

#%%

# Path of the input data
path_data = '../data/processed/lc_db_processed.csv'

if final_performance:
    rstr = False
else:
    rstr = True
clbck = krs.callbacks.EarlyStopping(
    monitor="val_loss", restore_best_weights=rstr, patience=50
)

X_trainval, y_trainval, X_test, y_test, dummy = prepare_data(path_data, setup, numerical, multiclass=multiclass)

if multiclass:
    setup = 'multiclass/'+setup

#%%
## =============================================================================
## Test LogisticRegression to have 1:1 comparison to Simone's results
if cv_fixed_models:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 5645)
    accs = []
    sens = []
    specs = []
    
    if logreg_keras:
        mdls = build_models(dummy, nout=nout, activation=activation, loss=loss)
        model_logreg = mdls['model0']
    if fit_crafted_model:
        mdls = build_models(dummy, nout=nout, activation=activation, loss=loss)
        model_crafted = mdls['model5']
    
    cv = 0
    for train_index, val_index in kf.split(X=X_trainval, y=y_trainval):
        if final_performance:
            X_train = X_trainval.copy()
            X_val = X_test.copy()
            y_train = y_trainval.copy()
            y_val = y_test.copy()
            tststr = 'test/'
        else:
            X_train = X_trainval.iloc[train_index]
            X_val = X_trainval.iloc[val_index]
            y_train = y_trainval.iloc[train_index]
            y_val = y_trainval.iloc[val_index]
            tststr = ''
    
        sclr = MinMaxScaler()
        sclr.fit(X_train[numerical])
        new_train = X_train.copy()
        new_train.loc[:, numerical] = sclr.transform(X_train[numerical])
    
        new_val = X_val.copy()
        new_val.loc[:, numerical] = sclr.transform(X_val[numerical])
                
        if logreg_sklearn:
            lrc = LogisticRegression(n_jobs = -1)
            lrc.fit(new_train, y_train)
            y_pred = lrc.predict(new_val)
            y_pred_cls = np.round(y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
    
            accs.append(accuracy_score(y_val, y_pred))
            sens.append(recall_score(y_val, y_pred))
            specs.append(tn/(tn+fp))
            
        if logreg_keras:
            fldr_curr = fldr+setup+'/'+'model0/'+tststr
            os.makedirs(fldr_curr, exist_ok=True)
            model_logreg = dsp.shuffle_weights(model_logreg)
            hist = model_logreg.fit(new_train, y_train,
                                    validation_data=(new_val, y_val),
                                    batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(fldr_curr+'model0_training_hist_cv'+str(cv)+'.txt',index=False)
            model_logreg.save(fldr_curr+'model0_cv'+str(cv)+'.h5')
            dsp.plot_history(hist, file=fldr_curr+'training_hist'+str(cv)+'.pdf')
            y_pred = model_logreg.predict(new_val)
            if multiclass:
                y_pred_cls = pd.DataFrame(y_pred).idxmax(axis=1)
                rec = recall_score(y_val, y_pred_cls, average='macro')
                f1  = f1_score(y_val, y_pred_cls, average='macro')
                acc = accuracy_score(y_val, y_pred_cls)
                prc = precision_score(y_val, y_pred_cls, average='macro')
                arr = np.array([acc, rec, prc, f1])
            else:
                y_pred_cls = np.round(y_pred)
                rec = recall_score(y_val, y_pred_cls)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
                spc = tn/(tn+fp)
                f1  = f1_score(y_val, y_pred_cls)
                acc = accuracy_score(y_val, y_pred_cls)
                arr = np.array([acc, rec, spc, f1])
            with open(fldr_curr+'scores.txt', 'ab') as f:
                np.savetxt(f, arr.reshape(1,4))

        
        if fit_crafted_model:
            fldr_curr = fldr+setup+'/'+'model5/'+tststr
            os.makedirs(fldr_curr, exist_ok=True)
            model_crafted = dsp.shuffle_weights(model_crafted)
            hist = model_crafted.fit(new_train, y_train,
                                    validation_data=(new_val, y_val),
                                    batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(fldr+setup+'/' + 'model5/'+tststr+'model5_training_hist_cv'+str(cv)+'.txt',index=False)
            model_crafted.save(fldr_curr+'model5_cv'+str(cv)+'.h5')
            dsp.plot_history(hist, file=fldr_curr+'training_hist'+str(cv)+'.pdf')
            y_pred = model_crafted.predict(new_val)
            if multiclass:
                y_pred_cls = pd.DataFrame(y_pred).idxmax(axis=1)
                rec = recall_score(y_val, y_pred_cls, average='macro')
                f1  = f1_score(y_val, y_pred_cls, average='macro')
                acc = accuracy_score(y_val, y_pred_cls)
                prc = precision_score(y_val, y_pred_cls, average='macro')
                arr = np.array([acc, rec, prc, f1])
            else:
                y_pred_cls = np.round(y_pred)
                rec = recall_score(y_val, y_pred_cls)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
                spc = tn/(tn+fp)
                f1  = f1_score(y_val, y_pred_cls)
                acc = accuracy_score(y_val, y_pred_cls)
                arr = np.array([acc, rec, spc, f1])
            with open(fldr_curr+'scores.txt', 'ab') as f:
                np.savetxt(f, arr.reshape(1,4))



        if fit_tuned_model:
            fldr_curr = fldr+setup+'/'+'tuned_models/'+tststr
            os.makedirs(fldr_curr, exist_ok=True)
            # load tuner
            tuner = Hyperband(
                hypmod,
                objective=kt.Objective('val_accuracy', direction='max'),
                max_epochs=1000,
                factor=3,
                directory=os.path.normpath(fldr_hyper+'/'+setup+'/'),
                project_name='cv0')
            best_hps = tuner.get_best_hyperparameters(num_trials=1)
            model = tuner.hypermodel.build(best_hps[0])
            hist = model.fit(new_train, y_train,
                             validation_data=(new_val, y_val),
                             batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
            pd.DataFrame.from_dict(hist.history).to_csv(fldr_curr+'model_tuned_training_hist_cv'+str(cv)+'.txt',index=False)
            model.save(fldr_curr+'model_tuned_cv'+str(cv)+'.h5')
            dsp.plot_history(hist, file=fldr_curr+'model_tuned_training_hist'+str(cv)+'.pdf')
            y_pred = model.predict(new_val)
            if multiclass:
                y_pred_cls = pd.DataFrame(y_pred).idxmax(axis=1)
                rec = recall_score(y_val, y_pred_cls, average='macro')
                f1  = f1_score(y_val, y_pred_cls, average='macro')
                acc = accuracy_score(y_val, y_pred_cls)
                prc = precision_score(y_val, y_pred_cls, average='macro')
                arr = np.array([acc, rec, prc, f1])
            else:
                y_pred_cls = np.round(y_pred)
                rec = recall_score(y_val, y_pred_cls)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred_cls).ravel()
                spc = tn/(tn+fp)
                f1  = f1_score(y_val, y_pred_cls)
                acc = accuracy_score(y_val, y_pred_cls)
                arr = np.array([acc, rec, spc, f1])
            with open(fldr_curr+'scores.txt', 'ab') as f:
                np.savetxt(f, arr.reshape(1,4))

        cv = cv + 1
        

# compare to single perceptron with keras

#%%
## =============================================================================
## Fit different Neural Networks implemented in Tensorflow with tuner

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = 0
for trn_ind, val_ind in kfold.split(np.zeros(X_trainval.shape[0]), y_trainval):
    casepath = fldr+setup+'/'+'/cv'+str(cv)+'/'
    os.makedirs(casepath, exist_ok=True)
    if setup=='fathead' or setup=='rainbow':
        # don't validate, just test
        X_trn = X_trainval.copy()
        X_val = X_test.copy()
        y_trn = y_trainval.copy()
        y_val = y_test.copy()
    else:
        # training and validation set
        X_trn = X_trainval.iloc[trn_ind,:]
        X_val = X_trainval.iloc[val_ind,:]
        y_trn = y_trainval.iloc[trn_ind,:]
        y_val = y_trainval.iloc[val_ind,:]

    if fit_base_model:
        mdls = build_models(dummy)
        model = mdls['model0']

        hist = model.fit(X_trn, y_trn,
                         validation_data=(X_val, y_val),
                         batch_size=32, epochs=1000, verbose=0, callbacks=[clbck])
        pd.DataFrame.from_dict(hist.history).to_csv(casepath + 'model_base_training_hist.txt',index=False)
        model.save(casepath+'model_base.h5')
        dsp.plot_history(hist, file=casepath+'model_base_training_hist.pdf')

    if hypertuning:
        clbck_hyper = krs.callbacks.EarlyStopping(
            monitor="val_loss", restore_best_weights=True, patience=50
            )
        tuner = Hyperband(
            hypmod,
            objective=kt.Objective('val_accuracy', direction='max'),
            max_epochs=1000,
            factor=3,
            directory=os.path.normpath(fldr_hyper+'/'+setup+'/'),
            project_name='cv'+str(cv))
        tuner.search(np.array(X_trn),
                          np.array(y_trn),
                          validation_data=(np.array(X_val), np.array(y_val)),
                          batch_size=32, epochs=1000, verbose=0, callbacks=[clbck_hyper])
    cv = cv + 1
#%%
## =============================================================================
## Look at best hyperparameters
if print_best_hyper:
    tuner = Hyperband(
        build_hypermodel,
        objective=kt.Objective('val_accuracy', direction='max'),
        max_epochs=1000,
        factor=3,
        directory=os.path.normpath(fldr_hyper+'/'+setup+'/'),
        project_name='cv0')
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    print(best_hps[0].values)