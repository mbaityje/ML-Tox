import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import pubchempy as pcp
from time import ctime
import helpers.DataSciPy as dsp
import tensorflow as tf
from tensorflow import keras

# MLP data preprocessing ------------------------------------------------------


def scores_cat(conc):
    """Return score of the concentration passed as input
    Inputs:
        - conc (float): concentration value
    Outputs:
        - score (int): score given to the concentration, from 1 to 5 (see report for detail)"""

    if conc <= 0.1:
        return 4
    elif (conc > 0.1) and (conc <= 1):
        return 3
    elif (conc > 1) and (conc <= 10):
        return 2
    elif (conc > 10) and (conc <= 100):
        return 1
    else:
        return 0


def binary_score(final_db):
    """Set label to 0 or 1 based on the selected threshold of 1 mg/L
    Inputs:
        - final_db (Pandas dataframe): cleaned dataframe
    Outputs:
        - final_db_scored (Pandas dataframe): dataframe with column score containing the labels"""

    # Computing score
    final_db_scored = final_db.copy()
    final_db_scored["score"] = np.where(final_db_scored.conc1_mean > 1, 0, 1)
    # Drop conc1_mean (it is replaced by score)
    final_db_scored = final_db_scored.drop(columns=["conc1_mean"])

    # Info
    num0 = len(final_db_scored[final_db_scored.score == 0])
    num1 = len(final_db_scored[final_db_scored.score == 1])
    print(
        "There are {} datapoints with label 0 and {} datapoints with label 1".format(
            num0, num1
        )
    )
    return final_db_scored


def multi_score(final_db):
    """Set label to 0 or 1 based on the selected threshold of 1 mg/L
    Inputs:
        - final_db (Pandas dataframe): cleaned dataframe
    Outputs:
        - final_db_scored (Pandas dataframe): dataframe with column score containing the labels"""

    # Computing score
    final_db_scored = final_db.copy()
    final_db_scored["score"] = final_db_scored.conc1_mean.apply(lambda x: scores_cat(x))
    # Drop conc1_mean (it is replaced by score)
    final_db_scored = final_db_scored.drop(columns=["conc1_mean"])

    # Info
    for i in range(0, 5):
        num = len(final_db_scored[final_db_scored.score == i])
        print("There are {} datapoints with label {}".format(num, i))

    return final_db_scored


def prepare_data(path_data, setup, numerical, multiclass=False):
    # ------------------------------------------------------------------------
    # Prepare data for use as features in MLP models
    # ------------------------------------------------------------------------
    final_db = pd.read_csv(path_data)
    final_db = final_db.drop(columns=["Unnamed: 0"])

    if setup == "fathead" or setup == "rainbow":
        if setup == "fathead":
            fish_curr = "Actinopterygii Cypriniformes Cyprinidae Pimephales promelas"
        elif setup == "rainbow":
            fish_curr = "Actinopterygii Salmoniformes Salmonidae Oncorhynchus mykiss"
        # keep only fathead minnow or rainbow trout in trainval and remove all but chemical features and collapse duplicated rows to median conc
        tmp = numerical.copy()
        tmp.extend(["test_cas", "conc1_mean"])
        X_trainval = final_db[final_db["fish"] == fish_curr]
        X_cllps_trainval = X_trainval.loc[:, tmp]
        X_cllps_trainval = (
            X_cllps_trainval.groupby(by="test_cas").agg("median").reset_index()
        )
        X_cllps_trainval.index = X_cllps_trainval.loc[:, "test_cas"]
        X_test = final_db[final_db["fish"] != fish_curr]
        X_cllps_test = X_test.loc[:, tmp]
        X_cllps_test = X_cllps_test.groupby(by="test_cas").agg("median").reset_index()
        X_cllps_test.index = X_cllps_test.loc[:, "test_cas"]

        X_trainval = (
            X_trainval.loc[:, ["test_cas", "pubchem2d"]]
            .drop_duplicates()
            .join(X_cllps_trainval, on=["test_cas"], lsuffix="_lft")
        )
        X_trainval = X_trainval.drop(columns=["test_cas_lft"])
        X_test = (
            X_test.loc[:, ["test_cas", "pubchem2d"]]
            .drop_duplicates()
            .join(X_cllps_test, on=["test_cas"], lsuffix="_lft")
        )
        X_test = X_test.drop(columns=["test_cas_lft"])

        fp = pd.DataFrame(X_trainval.pubchem2d.apply(dsp.splt_str))
        fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
        fp.columns = ["pub" + str(i) for i in range(1, 882)]
        # data with pubchem2d
        X_trainval = X_trainval.drop(columns=["pubchem2d"]).join(fp)
        X_trainval = X_trainval.drop(columns=["test_cas"])

        fp = pd.DataFrame(X_test.pubchem2d.apply(dsp.splt_str))
        fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
        fp.columns = ["pub" + str(i) for i in range(1, 882)]
        # data with pubchem2d
        X_test = X_test.drop(columns=["pubchem2d"]).join(fp)
        X_test = X_test.drop(columns=["test_cas"])

        # Prepare binary or multiclass problem
        if multiclass:
            X_trainval = multi_score(X_trainval)
            X_test = multi_score(X_test)
        else:
            X_trainval = binary_score(X_trainval)
            X_test = binary_score(X_test)

        sclr = MinMaxScaler()
        dummy = dsp.Dataset()
        dummy.setup_data(
            X=X_trainval.drop(columns=["score"]),
            y=X_trainval.loc[:, ["score"]],
            split_test=0,
        )

        y_trainval = X_trainval.loc[:, ["score"]]
        X_trainval = X_trainval.drop(columns=["score"])
        y_test = X_test.loc[:, ["score"]]
        X_test = X_test.drop(columns=["score"])
        # Tensorflow does not accept boolean (or int) data, so convert to float
        X_trainval = X_trainval.astype(np.float64)
        y_trainval = y_trainval.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_test = y_test.astype(np.float64)

        sclr = MinMaxScaler()
        sclr.fit(X_trainval[numerical])
        xtmp = X_trainval.copy()
        xtmp.loc[:, numerical] = sclr.transform(X_trainval.loc[:, numerical])
        X_trainval = xtmp.copy()
        xtmp = X_test.copy()
        xtmp.loc[:, numerical] = sclr.transform(X_test.loc[:, numerical])
        X_test = xtmp.copy()
        return X_trainval, y_trainval, X_test, y_test, dummy

    else:
        if setup == "onlychem":
            # remove all but chemical features and collapse duplicated rows to median conc
            tmp = numerical.copy()
            tmp.extend(["test_cas", "conc1_mean"])
            final_db_cllps = final_db.loc[:, tmp]
            final_db_cllps = (
                final_db_cllps.groupby(by="test_cas").agg("median").reset_index()
            )
            final_db_cllps.index = final_db_cllps.loc[:, "test_cas"]
            final_db = (
                final_db.loc[:, ["test_cas", "pubchem2d"]]
                .drop_duplicates()
                .join(final_db_cllps, on=["test_cas"], lsuffix="_lft")
            )
            final_db = final_db.drop(columns=["test_cas_lft"])
        else:
            final_db = final_db.drop(columns=["fish", "smiles"])

        fp = pd.DataFrame(final_db.pubchem2d.apply(dsp.splt_str))
        fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
        fp.columns = ["pub" + str(i) for i in range(1, 882)]

        # data with pubchem2d
        final_db = final_db.drop(columns=["pubchem2d"]).join(fp)
        final_db = final_db.drop(columns=["test_cas"])

        # Prepare binary or multiclass problem
        if multiclass:
            final_db = multi_score(final_db)
        else:
            final_db = binary_score(final_db)

        # use min max scaler (similar to Simone, and it seems reasonable looking at
        # the distribution of the data)
        # sclr = MinMaxScaler(feature_range=(-1,1))
        sclr = MinMaxScaler()

        dummy = dsp.Dataset()
        dummy.setup_data(
            X=final_db.drop(columns=["score"]),
            y=final_db.loc[:, ["score"]],
            split_test=0,
        )

        if setup != "onlychem":
            encode_these = [
                "species",
                "conc1_type",
                "exposure_type",
                "obs_duration_mean",
                "family",
                "genus",
                "tax_order",
                "class",
                "application_freq_unit",
                "media_type",
                "control_type",
            ]
            dummy.encode_categories(variables=encode_these, onehot=True)

        # Tensorflow does not accept boolean (or int) data, so convert to float
        X = dummy.X_train.astype(np.float64)
        y = dummy.y_train.astype(np.float64)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
        )

        sclr = MinMaxScaler()
        sclr.fit(X_trainval[numerical])
        xtmp = X_trainval.copy()
        xtmp.loc[:, numerical] = sclr.transform(X_trainval.loc[:, numerical])
        X_trainval = xtmp.copy()
        xtmp = X_test.copy()
        xtmp.loc[:, numerical] = sclr.transform(X_test.loc[:, numerical])
        X_test = xtmp.copy()
        return X_trainval, y_trainval, X_test, y_test, dummy


# MLP model building helpers -------------------------------------------------


def build_models(
    dat_obj,
    nout=1,
    activation="sigmoid",
    loss="binary_crossentropy",
    metrics=["accuracy"],
):

    # Define models
    model0 = keras.models.Sequential()
    initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=1)
    # regu = tf.compat.v1.keras.regularizers.L2(0.001)
    model0.add(keras.Input(shape=(dat_obj.X_train.shape[1],)))
    model0.add(
        keras.layers.Dense(nout, activation=activation, kernel_initializer=initializer)
    )
    model0.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )  # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

    model1 = keras.models.Sequential()
    initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=1)
    model1.add(keras.Input(shape=(dat_obj.X_train.shape[1],)))
    model1.add(
        keras.layers.Dense(50, activation="tanh", kernel_initializer=initializer)
    )
    model1.add(
        keras.layers.Dense(nout, activation=activation, kernel_initializer=initializer)
    )
    model1.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )  # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

    model2 = keras.models.Sequential()
    initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=1)
    model2.add(keras.Input(shape=(dat_obj.X_train.shape[1],)))
    model2.add(
        keras.layers.Dense(500, activation="tanh", kernel_initializer=initializer)
    )
    model2.add(
        keras.layers.Dense(nout, activation=activation, kernel_initializer=initializer)
    )
    model2.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )  # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

    model3 = keras.models.Sequential()
    initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=1)
    model3.add(keras.Input(shape=(dat_obj.X_train.shape[1],)))
    model3.add(
        keras.layers.Dense(500, activation="tanh", kernel_initializer=initializer)
    )
    model3.add(keras.layers.Dropout(rate=0.3))
    model3.add(
        keras.layers.Dense(200, activation="tanh", kernel_initializer=initializer)
    )
    model3.add(
        keras.layers.Dense(nout, activation=activation, kernel_initializer=initializer)
    )
    model3.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )  # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

    model4 = keras.models.Sequential()
    initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=1)
    model4.add(keras.Input(shape=(dat_obj.X_train.shape[1],)))
    model4.add(
        keras.layers.Dense(500, activation="tanh", kernel_initializer=initializer)
    )
    model4.add(keras.layers.Dropout(rate=0.3))
    model4.add(
        keras.layers.Dense(200, activation="tanh", kernel_initializer=initializer)
    )
    model4.add(keras.layers.Dropout(rate=0.3))
    model4.add(
        keras.layers.Dense(nout, activation=activation, kernel_initializer=initializer)
    )
    model4.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )  # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

    model5 = keras.models.Sequential()
    initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=1)
    model5.add(keras.Input(shape=(dat_obj.X_train.shape[1],)))
    model5.add(
        keras.layers.Dense(500, activation="tanh", kernel_initializer=initializer)
    )
    model5.add(keras.layers.Dropout(rate=0.3))
    model5.add(
        keras.layers.Dense(200, activation="tanh", kernel_initializer=initializer)
    )
    model5.add(keras.layers.Dropout(rate=0.3))
    model5.add(
        keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)
    )
    model5.add(
        keras.layers.Dense(50, activation="tanh", kernel_initializer=initializer)
    )
    model5.add(
        keras.layers.Dense(nout, activation=activation, kernel_initializer=initializer)
    )
    model5.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )  # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

    model6 = keras.models.Sequential()
    initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=1)
    model6.add(keras.Input(shape=(dat_obj.X_train.shape[1],)))
    model6.add(
        keras.layers.Dense(500, activation="tanh", kernel_initializer=initializer)
    )
    model6.add(keras.layers.Dropout(rate=0.5))
    model6.add(
        keras.layers.Dense(200, activation="tanh", kernel_initializer=initializer)
    )
    model6.add(keras.layers.Dropout(rate=0.5))
    model6.add(
        keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)
    )
    model6.add(keras.layers.Dropout(rate=0.3))
    model6.add(
        keras.layers.Dense(50, activation="tanh", kernel_initializer=initializer)
    )
    model6.add(
        keras.layers.Dense(nout, activation=activation, kernel_initializer=initializer)
    )
    model6.compile(
        optimizer="adam", loss=loss, metrics=metrics
    )  # 'cross-entropy' is the same as the 'log-loss' of scikitlearn

    # Make a dict of the models to be fitted
    mdls = {
        "model0": model0,
        "model1": model1,
        "model2": model2,
        "model3": model3,
        "model4": model4,
        "model5": model5,
        "model6": model6,
    }

    return mdls


def build_hypermodel(hp):
    model = keras.Sequential()
    for i in range(hp.Int("num_layers", 2, 5)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=64, max_value=1024, step=64),
                activation="tanh",
            )
        )
        model.add(
            keras.layers.Dropout(
                rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
            )
        )
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_hypermodel_multiclass(hp):
    model = keras.Sequential()
    for i in range(hp.Int("num_layers", 2, 5)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=64, max_value=1024, step=64),
                activation="tanh",
            )
        )
        model.add(
            keras.layers.Dropout(
                rate=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
            )
        )
    model.add(keras.layers.Dense(5, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
