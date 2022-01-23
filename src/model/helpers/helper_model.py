import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    precision_score,
    accuracy_score,
    f1_score,
)
from time import ctime
from collections import Counter
from numpy.lib.stride_tricks import as_strided
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import os
import warnings

warnings.filterwarnings("ignore")


# ------------------------------------------features name-----------------------------------

categorical = [
    "class",
    "tax_order",
    "family",
    "genus",
    "species",
    "control_type",
    "media_type",
    "application_freq_unit",
    "exposure_type",
    "conc1_type",
    "obs_duration_mean",
]

numerical = [
    "ring_number",
    "tripleBond",
    "doubleBond",
    "alone_atom_number",
    "oh_count",
    "atom_number",
    "bonds_number",
    "MorganDensity",
    "LogP",
    "Mol",
    "WaterSolubility",
    "MeltingPoint",
]

# comparing was used to identify similar experiments
comparing = ["test_cas"] + categorical


def create_pub_column(db):
    """This function separates the column with pubchem fingerprint into 881 columns."""
    db_pub = pd.concat(
        [
            db["test_cas"],
            pd.DataFrame(
                pd.DataFrame(db["pubchem2d"].values)
                .apply(
                    lambda x: x.str.replace("", " ").str.strip().str.split(" "), axis=1
                )[0]
                .to_list(),
                columns=["pub" + str(i) for i in range(1, 882)],
            ),
        ],
        axis=1,
    ).drop_duplicates()
    return db_pub


def hamming_matrix(X1, X2, cat_features):
    """This function calculates the Hamming distance between two experiments based
    on categorical features, and the final result is a distance matrix containing
    the distance from each experiment in set X1 to each experiment in set X2."""
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            return squareform(pdist(X1[cat_features], metric="hamming"))
    else:
        return cdist(X1[cat_features], X2[cat_features], metric="hamming")


def euclidean_matrix(X1, X2, num_features):
    """This function calculates the Euclidean distance between two experiments based
    on numerical features, and the final result is a distance matrix containing the
    distance from each experiment in set X1 to each experiment in set X2."""
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            return squareform(pdist(X1[num_features], metric="euclidean"))
    else:
        return cdist(X1[num_features], X2[num_features], metric="euclidean")


def pubchem2d_matrix(X1, X2):
    """This function calculates the Hamming distance between two experiments based
    on pubchem features, and the final result is a distance matrix containing the
    distance from each experiment in set X1 to each experiment in set X2."""
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            return squareform(
                pdist(X1[X1.columns[X1.columns.str.contains("pub")]], metric="hamming")
            )
    else:
        return cdist(
            X1[X1.columns[X1.columns.str.contains("pub")]],
            X2[X2.columns[X2.columns.str.contains("pub")]],
            metric="hamming",
        )


def cal_matrixs(X1, X2, categorical, non_categorical):
    """This function calculates the three distance matrices using the input
    two experiments sets."""
    basic_mat = euclidean_matrix(X1, X2, non_categorical)
    matrix_h = hamming_matrix(X1, X2, categorical)
    matrix_p = pubchem2d_matrix(X1, X2)
    return basic_mat, matrix_h, matrix_p


# euclidean matrix will always has 1 as parameter.
def matrix_combine(basic_mat, matrix_h, matrix_p, ah, ap):
    """This function combines the three distance matrices with alpha values."""
    dist_matr = ah * matrix_h
    dist_matr += basic_mat
    dist_matr += ap * matrix_p
    dist_matr = pd.DataFrame(dist_matr)
    return dist_matr


def dist_matrix(X1, X2, non_categorical, categorical, ah, ap):
    """This function calculates the three distance matrices using the input
    two experiments sets and combines those into one final distance matrix.
    """
    matrix_h = hamming_matrix(X1, X2, categorical)
    dist_matr = ah * matrix_h
    del matrix_h
    basic_mat = euclidean_matrix(X1, X2, non_categorical)
    dist_matr += basic_mat
    del basic_mat
    matrix_p = pubchem2d_matrix(X1, X2)
    dist_matr += ap * matrix_p
    del matrix_p
    dist_matr = pd.DataFrame(dist_matr)
    return dist_matr


def multiclass_encoding(var, threshold=[10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2]):
    """Classified the experiments into five classes: non-toxic (0), slightly toxic (1),
    moderately toxic (2), highly toxic (3), very highly toxic (4).
    Classified the experiments into five classes: non-toxic (0), slightly toxic (1),
    moderately toxic (2), highly toxic (3), very highly toxic (4)."""
    var_ls = list(var)
    for i in range(len(var_ls)):
        if var_ls[i] <= threshold[0]:
            var_ls[i] = 4

        elif threshold[0] < var_ls[i] <= threshold[1]:
            var_ls[i] = 3

        elif threshold[1] < var_ls[i] <= threshold[2]:
            var_ls[i] = 2

        elif threshold[2] < var_ls[i] <= threshold[3]:
            var_ls[i] = 1

        else:
            var_ls[i] = 0
    return pd.to_numeric(var_ls, downcast="integer")


def encoding_categorical(cat_cols, database, database_df, cate_encoding="ordinal"):
    """Transform the categorical features into ordinal values or one-hot encoded
    values.
    """

    # fit to the categories in datafusion dataset and training dataset
    categories = []
    for column in database[cat_cols]:
        # add non-existing new category in df dataset but not in training dataset.
        cat_final = np.append(
            database[column].unique(),
            tuple(
                i
                for i in database_df[column].unique()
                if i not in database[column].unique()
            ),
        )
        cat_final = np.array(sorted(cat_final))
        categories.append(cat_final)

    if cate_encoding == "ordinal":
        # Ordinal Encoding for categorical variables
        encoder = OrdinalEncoder(categories=categories, dtype=int)
        encoder.fit(database[cat_cols])
        database[cat_cols] = encoder.transform(database[cat_cols]) + 1

        encoder.fit(database_df[cat_cols])
        database_df[cat_cols] = encoder.transform(database_df[cat_cols]) + 1

    elif cate_encoding == "onehot":
        # Ordinal Encoding for categorical variables
        encoder = OneHotEncoder(categories=categories, dtype=int, sparse=False)

        encoder.fit(database[cat_cols])
        # database[cat_cols] = encoder.transform(database[cat_cols])
        database = pd.concat(
            [
                database.drop(columns=cat_cols),
                pd.DataFrame(
                    encoder.transform(database[cat_cols]),
                    columns=encoder.get_feature_names(cat_cols),
                ),
            ],
            axis=1,
        )

        encoder.fit(database_df[cat_cols])
        # database_df[cat_cols] = encoder.transform(database_df[cat_cols])
        database_df = pd.concat(
            [
                database_df.drop(columns=cat_cols),
                pd.DataFrame(
                    encoder.transform(database_df[cat_cols]),
                    columns=encoder.get_feature_names(cat_cols),
                ),
            ],
            axis=1,
        )

    return database, database_df


# ------------------------------------------load data-----------------------------------
def load_data(
    DATA_PATH,
    encoding,
    categorical_columns,
    drop_columns=["Unnamed: 0"],
    conc_column="conc1_mean",
    label="cte",
    cate_encoding="ordinal",
    encoding_value=1,
    seed=42,
):
    """load the processed dataset, encode the categorical features and labeling"""
    db = pd.read_csv(DATA_PATH).drop(drop_columns, axis=1)
    # db = db[:4000]
    db_pub = create_pub_column(db)
    if label == "c":
        db.drop(columns=categorical, inplace=True)
        db = db.groupby("test_cas").agg("median").reset_index()
        db = db.merge(db_pub, on="test_cas").reset_index(drop=True)
    elif label == "cte":
        db = db.merge(db_pub, on="test_cas").reset_index(drop=True)
        db.drop(columns="pubchem2d", inplace=True)
        if cate_encoding == "ordinal":
            # Ordinal Encoding for categorical variables
            encoder = OrdinalEncoder(dtype=int)
            encoder.fit(db[categorical_columns])
            db[categorical_columns] = encoder.transform(db[categorical_columns]) + 1
        elif cate_encoding == "onehot":
            # One hot Encoding for categorical variables
            ohe = OneHotEncoder(sparse=False)
            ohe.fit(db[categorical_columns])
            db = pd.concat(
                [
                    db.drop(columns=categorical),
                    pd.DataFrame(
                        ohe.transform(db[categorical]),
                        columns=ohe.get_feature_names(categorical),
                    ),
                ],
                axis=1,
            )

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        # print(
        #     "labelling the experiment as " + encoding + " using " + str(encoding_value)
        # )
        db[conc_column] = np.where(db[conc_column].values > encoding_value, 0, 1)

    elif encoding == "multiclass":
        # print(
        #     "labelling the experiment as " + encoding + " using " + str(encoding_value)
        # )
        db[conc_column] = multiclass_encoding(db[conc_column].copy(), encoding_value)

    try:
        X = db.drop(columns=[conc_column, "test_cas", "fish", "smiles"])
    except:
        X = db.drop(columns=[conc_column, "test_cas"])

    Y = db[conc_column].values
    return X, Y


def load_datafusion_datasets(
    DATA_MORTALITY_PATH,
    DATA_OTHER_ENDPOINT_PATH,
    categorical_columns,
    label="cte",
    drop_columns=["Unnamed: 0"],
    conc_column="conc1_mean",
    cate_encoding="ordinal",
    encoding="binary",
    encoding_value=1,
    seed=42,
):
    """load the processed datasets, mortality dataset and datfusion datset,
    encode the categorical features and labeling"""
    db_datafusion = pd.read_csv(DATA_OTHER_ENDPOINT_PATH).drop(columns=drop_columns)
    db_mortality = pd.read_csv(DATA_MORTALITY_PATH).drop(columns=drop_columns)
    # db_datafusion = db_datafusion[:1000]
    # db_mortality = db_mortality[:4000]

    db_mor_pub = create_pub_column(db_mortality)
    db_df_pub = create_pub_column(db_datafusion)

    if label == "cte":
        db_mortality = db_mortality.merge(db_mor_pub, on="test_cas").reset_index(
            drop=True
        )
        db_mortality.drop(columns=["pubchem2d", "fish", "smiles"], inplace=True)

        db_datafusion = db_datafusion.merge(db_df_pub, on="test_cas").reset_index(
            drop=True
        )
        db_datafusion.drop(columns=["pubchem2d", "smiles"], inplace=True)

        #  Encoding for categorical variables
        db_mortality, db_datafusion = encoding_categorical(
            categorical_columns, db_mortality, db_datafusion, cate_encoding
        )

    elif label == "c":
        db_datafusion.drop(columns=categorical, inplace=True)
        db_datafusion = (
            db_datafusion.groupby(["test_cas", "endpoint", "effect"])
            .agg("median")
            .reset_index()
        )
        db_datafusion = db_datafusion.merge(db_df_pub, on="test_cas").reset_index(
            drop=True
        )
        db_mortality.drop(columns=categorical, inplace=True)
        db_mortality = db_mortality.groupby("test_cas").agg("median").reset_index()
        db_mortality = db_mortality.merge(db_mor_pub, on="test_cas").reset_index(
            drop=True
        )

    #  class labeling the concentration value
    if encoding == "binary":
        db_mortality[conc_column] = np.where(
            db_mortality[conc_column].values > encoding_value, 0, 1
        )

        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, conc_column]
            db_datafusion.loc[db_datafusion.effect == ef, conc_column] = np.where(
                conc > np.median(conc), 0, 1
            )

    elif encoding == "multiclass":

        db_mortality[conc_column] = multiclass_encoding(
            db_mortality[conc_column].copy(), encoding_value
        )
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, conc_column].copy()
            db_datafusion.loc[
                db_datafusion.effect == ef, conc_column
            ] = multiclass_encoding(
                conc.values, conc.quantile([0.2, 0.4, 0.6, 0.8]).values
            )

    return db_mortality, db_datafusion


def fit_and_predict(model, X_train, y_train, X_test, y_test, encoding="binary"):
    """fit the model and predict the score on the test data."""
    df_output = pd.DataFrame()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if encoding == "binary":

        df_output.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
        df_output.loc[0, "recall"] = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        df_output.loc[0, "specificity"] = tn / (tn + fp)
        df_output.loc[0, "f1"] = f1_score(y_test, y_pred)
        df_output.loc[0, "precision"] = precision_score(y_test, y_pred)

    elif encoding == "multiclass":
        df_output.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
        df_output.loc[0, "recall"] = recall_score(y_test, y_pred, average="macro")
        df_output.loc[0, "specificity"] = np.nan
        df_output.loc[0, "f1"] = f1_score(y_test, y_pred, average="macro")
        df_output.loc[0, "precision"] = precision_score(y_test, y_pred, average="macro")
    return df_output


# ----------------------------------------------------------------------KNN model------------------------------------------------------------


def select_alpha(
    X,
    Y,
    categorical_columns,
    non_categorical_columns,
    sequence_ham,
    leaf_ls,
    neighbors,
    encoding,
):
    """This function selected the best combination of alpha values using KNN model."""
    # creating the distance matrix
    print("Start calculating distance matrix...", ctime())
    matrix_euc, matrix_h, matrix_p = cal_matrixs(
        X, X, categorical_columns, non_categorical_columns
    )
    print("Distance matrix finished.", ctime())

    best_accs = 0
    count = 0
    print("Start selecting the best parameters....")
    for ah in sequence_ham:
        for ap in sequence_ham:
            for leaf in leaf_ls:
                print(
                    "*" * 50,
                    count / len(sequence_ham) ** 2 * len(leaf_ls),
                    ctime(),
                    end="\r",
                )
                count = count + 1
                result = KNN_model(
                    matrix_euc,
                    matrix_h,
                    matrix_p,
                    X,
                    Y,
                    int(leaf),
                    int(neighbors),
                    ah,
                    ap,
                    encoding,
                )

                if np.mean(result.accuracy) > best_accs + 0.001:
                    best_accs = np.mean(result.accuracy)
                    best_alpha_h = ah
                    best_alpha_p = ap
                    best_leaf = leaf
                    best_result = result
    return best_alpha_h, best_alpha_p, best_leaf, best_result


def KNN_model(
    matrix_euc, matrix_h, matrix_p, X, Y, leaf, neighbor, ah, ap, encoding, seed=25
):
    """Cross validation of KNN model"""
    # using 5-fold cross validation to choose the alphas with best accuracy
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    list_df_output = []

    for train_index, test_index in kf.split(X):

        # Min max transform the dataset according to the training dataset
        matrix_euc = pd.DataFrame(matrix_euc)
        max_euc = matrix_euc.iloc[train_index, train_index].values.max()
        dist_matr = pd.DataFrame(
            ah * matrix_h + ap * matrix_p + matrix_euc.divide(max_euc).values
        )

        x_train = dist_matr.iloc[train_index, train_index]
        x_test = dist_matr.iloc[test_index, train_index]
        y_train = Y[train_index]
        y_test = Y[test_index]

        model = KNeighborsClassifier(
            n_neighbors=neighbor, metric="precomputed", leaf_size=leaf
        )

        result = fit_and_predict(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            encoding,
        )
        list_df_output.append(result)

    df_output = pd.concat(list_df_output, axis=0)

    return df_output


def cv_train_model(X, y, model, encoding="binary"):
    "cross validation of the input model"
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    list_df_output = []

    for train_index, test_index in kf.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        minmax = MinMaxScaler().fit(X_train[numerical])
        new_train = X_train.copy()
        new_test = X_test.copy()

        new_train[numerical] = minmax.transform(X_train.loc[:, numerical])
        new_test[numerical] = minmax.transform(X_test.loc[:, numerical])

        df_score = fit_and_predict(
            model, new_train, y_train, new_test, y_test, encoding
        )
        list_df_output.append(df_score)

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


# ----------------------------------------- simple RASAR---------------------------------------------------------


def take_per_row_strided(A, indx, num_elem=2):
    m, n = A.shape
    A = A.reshape(-1)
    s0 = A.strides[0]

    l_indx = indx + n * np.arange(len(indx))

    out = as_strided(A, (len(A) - int(num_elem) + 1, num_elem), (s0, s0))[l_indx]
    A.shape = m, n
    return out


def right_neighbor(neighbors, X_train, X_train_i):
    """find the nearest neighbor experiment in the X_train set for input experiment."""

    # IDX Neighbors
    idx_neigh_0 = pd.DataFrame(neighbors[1])[0].apply(lambda x: X_train_i.iloc[x].name)
    idx_neigh_1 = pd.DataFrame(neighbors[1])[1].apply(lambda x: X_train_i.iloc[x].name)

    idx_neigh = idx_neigh_0.copy()

    # where the index of the first neighbor is equal to itself
    idx_neigh[X_train.index == idx_neigh_0] = idx_neigh_1[
        X_train.index == idx_neigh_0
    ].values

    # Distance from the Nearest Neighbor that is NOT itself
    dist_0 = pd.DataFrame(neighbors[0])[0]
    dist_1 = pd.DataFrame(neighbors[0])[1]

    distance = dist_0.copy()
    distance[X_train.index == idx_neigh_0] = dist_1[X_train.index == idx_neigh_0].values

    return idx_neigh, distance


def woalphas_cal_s_rasar(X_train, X_test, y_train, y_test, encoding="binary"):
    """calculate the nearest points in each class for training experiments"""
    if encoding == "binary":
        labels = [0, 1]
    elif encoding == "multiclass":
        labels = [0, 1, 2, 3, 4]
    train_rasar = pd.DataFrame()
    test_rasar = pd.DataFrame()
    train_rasar["idx_train"] = X_train.index.values
    test_rasar["idx_test"] = X_test.index.values
    train_rasar["label_train"] = y_train
    test_rasar["label_test"] = y_test

    for i in labels:
        X_train_i = X_train[y_train == i].copy()
        knn = KNeighborsClassifier(n_neighbors=2).fit(X_train_i, y_train[y_train == i])
        neigh_train = knn.kneighbors(X_train, return_distance=True)
        idx_neigh, dist = right_neighbor(neigh_train, X_train, X_train_i)
        train_rasar["dist_neigh" + str(i)] = dist
        train_rasar["idx_neigh" + str(i)] = idx_neigh.values

        knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_i, y_train[y_train == i])
        neigh_test = knn.kneighbors(X_test, return_distance=True)
        idx_neigh_test = pd.DataFrame(neigh_test[1])[0].apply(
            lambda x: X_train_i.iloc[x].name
        )
        test_rasar["dist_neigh" + str(i)] = neigh_test[0].ravel()
        test_rasar["idx_neigh" + str(i)] = idx_neigh_test.values

    return train_rasar, test_rasar


def woalphas_model_s_rasar(X, y, model, encoding="binary"):
    """simple rasar model with cross validation"""
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    list_df_output = []

    for train_index, test_index in kf.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        minmax = MinMaxScaler().fit(X_train[numerical])
        new_train = X_train.copy()
        new_test = X_test.copy()
        new_train[numerical] = minmax.transform(X_train.loc[:, numerical])
        new_test[numerical] = minmax.transform(X_test.loc[:, numerical])

        train_rasar, test_rasar = woalphas_cal_s_rasar(
            new_train, new_test, y_train, y_test, encoding
        )

        df_score = fit_and_predict(
            model,
            train_rasar[train_rasar.filter(like="dist").columns],
            y_train,
            test_rasar[test_rasar.filter(like="dist").columns],
            y_test,
            encoding,
        )
        list_df_output.append(df_score)

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


def cal_s_rasar(
    train_distance_matrix,
    test_distance_matrix,
    y_train,
    n_neighbors=1,
    encoding="binary",
):
    """calculate the nearest points in each class for training experiments with alphas"""
    df_rasar_train = pd.DataFrame()
    df_rasar_test = pd.DataFrame()

    if encoding == "binary":
        labels = [0, 1]
    elif encoding == "multiclass":
        labels = [0, 1, 2, 3, 4]

    for i in labels:
        dist_matr_train_train_i = train_distance_matrix.iloc[:, y_train == i]
        values = dist_matr_train_train_i.values
        values.sort(axis=1)
        indx = (dist_matr_train_train_i == 0).astype("int64").sum(axis=1).values
        disti = pd.DataFrame(take_per_row_strided(values, indx, n_neighbors))
        df_rasar_train = pd.concat([disti, df_rasar_train], axis=1)

        dist_matr_test_test_i = test_distance_matrix.iloc[:, y_train == i]
        values = dist_matr_test_test_i.values
        values.sort(axis=1)
        indx = (dist_matr_test_test_i == 0).astype("int64").sum(axis=1).values
        disti = pd.DataFrame(take_per_row_strided(values, indx, n_neighbors))
        df_rasar_test = pd.concat([disti, df_rasar_test], axis=1)
    df_rasar_train.columns = range(df_rasar_train.shape[1])
    df_rasar_train.columns = [str(x) for x in df_rasar_train.columns]
    df_rasar_test.columns = range(df_rasar_test.shape[1])
    df_rasar_test.columns = [str(x) for x in df_rasar_test.columns]
    return df_rasar_train, df_rasar_test


def model_s_rasar(
    matrix_euc,
    matrix_h,
    matrix_p,
    ah,
    ap,
    Y,
    X,
    n_neighbors=int(1),
    encoding="binary",
    model=RandomForestClassifier(random_state=0, n_estimators=200),
):
    """simple rasar model with cross validation with alphas"""
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    list_df_output = []
    for train_index, test_index in kf.split(X):
        matrix_euc = pd.DataFrame(matrix_euc)
        max_euc = matrix_euc.iloc[train_index, train_index].values.max()

        distance_matrix = pd.DataFrame(
            ah * matrix_h + ap * matrix_p + matrix_euc.divide(max_euc).values
        )

        y_train = Y[train_index]
        y_test = Y[test_index]
        train_rasar, test_rasar = cal_s_rasar(
            distance_matrix.iloc[train_index, train_index],
            distance_matrix.iloc[test_index, train_index],
            y_train,
            n_neighbors,
            encoding,
        )

        df_score = fit_and_predict(
            model,
            train_rasar,
            y_train,
            test_rasar,
            y_test,
            encoding,
        )
        list_df_output.append(df_score)

        del y_train

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


# ---------------------------- datafusion-----------------------------------------------


def find_similar_exp(db_mortality, db_datafusion_rasar, db_endpoint, effect, encoding):
    """Given an LC50-Mortality experiment, this function checks if there is another experiment with
    other endpoint and effect done in the same condition. If it exists in the other database and the
    similar experiment has negative/positive (non toxic/toxic) outcome, the function will return
    respectively -1 or 1. If the similar experiment does not exist, the function will return 0."""
    try:
        temp = pd.merge(
            db_mortality.reset_index(drop=True),
            db_endpoint[db_endpoint.effect == effect],
            on=comparing,
            how="left",
        )
        temp = temp[
            temp[np.setdiff1d(db_mortality.columns, comparing)[0] + "_y"].notnull()
        ]
        temp = pd.DataFrame(
            temp.groupby(comparing)["conc1_mean"].agg(pd.Series.mode)
        ).reset_index()
        temp = pd.merge(
            db_mortality.reset_index(drop=True), temp, on=comparing, how="left"
        )
        if encoding == "binary":
            # equalize the distance from class 0 or class 1 to Unknown
            temp["conc1_mean"] = np.where(
                temp["conc1_mean"] == 0, -1, (np.where(temp["conc1_mean"] == 1, 1, 0))
            )
        elif encoding == "multiclass":
            temp["conc1_mean"] = temp["conc1_mean"].fillna("Unknown")
    except:
        if encoding == "binary":
            temp = pd.DataFrame(
                0, index=np.arange(len(db_datafusion_rasar)), columns=["conc1_mean"]
            )
        elif encoding == "multiclass":
            temp = pd.DataFrame(
                "Unknown",
                index=np.arange(len(db_datafusion_rasar)),
                columns=["conc1_mean"],
            )

    return temp


def woalphas_find_similar_exp(
    exp_mortality, db_endpoint_effect, compare_features, encoding="binary"
):
    """Given an LC50-Mortality experiment, this function checks the most common
    label of the experiments with other endpoint and effect done in the same condition."""
    out = db_endpoint_effect.conc1_mean[
        (db_endpoint_effect[compare_features] == exp_mortality[compare_features]).all(
            axis=1
        )
    ].values
    if encoding == "binary":
        try:
            return -1 if Counter(out).most_common(1)[0][0] == 0 else 1
        except:
            return 0
    elif encoding == "multiclass":
        try:
            return Counter(out).most_common(1)[0][0]
        except:
            return "Unknown"


def find_datafusion_neighbor(
    db_datafusion_rasar_train,
    db_datafusion_rasar_test,
    db_datafusion,
    db_datafusion_matrix,
    train_index,
    test_index,
    effect,
    endpoint,
    encoding="binary",
):
    """Find the nearest experiment with other endpoint and effect done in the same
    condition for all the training experiments"""
    if encoding == "binary":
        label = [0, 1]
    elif encoding == "multiclass":
        label = [0, 1, 2, 3, 4]

    for a in label:
        db_end_eff = db_datafusion.loc[
            (db_datafusion.effect == effect)
            & (db_datafusion.conc1_mean == a)
            & (db_datafusion.endpoint == endpoint)
        ]
        if len(db_end_eff) == 0:
            continue
        else:
            train_test_matrix = db_datafusion_matrix.iloc[
                train_index,
                np.array(
                    (db_datafusion.effect == effect)
                    & (db_datafusion.conc1_mean == a)
                    & (db_datafusion.endpoint == endpoint)
                ).nonzero()[0],
            ]
            train_test_matrix = train_test_matrix.reset_index(drop=True)
            train_test_matrix.columns = range(train_test_matrix.shape[1])
            col_name = endpoint + "_" + effect + "_" + str(a)

            db_datafusion_rasar_train[col_name] = np.array(
                train_test_matrix.min(axis=1)
            )

            test_test_matrix = db_datafusion_matrix.iloc[
                test_index,
                np.array(
                    (db_datafusion.effect == effect)
                    & (db_datafusion.conc1_mean == a)
                    & (db_datafusion.endpoint == endpoint)
                ).nonzero()[0],
            ]
            test_test_matrix = test_test_matrix.reset_index(drop=True)
            test_test_matrix.columns = range(test_test_matrix.shape[1])

            db_datafusion_rasar_test[col_name] = np.array(test_test_matrix.min(axis=1))

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def woalphas_label_df_rasar(
    X_train,
    X_test,
    db_datafusion,
    comparing=["test_cas"] + categorical,
    encoding="binary",
):
    """This function checks the most common label of the experiments with other endpoint
    and effect done in the same condition for all the training experiments."""

    grouped_datafusion = db_datafusion.groupby(by=["endpoint", "effect"])

    df_rasar_label_train = pd.DataFrame()
    df_rasar_label_test = pd.DataFrame()
    count = 0
    for g in grouped_datafusion.groups:
        print(
            "*" * 50,
            count / len(grouped_datafusion.groups),
            ctime(),
            end="\r",
        )
        count = count + 1
        name = g[0] + "_" + g[1] + "_" + "label"

        group = grouped_datafusion.get_group(g).drop(columns=["endpoint", "effect"])

        df_rasar_label_train[name] = X_train.apply(
            lambda x: woalphas_find_similar_exp(x, group, comparing, encoding), axis=1
        ).reset_index(drop=True)

        df_rasar_label_test[name] = X_test.apply(
            lambda x: woalphas_find_similar_exp(x, group, comparing, encoding), axis=1
        ).reset_index(drop=True)

    return df_rasar_label_train, df_rasar_label_test


def woalphas_cal_df_rasar(X_train, X_test, db_datafusion):
    """This function finds the nearest experiment with other endpoint
    and effect done in the same condition for all the training experiments."""

    db_datafusion = db_datafusion.drop(columns="test_cas")
    grouped_datafusion = db_datafusion.groupby(by=["endpoint", "effect", "conc1_mean"])

    df_rasar_train = pd.DataFrame()
    df_rasar_test = pd.DataFrame()

    for group in grouped_datafusion.groups:

        name = group[0] + "_" + group[1] + "_" + str(group[2])

        train_X = grouped_datafusion.get_group(group).drop(
            columns=["endpoint", "effect", "conc1_mean"]
        )
        train_y = grouped_datafusion.get_group(group)["conc1_mean"].values

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(train_X, train_y)

        test_X = X_train.copy()
        neigh = knn.kneighbors(test_X, return_distance=True)
        df_rasar_train[name] = neigh[0].ravel()

        test_X = X_test.copy()
        neigh = knn.kneighbors(test_X, return_distance=True)
        df_rasar_test[name] = neigh[0].ravel()

    return df_rasar_train, df_rasar_test


def woalphas_model_df_rasar(X, y, df_db, model, df_rasar_label, encoding="binary"):
    """datafusion rasar model with cross validation"""
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    list_df_output = []

    for train_index, test_index in kf.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        df_rasar_label_train = df_rasar_label.iloc[train_index].reset_index(drop=True)
        df_rasar_label_test = df_rasar_label.iloc[test_index].reset_index(drop=True)

        minmax = MinMaxScaler().fit(X_train[numerical])
        new_train = X_train.copy()
        new_test = X_test.copy()
        new_train[numerical] = minmax.transform(X_train.loc[:, numerical])
        new_test[numerical] = minmax.transform(X_test.loc[:, numerical])

        s_rasar_train, s_rasar_test = woalphas_cal_s_rasar(
            new_train, new_test, y_train, y_test, encoding
        )

        df_rasar_train, df_rasar_test = woalphas_cal_df_rasar(
            new_train, new_test, df_db
        )
        train_rasar = pd.concat(
            [
                s_rasar_train[s_rasar_train.filter(like="dist").columns],
                df_rasar_train,
                df_rasar_label_train,
            ],
            axis=1,
        )
        test_rasar = pd.concat(
            [
                s_rasar_test[s_rasar_test.filter(like="dist").columns],
                df_rasar_test,
                df_rasar_label_test,
            ],
            axis=1,
        )

        if encoding == "binary":
            df_score = fit_and_predict(
                model, train_rasar, y_train, test_rasar, y_test, encoding
            )
            list_df_output.append(df_score)

        elif encoding == "multiclass":

            train_rasar.loc[:, "target"] = y_train
            test_rasar.loc[:, "target"] = y_test
            train_rf_h2o = h2o.H2OFrame(train_rasar)
            test_rf_h2o = h2o.H2OFrame(test_rasar)

            for col in train_rasar.columns:
                if "label" in col:
                    train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                    test_rf_h2o[col] = test_rf_h2o[col].asfactor()

            train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
            test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

            model.train(y="target", training_frame=train_rf_h2o)
            y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]
            del train_rf_h2o, test_rf_h2o
            df_score = pd.DataFrame()
            df_score.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
            df_score.loc[0, "recall"] = recall_score(y_test, y_pred, average="macro")
            df_score.loc[0, "specificity"] = np.nan
            df_score.loc[0, "f1"] = f1_score(y_test, y_pred, average="macro")
            df_score.loc[0, "precision"] = precision_score(
                y_test, y_pred, average="macro"
            )
            list_df_output.append(df_score)
    df_output = pd.concat(list_df_output, axis=0)
    return df_output


def cal_df_rasar(
    train_index,
    test_index,
    db_mortality_train,
    db_mortality_test,
    db_datafusion,
    db_datafusion_matrix,
    train_endpoint,
    train_effect,
    encoding,
):
    """This function finds the nearest experiment with other endpoint
    and effect done in the same condition for all the training experiments."""
    db_datafusion_rasar_train = pd.DataFrame()
    db_datafusion_rasar_test = pd.DataFrame()

    for endpoint in db_datafusion.endpoint.unique():

        db_endpoint = db_datafusion[db_datafusion.endpoint == endpoint]
        for effect in db_endpoint.effect.unique():
            if (str(effect) == train_effect) & (str(endpoint) in train_endpoint):
                continue
            else:
                (
                    db_datafusion_rasar_train,
                    db_datafusion_rasar_test,
                ) = find_datafusion_neighbor(
                    db_datafusion_rasar_train,
                    db_datafusion_rasar_test,
                    db_datafusion,
                    db_datafusion_matrix,
                    train_index,
                    test_index,
                    effect,
                    endpoint,
                    encoding,
                )

                # FINDING LABELS
                col_name = endpoint + "_" + effect + "_label"
                temp = find_similar_exp(
                    db_mortality_train,
                    db_datafusion_rasar_train,
                    db_endpoint,
                    effect,
                    encoding,
                )

                db_datafusion_rasar_train[col_name] = temp["conc1_mean"]

                temp = find_similar_exp(
                    db_mortality_test,
                    db_datafusion_rasar_test,
                    db_endpoint,
                    effect,
                    encoding,
                )

                db_datafusion_rasar_test[col_name] = temp["conc1_mean"]

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def model_df_rasar(
    matrix_euc,
    matrix_h,
    matrix_p,
    matrix_euc_df,
    matrix_h_df,
    matrix_p_df,
    ah,
    ap,
    X,
    Y,
    db_datafusion,
    train_endpoint,
    train_effect,
    model=RandomForestClassifier(random_state=10),
    n_neighbors=1,
    encoding="binary",
):
    """datafusion rasar model with cross validation and alphas, using nearest experiment
    in mortality datasets, nearest experiment on each other effect and endpoint in datafusion
    dataset and whether same results exist on each effect and endpoint in datafusion dataset to train the model."""

    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    list_df_output = []
    for train_index, test_index in kf.split(X):
        matrix_euc = pd.DataFrame(matrix_euc)
        max_euc = matrix_euc.iloc[train_index, train_index].values.max()

        distance_matrix = pd.DataFrame(
            ah * matrix_h + ap * matrix_p + matrix_euc.divide(max_euc).values
        )
        db_datafusion_matrix = pd.DataFrame(
            ah * matrix_h_df
            + ap * matrix_p_df
            + pd.DataFrame(matrix_euc_df).divide(max_euc).values
        )

        dist_matr_train = distance_matrix.iloc[train_index, train_index]
        dist_matr_test = distance_matrix.iloc[test_index, train_index]

        new_X = X.copy()

        new_db_datafusion = db_datafusion.copy()

        y_train = Y[train_index]
        y_test = Y[test_index]

        s_rasar_train, s_rasar_test = cal_s_rasar(
            dist_matr_train, dist_matr_test, y_train, n_neighbors, encoding
        )

        df_rasar_train, df_rasar_test = cal_df_rasar(
            train_index,
            test_index,
            new_X.iloc[train_index],
            new_X.iloc[test_index],
            new_db_datafusion,
            db_datafusion_matrix,
            train_endpoint,
            train_effect,
            encoding,
        )

        train_rasar = pd.concat([s_rasar_train, df_rasar_train], axis=1)
        test_rasar = pd.concat([s_rasar_test, df_rasar_test], axis=1)
        del (
            dist_matr_train,
            dist_matr_test,
            s_rasar_train,
            s_rasar_test,
            df_rasar_train,
            df_rasar_test,
        )

        if encoding == "binary":
            df_score = fit_and_predict(
                model,
                train_rasar,
                y_train,
                test_rasar,
                y_test,
                encoding,
            )
            list_df_output.append(df_score)
        elif encoding == "multiclass":

            train_rasar.loc[:, "target"] = y_train
            test_rasar.loc[:, "target"] = y_test

            train_rasar_h2o = h2o.H2OFrame(train_rasar)
            test_rasar_h2o = h2o.H2OFrame(test_rasar)

            for col in train_rasar.columns:
                if "label" in col:
                    train_rasar_h2o[col] = train_rasar_h2o[col].asfactor()
                    test_rasar_h2o[col] = test_rasar_h2o[col].asfactor()

            train_rasar_h2o["target"] = train_rasar_h2o["target"].asfactor()
            test_rasar_h2o["target"] = test_rasar_h2o["target"].asfactor()

            model.train(y="target", training_frame=train_rasar_h2o)
            y_pred = model.predict(test_rasar_h2o).as_data_frame()["predict"]
            df_score = pd.DataFrame()
            df_score.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
            df_score.loc[0, "recall"] = recall_score(y_test, y_pred, average="macro")
            df_score.loc[0, "specificity"] = np.nan
            df_score.loc[0, "f1"] = f1_score(y_test, y_pred, average="macro")
            df_score.loc[0, "precision"] = precision_score(
                y_test, y_pred, average="macro"
            )
            list_df_output.append(df_score)

        del train_rasar, test_rasar

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


def df2file(info, outputFile):
    """save the dataframe into file."""
    filename = outputFile
    dirname = os.path.dirname(filename)
    if (not os.path.exists(dirname)) & (dirname != ""):
        os.makedirs(dirname)
    info.to_csv(filename)
    print("Result saved.")
