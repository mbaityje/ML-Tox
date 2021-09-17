import pandas as pd
from scipy.spatial.distance import jaccard, pdist, squareform, cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from time import ctime
from tqdm import tqdm
import argparse
from helper_model import *
import multiprocessing as mp
from sklearn.model_selection import train_test_split, ParameterSampler
import pickle
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description="Running Datafusion RASAR model for datasets to find the best N (neighbor) number."
    )
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-idf", "--input_df", dest="inputFile_df", required=True)
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )
    parser.add_argument("-iv", "--invitro", dest="invitro", default=False)
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")
    parser.add_argument("-label", "--train_label", dest="train_label", required=True)
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-effect", "--train_effect", dest="train_effect", required=True)
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()


def func(
    train_index,
    test_index,
    new_X,
    new_db_datafusion,
    db_datafusion_matrix,
    train_label,
    train_effect,
    encoding,
    dist_matr_train,
    dist_matr_test,
    y_train,
    y_test,
    n_neighbors,
    ah,
    ap,
    num,
    model,
):
    simple_rasar_train, simple_rasar_test = cal_data_simple_rasar(
        dist_matr_train, dist_matr_test, y_train, n_neighbors, "binary",
    )
    datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
        train_index,
        test_index,
        new_X.iloc[train_index],
        new_X.iloc[test_index],
        new_db_datafusion,
        db_datafusion_matrix,
        train_label,
        train_effect,
        encoding,
    )

    train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
    test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)
    invitro = args.w_invitro
    invitro_form = args.invitro_label
    db_invitro = args.db_invitro

    if invitro == "own":
        # train_rf = pd.DataFrame()
        # test_rf = pd.DataFrame()
        train_rf, test_rf = simple_rasar_train, simple_rasar_test

    if str(db_invitro) == "overlap":
        if (invitro != "False") & (invitro_form == "number"):
            train_rf["invitro_conc"] = X.iloc[train_index, :].invitro_conc.reset_index(
                drop=True
            )
            test_rf["invitro_conc"] = X.iloc[test_index, :].invitro_conc.reset_index(
                drop=True
            )

        elif (invitro != "False") & (invitro_form == "label"):
            train_rf["invitro_label"] = X.iloc[
                train_index, :
            ].invitro_label.reset_index(drop=True)
            test_rf["invitro_label"] = X.iloc[test_index, :].invitro_label.reset_index(
                drop=True
            )

        elif (invitro != "False") & (invitro_form == "both"):
            train_rf["invitro_conc"] = X.iloc[train_index, :].invitro_conc.reset_index(
                drop=True
            )
            test_rf["invitro_conc"] = X.iloc[test_index, :].invitro_conc.reset_index(
                drop=True
            )
            train_rf["invitro_label"] = X.iloc[
                train_index, :
            ].invitro_label.reset_index(drop=True)
            test_rf["invitro_label"] = X.iloc[test_index, :].invitro_label.reset_index(
                drop=True
            )
        elif (invitro != "False") & (invitro_form == "label_half"):
            train_rf["invitro_label_half"] = X.iloc[
                train_index, :
            ].invitro_label_half.reset_index(drop=True)
            test_rf["invitro_label_half"] = X.iloc[
                test_index, :
            ].invitro_label_half.reset_index(drop=True)

        elif (invitro != "False") & (invitro_form == "both_half"):
            train_rf["invitro_conc"] = X.iloc[train_index, :].invitro_conc.reset_index(
                drop=True
            )
            test_rf["invitro_conc"] = X.iloc[test_index, :].invitro_conc.reset_index(
                drop=True
            )
            train_rf["invitro_label_half"] = X.iloc[
                train_index, :
            ].invitro_label_half.reset_index(drop=True)
            test_rf["invitro_label_half"] = X.iloc[
                test_index, :
            ].invitro_label_half.reset_index(drop=True)
        elif (invitro != "False") & (invitro_form == "label_reserved"):
            train_rf["invitro_label_reserved"] = X.iloc[
                train_index, :
            ].invitro_label_reserved.reset_index(drop=True)
            test_rf["invitro_label_reserved"] = X.iloc[
                test_index, :
            ].invitro_label_reserved.reset_index(drop=True)

        elif (invitro != "False") & (invitro_form == "both_reserved"):
            train_rf["invitro_conc"] = X.iloc[train_index, :].invitro_conc.reset_index(
                drop=True
            )
            test_rf["invitro_conc"] = X.iloc[test_index, :].invitro_conc.reset_index(
                drop=True
            )
            train_rf["invitro_label_reserved"] = X.iloc[
                train_index, :
            ].invitro_label_reserved.reset_index(drop=True)
            test_rf["invitro_label_reserved"] = X.iloc[
                test_index, :
            ].invitro_label_reserved.reset_index(drop=True)
        elif (invitro != "False") & (invitro_form == "label_half_reserved"):
            train_rf["invitro_label_half_reserved"] = X.iloc[
                train_index, :
            ].invitro_label_half_reserved.reset_index(drop=True)
            test_rf["invitro_label_half_reserved"] = X.iloc[
                test_index, :
            ].invitro_label_half_reserved.reset_index(drop=True)

        elif (invitro != "False") & (invitro_form == "both_half_reserved"):
            train_rf["invitro_conc"] = X.iloc[train_index, :].invitro_conc.reset_index(
                drop=True
            )
            test_rf["invitro_conc"] = X.iloc[test_index, :].invitro_conc.reset_index(
                drop=True
            )
            train_rf["invitro_label_half_reserved"] = X.iloc[
                train_index, :
            ].invitro_label_half_reserved.reset_index(drop=True)
            test_rf["invitro_label_half_reserved"] = X.iloc[
                test_index, :
            ].invitro_label_half_reserved.reset_index(drop=True)

    # print(train_rf.columns)
    model.fit(train_rf, y_train)
    y_pred = model.predict(test_rf)

    results = {}
    results["accs"] = accuracy_score(y_test, y_pred)
    results["f1"] = f1_score(y_test, y_pred, average="macro")
    results["neighbors"] = n_neighbors
    results["ah"] = ah
    results["ap"] = ap
    results["fold"] = num
    return results


if __name__ == "__main__":

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
    # non_categorical was numerical features, whcih will be standarized. \
    # Mol,bonds_number, atom_number was previously log transformed due to the maginitude of their values.

    non_categorical = [
        "ring_number",
        "tripleBond",
        "doubleBond",
        "alone_atom_number",
        "oh_count",
        "atom_number",
        "bonds_number",
        "mol_weight",
        "MorganDensity",
        "LogP",
        "water_solubility",
        "melting_point",
    ]
    print("load data...", ctime())

    if args.invitro:
        categorical = ["class", "tax_order", "family", "genus", "species"]

    encoding = "binary"
    encoding_value = 1

    db_mortality, db_datafusion = load_datafusion_datasets(
        args.inputFile,
        args.inputFile_df,
        categorical_columns=categorical,
        encoding=encoding,
        encoding_value=encoding_value,
    )

    X = db_mortality.drop(columns="conc1_mean").copy()
    Y = db_mortality.conc1_mean.values

    # X = X[:150]
    # Y = Y[:150]
    # db_datafusion = db_datafusion[:300]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("calcultaing distance matrix..", ctime())

    matrix_euc, matrix_h, matrix_p = cal_matrixs(
        X_train, X_train, categorical, non_categorical
    )
    matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
        X_train,
        db_datafusion.drop(columns="conc1_mean").copy(),
        categorical,
        non_categorical,
    )

    print("distance matrix calculation finished", ctime())

    results = []
    splitter = KFold(n_splits=4, shuffle=True, random_state=10)
    folds = list(splitter.split(X_train, Y_train))

    i = 1
    if args.hamming_alpha == "logspace":
        sequence_ap = np.logspace(-2, 0, 20)
        sequence_ah = sequence_ap
    else:
        sequence_ap = [float(args.pubchem2d_alpha)]
        sequence_ah = [float(args.hamming_alpha)]

    hyper_params_tune = {
        "max_depth": [i for i in range(10, 30, 6)],
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    }

    params_comb = list(ParameterSampler(hyper_params_tune, n_iter=20, random_state=3))
    final_result = pd.DataFrame()
    for n in args.neighbors:
        best_result = {}
        avg_accs = 0
        for ah in sequence_ah:
            for ap in sequence_ap:
                for j in range(0, len(params_comb)):
                    model = RandomForestClassifier(random_state=10)
                    for k, v in params_comb[j].items():
                        setattr(model, k, v)
                    results = []
                    print(
                        "*" * 50,
                        i
                        / (
                            len(sequence_ap) ** 2
                            * len(params_comb)
                            * len(args.neighbors)
                        ),
                        ctime(),
                        end="\r",
                    )
                    with mp.Pool(3) as pool:
                        for num, fold in enumerate(folds):
                            y_train = Y[fold[0]]
                            y_test = Y[fold[1]]
                            matrix_euc = pd.DataFrame(matrix_euc)
                            max_euc = matrix_euc.iloc[fold[0], fold[0]].values.max()

                            distance_matrix = pd.DataFrame(
                                ah * matrix_h
                                + ap * matrix_p
                                + matrix_euc.divide(max_euc).values
                            )
                            db_datafusion_matrix = pd.DataFrame(
                                ah * matrix_h_df
                                + ap * matrix_p_df
                                + pd.DataFrame(matrix_euc_df).divide(max_euc).values
                            )

                            train_matrix = distance_matrix.iloc[fold[0], fold[0]]
                            test_matrix = distance_matrix.iloc[fold[1], fold[0]]

                            new_X = X_train.copy()
                            new_db_datafusion = db_datafusion.copy()

                            res = pool.apply_async(
                                func,
                                args=(
                                    fold[0],
                                    fold[1],
                                    new_X,
                                    new_db_datafusion,
                                    db_datafusion_matrix,
                                    args.train_label,
                                    args.train_effect,
                                    encoding,
                                    train_matrix,
                                    test_matrix,
                                    y_train,
                                    y_test,
                                    n,
                                    ah,
                                    ap,
                                    num,
                                    model,
                                ),
                            ).get()

                            results.append(res)
                            del res, distance_matrix, train_matrix, test_matrix

                        # results = [res.get() for res in results]
                        # print(results)
                    i = i + 1

                    if (
                        np.mean([results[k]["accs"] for k in range(len(results))])
                        > avg_accs
                    ):
                        best_result["accs"] = np.mean(
                            [results[k]["accs"] for k in range(len(results))]
                        )
                        best_result["f1"] = np.mean(
                            [results[k]["f1"] for k in range(len(results))]
                        )
                        best_result["se_accs"] = sem(
                            [results[k]["accs"] for k in range(len(results))]
                        )
                        best_result["se_f1"] = sem(
                            [results[k]["f1"] for k in range(len(results))]
                        )
                        best_result["model"] = params_comb[j]
                        best_result["neighbors"] = n
                        best_result["ah"] = ah
                        best_result["ap"] = ap
                        avg_accs = np.mean(
                            [results[k]["accs"] for k in range(len(results))]
                        )
                        print(avg_accs, "success!")

        del results
        for k, v in best_result["model"].items():
            setattr(model, k, v)

        train_index = X_train.index
        test_index = X_test.index

        matrix_euc, matrix_h, matrix_p = cal_matrixs(X, X, categorical, non_categorical)
        matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
            X,
            db_datafusion.drop(columns="conc1_mean").copy(),
            categorical,
            non_categorical,
        )

        matrix_euc = pd.DataFrame(matrix_euc)
        max_euc = matrix_euc.iloc[train_index, train_index].values.max()

        matrix = pd.DataFrame(
            best_result["ah"] * matrix_h
            + best_result["ap"] * matrix_p
            + matrix_euc.divide(max_euc).values
        )
        db_datafusion_matrix = pd.DataFrame(
            best_result["ah"] * matrix_h_df
            + best_result["ap"] * matrix_p_df
            + pd.DataFrame(matrix_euc_df).divide(max_euc).values
        )

        del (matrix_euc, matrix_h, matrix_p, matrix_euc_df, matrix_h_df, matrix_p_df)

        simple_rasar_train, simple_rasar_test = cal_data_simple_rasar(
            matrix.iloc[train_index.astype("int64"), train_index.astype("int64")],
            matrix.iloc[test_index.astype("int64"), train_index.astype("int64")],
            Y_train,
            n,
            encoding,
        )

        datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
            train_index,
            test_index,
            X_train,
            X_test,
            db_datafusion,
            db_datafusion_matrix,
            args.train_label,
            args.train_effect,
            encoding,
        )
        del (matrix, db_datafusion_matrix)
        train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
        test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)

        invitro_form = args.invitro_label
        db_invitro = args.db_invitro
        invitro = args.w_invitro

        if invitro == "own":
            train_rf = simple_rasar_train
            test_rf = simple_rasar_test

        if str(db_invitro) == "overlap":
            if (invitro != "False") & (invitro_form == "number"):
                train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
                test_rf["invitro_conc"] = X_test.invitro_conc.reset_index(drop=True)
            elif (invitro != "False") & (invitro_form == "label"):
                train_rf["invitro_label"] = X_train.invitro_label.reset_index(drop=True)
                test_rf["invitro_label"] = X_test.invitro_label.reset_index(drop=True)

            elif (invitro != "False") & (invitro_form == "both"):
                train_rf["ec50"] = X_train.invitro_conc.reset_index(drop=True)
                test_rf["ec50"] = X_test.invitro_conc.reset_index(drop=True)
                train_rf["invitro_label"] = X_train.invitro_label.reset_index(drop=True)
                test_rf["invitro_label"] = X_test.invitro_label.reset_index(drop=True)
            elif (invitro != "False") & (invitro_form == "label_half"):
                train_rf["invitro_label_half"] = X_train.invitro_label_half.reset_index(
                    drop=True
                )
                test_rf["invitro_label_half"] = X_test.invitro_label_half.reset_index(
                    drop=True
                )

            elif (invitro != "False") & (invitro_form == "both_half"):
                train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
                test_rf["invitro_conc"] = X_test.invitro_conc.reset_index(drop=True)
                train_rf["invitro_label_half"] = X_train.invitro_label_half.reset_index(
                    drop=True
                )
                test_rf["invitro_label_half"] = X_test.invitro_label_half.reset_index(
                    drop=True
                )
            elif (invitro != "False") & (invitro_form == "label_reserved"):
                train_rf[
                    "invitro_label_reserved"
                ] = X_train.invitro_label_reserved.reset_index(drop=True)
                test_rf[
                    "invitro_label_reserved"
                ] = X_test.invitro_label_reserved.reset_index(drop=True)

            elif (invitro != "False") & (invitro_form == "both_reserved"):
                train_rf["ec50"] = X_train.invitro_conc.reset_index(drop=True)
                test_rf["ec50"] = X_test.invitro_conc.reset_index(drop=True)
                train_rf[
                    "invitro_label_reserved"
                ] = X_train.invitro_label_reserved.reset_index(drop=True)
                test_rf[
                    "invitro_label_reserved"
                ] = X_test.invitro_label_reserved.reset_index(drop=True)
            elif (invitro != "False") & (invitro_form == "label_half_reserved"):
                train_rf[
                    "invitro_label_half_reserved"
                ] = X_train.invitro_label_half_reserved.reset_index(drop=True)
                test_rf[
                    "invitro_label_half_reserved"
                ] = X_test.invitro_label_half_reserved.reset_index(drop=True)

            elif (invitro != "False") & (invitro_form == "both_half_reserved"):
                train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
                test_rf["invitro_conc"] = X_test.invitro_conc.reset_index(drop=True)
                train_rf[
                    "invitro_label_half_reserved"
                ] = X_train.invitro_label_half_reserved.reset_index(drop=True)
                test_rf[
                    "invitro_label_half_reserved"
                ] = X_test.invitro_label_half_reserved.reset_index(drop=True)

        print(train_rf.columns)
        model.fit(train_rf, Y_train)
        y_pred = model.predict(test_rf)
        test_result = {}
        test_result["neighbors"] = best_result["neighbors"]
        test_result["ah"] = best_result["ah"]
        test_result["ap"] = best_result["ap"]
        test_result["accs"] = accuracy_score(Y_test, y_pred)
        test_result["se_accs"] = best_result["se_accs"]
        test_result["sens"] = recall_score(Y_test, y_pred, average="macro")
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
        test_result["specs"] = tn / (tn + fp)
        test_result["precs"] = precision_score(Y_test, y_pred, average="macro")
        test_result["f1"] = f1_score(Y_test, y_pred, average="macro")
        test_result["se_f1"] = best_result["se_f1"]

        final_result = pd.concat([final_result, pd.DataFrame([test_result])])
        print("finished", test_result["accs"], ctime())
    filename = args.outputFile
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    final_result.to_csv(args.outputFile)

