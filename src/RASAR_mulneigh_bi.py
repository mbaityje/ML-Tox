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
        description="Running simple RASAR model for invivo datasets to find the best N (neighbor) number."
    )
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-iv", "--invitro", dest="invitro", default="False")
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")

    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()


def func(
    train_index,
    test_index,
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
    train_rf, test_rf = cal_data_simple_rasar(
        dist_matr_train, dist_matr_test, y_train, n_neighbors, args.encoding,
    )

    invitro = args.w_invitro
    invitro_form = args.invitro_label
    db_invitro = args.db_invitro

    if invitro == "own":
        train_rf = pd.DataFrame()
        test_rf = pd.DataFrame()

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
    else:
        if (invitro != "False") & (invitro_form == "number"):
            matrix_invitro_euc = pd.DataFrame(db_invitro_matrix[2])

            db_invitro_matrix_new = pd.DataFrame(
                ah * db_invitro_matrix[0]
                + ap * db_invitro_matrix[1]
                + matrix_invitro_euc.divide(max_euc).values
            )
            dist = np.array(db_invitro_matrix_new.iloc[train_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[train_index, :].idxmin(axis=1))
            conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
            dist = db_invitro_matrix_new.lookup(
                pd.Series(ls).index, pd.Series(ls).values
            )
            train_rf["invitro_conc"] = np.array(conc)
            train_rf["invitro_dist"] = dist

            dist = np.array(db_invitro_matrix_new.iloc[test_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[test_index, :].idxmin(axis=1))
            conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
            dist = db_invitro_matrix_new.lookup(
                pd.Series(ls).index, pd.Series(ls).values
            )
            test_rf["invitro_conc"] = np.array(conc)
            test_rf["invitro_dist"] = dist

        elif (invitro != "False") & (invitro_form == "label"):
            matrix_invitro_euc = pd.DataFrame(db_invitro_matrix[2])

            db_invitro_matrix_new = pd.DataFrame(
                ah * db_invitro_matrix[0]
                + ap * db_invitro_matrix[1]
                + matrix_invitro_euc.divide(max_euc).values
            )

            dist = np.array(db_invitro_matrix_new.iloc[train_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[train_index, :].idxmin(axis=1))
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            dist = db_invitro_matrix_new.lookup(
                pd.Series(ls).index, pd.Series(ls).values
            )
            train_rf["invitro_label"] = np.array(label)
            train_rf["invitro_dist"] = dist

            dist = np.array(db_invitro_matrix_new.iloc[test_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[test_index, :].idxmin(axis=1))
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            dist = db_invitro_matrix_new.lookup(
                pd.Series(ls).index, pd.Series(ls).values
            )
            test_rf["invitro_label"] = np.array(label)
            test_rf["invitro_dist"] = dist

        elif (invitro != "False") & (invitro_form == "both"):
            matrix_invitro_euc = pd.DataFrame(db_invitro_matrix[2])

            db_invitro_matrix_new = pd.DataFrame(
                ah * db_invitro_matrix[0]
                + ap * db_invitro_matrix[1]
                + matrix_invitro_euc.divide(max_euc).values
            )

            dist = np.array(db_invitro_matrix_new.iloc[train_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[train_index, :].idxmin(axis=1))
            conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            train_rf["invitro_conc"] = np.array(conc)
            train_rf["invitro_label"] = np.array(label)
            train_rf["invitro_dist"] = dist

            dist = np.array(db_invitro_matrix_new.iloc[test_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[test_index, :].idxmin(axis=1))
            conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            test_rf["invitro_conc"] = np.array(conc)
            test_rf["invitro_label"] = np.array(label)
            test_rf["invitro_dist"] = dist

        elif (invitro != "False") & (invitro_form == "label_half"):
            matrix_invitro_euc = pd.DataFrame(db_invitro_matrix[2])

            db_invitro_matrix_new = pd.DataFrame(
                ah * db_invitro_matrix[0]
                + ap * db_invitro_matrix[1]
                + matrix_invitro_euc.divide(max_euc).values
            )

            dist = np.array(db_invitro_matrix_new.iloc[train_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[train_index, :].idxmin(axis=1))
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            dist = db_invitro_matrix_new.lookup(
                pd.Series(ls).index, pd.Series(ls).values
            )
            train_rf["invitro_label_half"] = np.array(label)
            train_rf["invitro_dist"] = dist

            dist = np.array(db_invitro_matrix_new.iloc[test_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[test_index, :].idxmin(axis=1))
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            dist = db_invitro_matrix_new.lookup(
                pd.Series(ls).index, pd.Series(ls).values
            )
            test_rf["invitro_label_half"] = np.array(label)
            test_rf["invitro_dist"] = dist
        elif (invitro != "False") & (invitro_form == "both_half"):
            matrix_invitro_euc = pd.DataFrame(db_invitro_matrix[2])

            db_invitro_matrix_new = pd.DataFrame(
                ah * db_invitro_matrix[0]
                + ap * db_invitro_matrix[1]
                + matrix_invitro_euc.divide(max_euc).values
            )

            dist = np.array(db_invitro_matrix_new.iloc[train_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[train_index, :].idxmin(axis=1))
            conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            train_rf["invitro_conc"] = np.array(conc)
            train_rf["invitro_label_half"] = np.array(label)
            train_rf["invitro_dist"] = dist

            dist = np.array(db_invitro_matrix_new.iloc[test_index, :].min(axis=1))
            ls = np.array(db_invitro_matrix_new.iloc[test_index, :].idxmin(axis=1))
            conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
            label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
            test_rf["invitro_conc"] = np.array(conc)
            test_rf["invitro_label_half"] = np.array(label)
            test_rf["invitro_dist"] = dist

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
    conc_column = "conc1_mean"

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

    if args.invitro == "both":
        categorical = ["class", "tax_order", "family", "genus", "species"]
    elif args.invitro == "eawag":
        categorical = [
            "class",
            "tax_order",
            "family",
            "genus",
            "species",
            "cell_line",
            "endpoint",
            "solvent",
            "conc_determination_nominal_or_measured",
        ]
        conc_column = "ec50"
    elif args.invitro == "toxcast":
        categorical = [
            "class",
            "tax_order",
            "family",
            "genus",
            "species",
            # "modl"
        ]
        conc_column = "conc"

    X, Y = load_data(
        args.inputFile,
        args.encoding,
        categorical,
        conc_column=conc_column,
        encoding_value=1,
        seed=42,
    )
    # X = X[:200]
    # Y = Y[:200]
    print(args.db_invitro)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("calcultaing distance matrix..", ctime())

    matrix_euc, matrix_h, matrix_p = cal_matrixs(
        X_train, X_train, categorical, non_categorical
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
        "max_depth": [i for i in range(10, 30, 5)],
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    }

    params_comb = list(ParameterSampler(hyper_params_tune, n_iter=20, random_state=2))
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
                    # mp.cpu_count()-1
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
                    with mp.Pool(4) as pool:
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

                            train_matrix = distance_matrix.iloc[fold[0], fold[0]]
                            test_matrix = distance_matrix.iloc[fold[1], fold[0]]
                            res = pool.apply_async(
                                func,
                                args=(
                                    fold[0],
                                    fold[1],
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
                            )

                            results.append(res)
                            del res, distance_matrix, train_matrix, test_matrix
                        results = [res.get() for res in results]
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
                        # print(a)

        del results
        # with open(args["outputFile"] + f"{n}.pkl", "wb") as f:
        #     pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
        print(best_result, ctime())
        for k, v in best_result["model"].items():
            setattr(model, k, v)

        minmax = MinMaxScaler().fit(X_train[non_categorical])
        X_train[non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
        X_test[non_categorical] = minmax.transform(X_test.loc[:, non_categorical])
        train_index = X_train.index
        test_index = X_test.index
        matrix_test = dist_matrix(
            X_test,
            X_train,
            non_categorical,
            categorical,
            best_result["ah"],
            best_result["ap"],
        )
        matrix_train = dist_matrix(
            X_train,
            X_train,
            non_categorical,
            categorical,
            best_result["ah"],
            best_result["ap"],
        )

        train_rf, test_rf = cal_data_simple_rasar(
            matrix_train, matrix_test, Y_train, best_result["neighbors"], args.encoding
        )
        invitro_form = args.invitro_label
        db_invitro = args.db_invitro
        invitro = args.w_invitro

        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()

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

        print(train_rf.columns, ctime())
        model.fit(train_rf, Y_train)
        y_pred = model.predict(test_rf)
        test_result = {}
        test_result["neighbors"] = best_result["neighbors"]
        test_result["ah"] = best_result["ah"]
        test_result["ap"] = best_result["ap"]
        test_result["accs"] = accuracy_score(Y_test, y_pred)
        test_result["se_accs"] = best_result["se_accs"]
        test_result["sens"] = recall_score(Y_test, y_pred, average="macro")
        try:
            tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
            test_result["specs"] = tn / (tn + fp)
        except:
            pass
        test_result["precs"] = precision_score(Y_test, y_pred, average="macro")
        test_result["f1"] = f1_score(Y_test, y_pred, average="macro")
        test_result["se_f1"] = best_result["se_f1"]
        final_result = pd.concat([final_result, pd.DataFrame([test_result])])

    filename = args.outputFile
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    final_result.to_csv(args.outputFile)

