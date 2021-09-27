import pandas as pd
from scipy.spatial.distance import jaccard, pdist, squareform, cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from time import ctime
from tqdm import tqdm
import argparse
from helper_model import *
import multiprocessing as mp
from sklearn.model_selection import train_test_split, ParameterSampler
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description="Running simple RASAR model for invivo datasets to find the best N (neighbor) number."
    )
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument(
        "-e", "--encoding", help="encoding: binary, multiclass", default="binary"
    )
    parser.add_argument(
        "-il",
        "--invitro_label",
        help=" input invitro form: number, label, both, representing using the concentration value\
             of invitro experiment, labeled class value of the invitro experiment, or both",
        default="number",
    )
    parser.add_argument(
        "-wi",
        "--w_invitro",
        help="using the invitro as input or not: True, False, own;\
         representing using invivo plus invitro information as input, using only invivo information as input\
             using only invitro information as input",
        default="False",
    )
    parser.add_argument(
        "-vf",
        "--vitro_file",
        help="whether the input file is about invitro",
        default="False",
    )
    parser.add_argument(
        "-ah", "--alpha_h", help="alpha_hamming", required=True, nargs="?"
    )
    parser.add_argument(
        "-ap", "--alpha_p", help="alpha_pubchem", required=True, nargs="?"
    )
    parser.add_argument(
        "-n",
        "--n_neighbors",
        help="number of neighbors in the RASAR model",
        nargs="?",
        default=1,
        type=int,
    )
    parser.add_argument("-iv", "--invitro", dest="invitro", default="False")

    parser.add_argument(
        "-o", "--output", help="outputFile position", default="binary.txt"
    )
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
    if args.w_invitro == "True":
        db_invitro = "overlap"
    else:
        db_invitro = "noinvitro"

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
    if args.vitro_file:
        categorical = ["class", "tax_order", "family", "genus", "species"]

    print("load data...", ctime())
    if args.encoding == "binary":
        encoding = "binary"
        encoding_value = 1
    elif args.encoding == "multiclass":
        encoding = "multiclass"
        encoding_value = [0.1, 1, 10, 100]

    X, Y = load_data(
        args.inputFile,
        encoding,
        categorical,
        conc_column=conc_column,
        encoding_value=encoding_value,
        seed=42,
    )

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

    if args.w_invitro == "True":
        db_invitro = "overlap"
    else:
        db_invitro = "noinvitro"

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

        del results

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

