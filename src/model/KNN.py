from helpers.helper_model import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import argparse


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument(
        "-e",
        "--encoding",
        help="Classification type: binary, multiclass",
        default="binary",
    )
    parser.add_argument("-if", "--input_info", help="input information", default="cte")
    parser.add_argument(
        "-ce",
        "--cate_encoding",
        help="methods to encoding the categorical features",
        default="ordinal",
    )
    parser.add_argument(
        "-l",
        "--leaf_ls",
        help="list of leafs number to be chosen in the KNN mdoel",
        default=10,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "-k",
        "--k_num",
        help="list of neighbor value in the KNN model",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-o", "--output", help="outputFile", default="results/knn_bi.txt"
    )
    return parser.parse_args()


# example:
# python .../KNN.py -i ../lc50_processed.csv  -l 10 30 50 70 90 -n 1 -o .../1nn_bi.txt

args = getArguments()

if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]


# -------------------loading data--------------------
print("loading dataset...", ctime())
X, Y = load_data(
    args.input,
    encoding=encoding,
    label=args.input_info,
    cate_encoding=args.cate_encoding,
    categorical_columns=categorical,
    encoding_value=encoding_value,
    seed=42,
)

# splitting into train and test dataset
X_trainvalid, X_test, Y_trainvalid, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -------------------training --------------------
print("training start", ctime())
if args.k_num:
    # using 5-fold cross validation to choose the alphas with best accuracy
    sequence_alpha = np.logspace(-5, 0, 30)
    # sequence_alpha = np.logspace(-5, 0, 3)

    best_ah, best_ap, best_leaf, best_result = select_alpha(
        X_trainvalid,
        Y_trainvalid,
        categorical,
        numerical,
        sequence_alpha,
        args.leaf_ls,
        args.k_num,
        encoding,
    )

    df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
    df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()
else:
    k_ls = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 31]
    # k_ls = [1, 2, 3, 4, 5]
    df_knn_train = pd.DataFrame()
    for i in k_ls:
        model = KNeighborsClassifier(n_neighbors=i)
        result = cv_train_model(X_trainvalid, Y_trainvalid, model, encoding)

        df_mean = pd.DataFrame(result.mean(axis=0)).transpose()
        df_std = pd.DataFrame(result.std(axis=0)).transpose()
        df_mean.loc[0, "K"] = i
        df_std.loc[0, "K"] = i

        df_train_score = pd.concat(
            [df_mean, df_std], keys=["train_mean", "train_std"], names=["series_name"]
        ).reset_index(level=1, drop=True)
        df_knn_train = pd.concat([df_knn_train, df_train_score], axis=0)
print("training ends.", ctime())

# -------------------tested on test dataset--------------------
print("testing start.", ctime())
# min max transform the numerical columns
minmax = MinMaxScaler().fit(X_trainvalid[numerical])
temp_train = X_trainvalid.copy()
temp_test = X_test.copy()
temp_train[numerical] = minmax.transform(temp_train.loc[:, numerical])
temp_test[numerical] = minmax.transform(temp_test.loc[:, numerical])


if args.k_num:
    matrix_test = dist_matrix(
        temp_test, temp_train, numerical, categorical, best_ah, best_ap
    )
    matrix_train = dist_matrix(
        temp_train, temp_train, numerical, categorical, best_ah, best_ap
    )
    model = KNeighborsClassifier(
        n_neighbors=args.k_num, metric="precomputed", leaf_size=best_leaf
    )
    # model.fit(matrix_train, Y_trainvalid.astype("int").ravel())
    # y_pred = model.predict(matrix_test)
    df_knn_test = fit_and_predict(
        model,
        matrix_train,
        Y_trainvalid.astype("int").ravel(),
        matrix_test,
        Y_test,
        encoding,
    )
    df_knn = pd.concat(
        [df_mean, df_std, df_knn_test],
        keys=["train_mean", "train_std", "test"],
        names=["series_name"],
    ).reset_index(level=1, drop=True)

    df_knn["best_ah"], df_knn["best_ap"], df_knn["best_leaf"] = (
        best_ah,
        best_ap,
        best_leaf,
    )
else:
    df_knn_test = pd.DataFrame()
    for i in k_ls:
        model = KNeighborsClassifier(n_neighbors=i)
        df_test_score = fit_and_predict(
            model, temp_train, Y_trainvalid, temp_test, Y_test, encoding
        )
        df_test_score.loc[0, "K"] = i

        df_knn_test = pd.concat([df_knn_test, df_test_score], axis=0).reset_index(
            drop=True
        )

    df_knn_test["series_name"] = "test"
    df_knn_test.set_index("series_name", inplace=True)
    df_knn = pd.concat([df_knn_train, df_knn_test])


# ----------------save the information into a file-------
df2file(df_knn, args.output)
