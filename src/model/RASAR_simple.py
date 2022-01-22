from helpers.helper_model import *
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.ensemble import RandomForestClassifier
import argparse


def getArguments():
    parser = argparse.ArgumentParser(
        description="Simple RASAR model for invivo dataset or merged invivo & invitro dataset."
    )
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument("-if", "--input_info", help="input information", default="cte")
    parser.add_argument(
        "-ce",
        "--cate_encoding",
        help="methods to encoding the categorical features",
        default="ordinal",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        help="Classification type: binary, multiclass",
        default="binary",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model: logistic regression, random forest",
        default="rf",
    )
    parser.add_argument(
        "-ah",
        "--alpha_h",
        help="alpha_hamming",
        default=False,
        nargs="?",
        type=float,
    )
    parser.add_argument(
        "-ap", "--alpha_p", help="alpha_pubchem", default=False, nargs="?", type=float
    )
    parser.add_argument(
        "-n",
        "--n_neighbors",
        help="number of neighbors in the RASAR model",
        nargs="?",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-o", "--output", help="outputFile position", default="s_rasar_bi.txt"
    )
    return parser.parse_args()


# example:
# python .../RASAR_simple.py -i1 .../lc50_processed.csv    -ah 0.1 -ap 0.1 -o .../s_rasar.txt


args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# -------------------loading data & preprocessing--------------------
print("loading dataset...", ctime())
X, Y = load_data(
    args.input,
    encoding=encoding,
    categorical_columns=categorical,
    cate_encoding=args.cate_encoding,
    label=args.input_info,
    encoding_value=encoding_value,
    seed=42,
)


# X = X.drop(columns=["test_cas"])

X_trainvalid, X_test, Y_trainvalid, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

if args.alpha_h:
    print("calculating distance matrix..", ctime())
    matrix_euc, matrix_h, matrix_p = cal_matrixs(
        X_trainvalid, X_trainvalid, categorical, numerical
    )
    print("distance matrix calculation finished", ctime())


# -------------------training --------------------
if args.model == "rf":
    hyper_params_tune = {
        "max_depth": [i for i in range(10, 30, 6)],
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    }
elif args.model == "lr":
    hyper_params_tune = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "penalty": ["elasticnet"],
        "max_iter": list(range(100, 800, 100)),
        "l1_ratio": [int(i) / 10 for i in range(0, 11, 1)],
        "solver": ["saga"],
        "fit_intercept": [True, False],
    }
params_comb = list(ParameterSampler(hyper_params_tune, n_iter=50, random_state=2))
# n_iter = 50


best_accs = 0
best_p = dict()

count = 0
print("training process start..", ctime())

for i in range(0, len(params_comb)):
    print(
        "*" * 50,
        count / len(params_comb),
        ctime(),
        end="\r",
    )
    count = count + 1

    if args.model == "rf":
        model = RandomForestClassifier(random_state=10)
    elif args.model == "lr":
        model = LogisticRegression(random_state=10)
    for k, v in params_comb[i].items():
        setattr(model, k, v)

    if args.alpha_h:
        result = model_s_rasar(
            matrix_euc,
            matrix_h,
            matrix_p,
            float(args.alpha_h),
            float(args.alpha_p),
            Y_trainvalid,
            X_trainvalid,
            n_neighbors=args.n_neighbors,
            encoding=encoding,
            model=model,
        )
    else:
        result = woalphas_model_s_rasar(X_trainvalid, Y_trainvalid, model, encoding)

    if np.mean(result.accuracy) > best_accs:
        best_p = params_comb[i]
        best_accs = np.mean(result.accuracy)
        best_result = result

df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()

# -------------------tested on test dataset--------------------
print("testing start.", ctime())
for k, v in best_p.items():
    setattr(model, k, v)


minmax = MinMaxScaler().fit(X_trainvalid[numerical])
X_trainvalid[numerical] = minmax.transform(X_trainvalid.loc[:, numerical])
X_test[numerical] = minmax.transform(X_test.loc[:, numerical])


if args.alpha_h:
    matrix_test = dist_matrix(
        X_test, X_trainvalid, numerical, categorical, args.alpha_h, args.alpha_p
    )
    matrix_train = dist_matrix(
        X_trainvalid, X_trainvalid, numerical, categorical, args.alpha_h, args.alpha_p
    )
    train_rasar, test_rasar = cal_s_rasar(
        matrix_train, matrix_test, Y_trainvalid, args.n_neighbors, encoding
    )
    df_test_score = fit_and_predict(
        model, train_rasar, Y_trainvalid, test_rasar, Y_test, encoding
    )
# model.fit(train_rasar, Y_trainvalid)
# y_pred = model.predict(test_rasar)
else:
    train_rasar, test_rasar = woalphas_cal_s_rasar(
        X_trainvalid, X_test, Y_trainvalid, Y_test, args.encoding
    )

    df_test_score = fit_and_predict(
        model,
        train_rasar[train_rasar.filter(like="dist").columns],
        Y_trainvalid,
        test_rasar[test_rasar.filter(like="dist").columns],
        Y_test,
        encoding,
    )


df_output = pd.concat(
    [df_mean, df_std, df_test_score],
    keys=["train_mean", "train_std", "test"],
    names=["series_name"],
)
df_output["model"] = str(best_p)

# ----------------save the information into a file-------
df2file(df_output, args.output)
