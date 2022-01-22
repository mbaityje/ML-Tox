from helpers.helper_model import *
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
import argparse


def getArguments():
    parser = argparse.ArgumentParser(
        description="Running DataFusion RASAR model for invivo datasets or merged invivo & invitro dataset."
    )
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument(
        "-idf", "--input_df", help="input datafusion File position", required=True
    )
    parser.add_argument(
        "-ce",
        "--cate_encoding",
        help="methods to encoding the categorical features",
        default="ordinal",
    )
    parser.add_argument("-e", "--encoding", help="encoding", default="binary")
    parser.add_argument("-if", "--input_info", help="input information", default="cte")
    parser.add_argument(
        "-ah", "--alpha_h", help="alpha_hamming", default=False, nargs="?"
    )
    parser.add_argument(
        "-ap", "--alpha_p", help="alpha_pubchem", default=False, nargs="?"
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
        "-endpoint",
        "--train_endpoint",
        help="train_endpoint",
        default="['LC50','EC50']",
    )
    parser.add_argument("-effect", "--train_effect", help="train_effect", default="MOR")
    parser.add_argument(
        "-o", "--output", help="outputFile", default="results/df_rasar_bi.txt"
    )
    return parser.parse_args()


# example:
# python .../RASAR_df.py -i .../lc50_processed.csv  -idf  .../datafusion.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.1 -ap 0.1 -o df_rasar.txt

args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# -------------------loading data & preprocessing--------------------
print("loading dataset...", ctime())

db_mortality, db_datafusion = load_datafusion_datasets(
    args.input,
    args.input_df,
    categorical_columns=categorical,
    label=args.input_info,
    cate_encoding=args.cate_encoding,
    encoding=encoding,
    encoding_value=encoding_value,
)


X = db_mortality.drop(columns="conc1_mean").copy()
Y = db_mortality.conc1_mean.values

X_trainvalid, X_test, Y_trainvalid, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


if args.alpha_h:
    print("calcultaing distance matrixes..", ctime())
    matrix_euc, matrix_h, matrix_p = cal_matrixs(
        X_trainvalid, X_trainvalid, categorical, numerical
    )
    # print("calcultaing datafusion distance matrix..", ctime())
    matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
        X_trainvalid,
        db_datafusion.drop(columns="conc1_mean").copy(),
        categorical,
        numerical,
    )
    print("distance matrixes successfully calculated!", ctime())
else:
    comparing = np.setdiff1d(
        db_datafusion.columns,
        numerical
        + ["pub" + str(i) for i in range(1, 882)]
        + ["endpoint", "effect", "conc1_mean"],
    )
    df_rasar_label_train, df_rasar_label_test = woalphas_label_df_rasar(
        X_trainvalid, X_test, db_datafusion, comparing
    )
    X_trainvalid = X_trainvalid.drop(columns="test_cas")
    X_test = X_test.drop(columns="test_cas")

# -------------------training --------------------
if encoding == "binary":
    model = RandomForestClassifier(random_state=10)
    hyper_params_tune = {
        "max_depth": [i for i in range(10, 30, 6)],
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    }
elif encoding == "multiclass":
    h2o.init()
    h2o.no_progress()
    model = H2ORandomForestEstimator(seed=10)
    hyper_params_tune = {
        "ntrees": [i for i in range(10, 1000, 10)],
        "max_depth": [i for i in range(10, 1000, 10)],
        "min_rows": [1, 10, 100, 1000],
        "sample_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }

params_comb = list(ParameterSampler(hyper_params_tune, n_iter=20, random_state=2))
# n_iter=20, random_state=2

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

    for k, v in params_comb[i].items():
        setattr(model, k, v)
    if args.alpha_h:
        result = model_df_rasar(
            matrix_euc,
            matrix_h,
            matrix_p,
            matrix_euc_df,
            matrix_h_df,
            matrix_p_df,
            ah=float(args.alpha_h),
            ap=float(args.alpha_p),
            X=X_trainvalid,
            Y=Y_trainvalid,
            db_datafusion=db_datafusion,
            train_endpoint=args.train_endpoint,
            train_effect=args.train_effect,
            model=model,
            n_neighbors=args.n_neighbors,
            encoding=encoding,
        )
    else:
        result = woalphas_model_df_rasar(
            X_trainvalid,
            Y_trainvalid,
            db_datafusion,
            model,
            df_rasar_label_train,
            encoding,
        )
    if np.mean(result.accuracy) > best_accs:
        best_p = params_comb[i]
        best_accs = np.mean(result.accuracy)
        best_result = result
df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()

# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_p.items():
    setattr(model, k, v)

train_index = X_trainvalid.index
test_index = X_test.index

if args.alpha_h:
    matrix_euc, matrix_h, matrix_p = cal_matrixs(X, X, categorical, numerical)
    matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
        X, db_datafusion.drop(columns="conc1_mean").copy(), categorical, numerical
    )

    matrix_euc = pd.DataFrame(matrix_euc)
    max_euc = matrix_euc.iloc[train_index, train_index].values.max()

    matrix = pd.DataFrame(
        float(args.alpha_h) * matrix_h
        + float(args.alpha_p) * matrix_p
        + matrix_euc.divide(max_euc).values
    )
    db_datafusion_matrix = pd.DataFrame(
        float(args.alpha_h) * matrix_h_df
        + float(args.alpha_p) * matrix_p_df
        + pd.DataFrame(matrix_euc_df).divide(max_euc).values
    )

    s_rasar_train, s_rasar_test = cal_s_rasar(
        matrix.iloc[train_index.astype("int64"), train_index.astype("int64")],
        matrix.iloc[test_index.astype("int64"), train_index.astype("int64")],
        Y_trainvalid,
        args.n_neighbors,
        encoding,
    )

    df_rasar_train, df_rasar_test = cal_df_rasar(
        train_index,
        test_index,
        X_trainvalid,
        X_test,
        db_datafusion,
        db_datafusion_matrix,
        args.train_endpoint,
        args.train_effect,
        encoding,
    )

    train_rasar = pd.concat([s_rasar_train, df_rasar_train], axis=1)
    test_rasar = pd.concat([s_rasar_test, df_rasar_test], axis=1)


else:
    minmax = MinMaxScaler().fit(X_trainvalid[numerical])

    X_trainvalid[numerical] = minmax.transform(X_trainvalid.loc[:, numerical])
    X_test[numerical] = minmax.transform(X_test.loc[:, numerical])
    db_datafusion[numerical] = minmax.transform(db_datafusion.loc[:, numerical])

    s_rasar_train, s_rasar_test = woalphas_cal_s_rasar(
        X_trainvalid, X_test, Y_trainvalid, Y_test, encoding
    )

    df_rasar_train, df_rasar_test = woalphas_cal_df_rasar(
        X_trainvalid, X_test, db_datafusion
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
    df_test_score = fit_and_predict(
        model, train_rasar, Y_trainvalid, test_rasar, Y_test, encoding
    )
elif encoding == "multiclass":

    train_rasar.loc[:, "target"] = Y_trainvalid
    test_rasar.loc[:, "target"] = Y_test

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

    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    specs = np.nan
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")

    model.train(y="target", training_frame=train_rasar_h2o)
    y_pred = model.predict(test_rasar_h2o).as_data_frame()["predict"]

    df_test_score = pd.DataFrame()
    df_test_score.loc[0, "accuracy"] = accuracy_score(Y_test, y_pred)
    df_test_score.loc[0, "recall"] = recall_score(Y_test, y_pred, average="macro")
    df_test_score.loc[0, "specificity"] = np.nan
    df_test_score.loc[0, "f1"] = f1_score(Y_test, y_pred, average="macro")
    df_test_score.loc[0, "precision"] = precision_score(Y_test, y_pred, average="macro")

df_output = pd.concat(
    [df_mean, df_std, df_test_score],
    keys=["train_mean", "train_std", "test"],
    names=["series_name"],
)
df_output["model"] = str(best_p)

# ----------------save the information into a file-------
df2file(df_output, args.output)
