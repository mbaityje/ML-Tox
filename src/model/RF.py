from helpers.helper_model import *
import argparse


def getArguments():
    parser = argparse.ArgumentParser(
        description="Running Random forest models for datasets."
    )
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument("-if", "--input_info", help="input information", default="cte")
    parser.add_argument(
        "-ce",
        "--cate_encoding",
        help="methods to encoding the categorical features",
        default="onehot",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        help="Classification type: binary, multiclass",
        default="binary",
    )
    parser.add_argument(
        "-o", "--output", help="outputFile", default="results/rf_bi.txt"
    )
    return parser.parse_args()


# example:
# python .../RF.py -i ../lc_db_processed.csv   -o .../rf_bi.txt

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
    categorical_columns=categorical,
    label=args.input_info,
    cate_encoding=args.cate_encoding,
    encoding_value=encoding_value,
    seed=42,
)

# X = X.drop(columns=["test_cas"])

# splitting into train and test dataset

X_trainvalid, X_test, Y_trainvalid, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -------------------training --------------------

hyper_params_tune = {
    "max_depth": [i for i in range(10, 20, 2)],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32],
}
params_comb = list(ParameterSampler(hyper_params_tune, n_iter=100, random_state=2))
# n_iter=100

best_accs = 0
best_p = dict()

print("training start", ctime())
count = 0
for i in range(0, len(params_comb)):
    print(
        "*" * 50,
        count / len(params_comb),
        ctime(),
        end="\r",
    )
    count = count + 1

    model = RandomForestClassifier(random_state=10)

    for k, v in params_comb[i].items():
        setattr(model, k, v)
    result = cv_train_model(X_trainvalid, Y_trainvalid, model, encoding)

    if np.mean(result.accuracy) > best_accs:
        best_p = params_comb[i]
        best_accs = np.mean(result.accuracy)
        best_result = result

df_mean = pd.DataFrame(best_result.mean(axis=0)).transpose()
df_std = pd.DataFrame(best_result.sem(axis=0)).transpose()

# -------------------tested on test dataset--------------------

for k, v in best_p.items():
    setattr(model, k, v)

minmax = MinMaxScaler().fit(X_trainvalid[numerical])
new_train = X_trainvalid.copy()
new_test = X_test.copy()
new_train[numerical] = minmax.transform(X_trainvalid.loc[:, numerical])
new_test[numerical] = minmax.transform(X_test.loc[:, numerical])

model.fit(new_train, Y_trainvalid)
y_pred = model.predict(new_test)

df_test_score = fit_and_predict(
    model, new_train, Y_trainvalid, new_test, Y_test, encoding
)

df_output = pd.concat(
    [df_mean, df_std, df_test_score],
    keys=["train_mean", "train_std", "test"],
    names=["series_name"],
)

df_output["model"] = str(best_p)

df2file(df_output, args.output)
