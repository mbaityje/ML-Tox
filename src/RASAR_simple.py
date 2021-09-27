from helper_model import *
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.ensemble import RandomForestClassifier
import argparse


def getArguments():
    parser = argparse.ArgumentParser(
        description="Simple RASAR model for invivo dataset or merged invivo & invitro dataset."
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
    parser.add_argument(
        "-o", "--output", help="outputFile position", default="binary.txt"
    )
    return parser.parse_args()


# example:
# python .../RASAR_simple.py -i .../lc50_processed.csv    -ah 0.1 -ap 0.1 -o .../s_rasar.txt
# python .../RASAR_simple.py -i .../lc50_processed_w_invitro.csv -il label -wi True -ah 0.1 -ap 0.1 -o .../s_rasar_bi_invitro_label.txt

args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# -------------------loading data & preprocessing--------------------
X, Y = load_data(
    args.input,
    encoding=encoding,
    categorical_columns=categorical,
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

if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]

if args.w_invitro == "True":
    db_invitro = "overlap"
else:
    db_invitro = "noinvitro"


# -------------------training --------------------
hyper_params_tune = {
    "max_depth": [i for i in range(10, 30, 6)],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32],
}
params_comb = list(ParameterSampler(hyper_params_tune, n_iter=30, random_state=2))


best_accs = 0
best_p = dict()

count = 1
print("training process start..", ctime())
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in range(0, len(params_comb)):
            print(
                "*" * 50,
                count / (len(sequence_ap) ** 2 * len(params_comb)),
                ctime(),
                end="\r",
            )
            count = count + 1
            model = RandomForestClassifier(random_state=10)
            # model = LogisticRegression(random_state=10)
            for k, v in params_comb[i].items():
                setattr(model, k, v)

            results = RASAR_simple(
                matrix_euc,
                matrix_h,
                matrix_p,
                ah,
                ap,
                Y_train,
                X_train,
                db_invitro_matrix="nan",
                n_neighbors=args.n_neighbors,
                w_invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                db_invitro=db_invitro,
                encoding=encoding,
                model=model,
            )

            if results["avg_accs"] > best_accs:
                best_p = params_comb[i]
                best_accs = results["avg_accs"]
                results = results
                best_ah = ah
                best_ap = ap
                print("success.", best_accs)


# -------------------tested on test dataset--------------------
print("testing start.", ctime())
for k, v in best_p.items():
    setattr(model, k, v)


minmax = MinMaxScaler().fit(X_train[non_categorical])
X_train[non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
X_test[non_categorical] = minmax.transform(X_test.loc[:, non_categorical])

matrix_test = dist_matrix(
    X_test, X_train, non_categorical, categorical, best_ah, best_ap
)
matrix_train = dist_matrix(
    X_train, X_train, non_categorical, categorical, best_ah, best_ap
)

train_rf, test_rf = cal_data_simple_rasar(
    matrix_train, matrix_test, Y_train, args.n_neighbors, encoding
)

invitro_form = args.invitro_label
invitro = args.w_invitro

train_index = X_train.index
test_index = X_test.index

# adding invitro information or not
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

model.fit(train_rf, Y_train)
y_pred = model.predict(test_rf)

if encoding == "binary":

    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
    specs = tn / (tn + fp)
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")

elif encoding == "multiclass":
    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    specs = np.nan
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")


print(
    """Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        results["se_accs"],
        sens,
        results["se_sens"],
        specs,
        results["se_specs"],
        precs,
        results["se_precs"],
        f1,
        results["se_f1"],
    )
)
# ----------------save the information into a file-------
info = []
info.append(
    """Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        results["se_accs"],
        sens,
        results["se_sens"],
        specs,
        results["se_specs"],
        precs,
        results["se_precs"],
        f1,
        results["se_f1"],
    )
)
info.append(
    "alpha_h:{}, alpha_p: {}, hyperpatameters:{}".format(best_ah, best_ap, best_p)
)

str2file(info, args.output)
