from helper_model import *
from scipy.spatial.distance import cdist, pdist, squareform
from collections import Counter
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description="Simple RASAR model for invivo dataset or merged invivo & invitro dataset."
    )
    parser.add_argument("-i1", "--input", dest="inputFile", required=True)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True, nargs="?")
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True, nargs="?")
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]


if args.invitroFile == "True":
    categorical = ["class", "tax_order", "family", "genus", "species"]


X, Y = load_data(
    args.inputFile,
    encoding=encoding,
    categorical_columns=categorical,
    encoding_value=encoding_value,
    seed=42,
)
# X = X[:200]
# Y = Y[:200]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print("calcultaing distance matrix..", ctime())

matrix_euc, matrix_h, matrix_p = cal_matrixs(
    X_train, X_train, categorical, non_categorical
)

print("distance matrix calculation finished", ctime())
hyper_params_tune = {
    "max_depth": [i for i in range(10, 30, 6)],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32],
}

params_comb = list(ParameterSampler(hyper_params_tune, n_iter=30, random_state=2))

best_accs = 0
best_p = dict()

if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]
j = 1
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in range(0, len(params_comb)):
            print(
                "*" * 50,
                j / (len(sequence_ap) ** 2 * len(params_comb)),
                ctime(),
                end="\r",
            )
            j = j + 1
            model = RandomForestClassifier(random_state=10)
            # model = LogisticRegression(random_state=10)
            for k, v in params_comb[i].items():
                setattr(model, k, v)

            best_results = RASAR_simple(
                matrix_euc,
                matrix_h,
                matrix_p,
                ah,
                ap,
                Y_train,
                X_train,
                db_invitro_matrix="noinvitro",
                n_neighbors=args.n_neighbors,
                invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                db_invitro=args.db_invitro,
                encoding=encoding,
                model=model,
            )

            if best_results["avg_accs"] > best_accs:
                best_p = params_comb[i]
                best_accs = best_results["avg_accs"]
                best_results = best_results

# -------------------tested on test dataset--------------------
for k, v in best_p.items():
    setattr(model, k, v)


minmax = MinMaxScaler().fit(X_train[non_categorical])
X_train[non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
X_test[non_categorical] = minmax.transform(X_test.loc[:, non_categorical])

matrix_test = dist_matrix(
    X_test, X_train, non_categorical, categorical, args.alpha_h, args.alpha_p
)
matrix_train = dist_matrix(
    X_train, X_train, non_categorical, categorical, args.alpha_h, args.alpha_p
)

train_rf, test_rf = cal_data_simple_rasar(
    matrix_train, matrix_test, Y_train, args.n_neighbors, encoding
)
invitro_form = args.invitro_label
db_invitro = args.db_invitro
invitro = args.w_invitro

train_index = X_train.index
test_index = X_test.index


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
        best_results["se_accs"],
        sens,
        best_results["se_sens"],
        specs,
        best_results["se_specs"],
        precs,
        best_results["se_precs"],
        f1,
        best_results["se_f1"],
    )
)

info = []
info.append(
    """Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        best_results["se_accs"],
        sens,
        best_results["se_sens"],
        specs,
        best_results["se_specs"],
        precs,
        best_results["se_precs"],
        f1,
        best_results["se_f1"],
    )
)
info.append("Alpha_h:{}, Alpha_p: {}".format(args.alpha_h, args.alpha_p))

filename = args.outputFile
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)

with open(filename, "w") as file_handler:
    for item in info:
        file_handler.write("{}\n".format(item))
