from helper_model import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import argparse


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument(
        "-e", "--encoding", help="encoding: binary, multiclass", default="binary"
    )
    parser.add_argument(
        "-l",
        "--leaf_ls",
        help="list of leafs number to be chosen in the KNN mdoel",
        required=True,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "-vf",
        "--vitro_file",
        help="whether the input file is about invitro",
        default="False",
    )
    parser.add_argument(
        "-n",
        "--neighbors",
        help="list of neighbor value in the KNN model",
        required=True,
        nargs="+",
        type=int,
    )

    parser.add_argument("-o", "--output", help="outputFile", default="binary.txt")
    return parser.parse_args()


# example:
# python .../KNN.py -i ../lc50_processed.csv  -l 10 30 50 70 90 -n 1 -o .../1nn_bi.txt
# python .../KNN.py -i ../invitro_processed.csv  -l 10 30 50 70 90 -vf True -n 1 -o  .../invitro_1nn_bi.txt

args = getArguments()

if args.vitro_file == "True":
    categorical = ["class", "tax_order", "family", "genus", "species"]

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
    encoding_value=encoding_value,
    seed=42,
)

# splitting into train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -------------------training --------------------
# using 5-fold cross validation to choose the alphas with best accuracy
sequence_alpha = np.logspace(-5, 0, 30)
# sequence_alpha = np.logspace(-2, 0, 50)

print("training start", ctime())
best_ah, best_ap, best_leaf, best_neighbor, best_results = select_alpha(
    X_train,
    Y_train,
    categorical,
    non_categorical,
    sequence_alpha,
    args.leaf_ls,
    args.neighbors,
    encoding,
)
print(ctime())

# -------------------tested on test dataset--------------------

# min max transform the numerical columns
minmax = MinMaxScaler().fit(X_train[non_categorical])
X_train[non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
X_test[non_categorical] = minmax.transform(X_test.loc[:, non_categorical])

matrix = dist_matrix(X_test, X_train, non_categorical, categorical, best_ah, best_ap)
matrix_train = dist_matrix(
    X_train, X_train, non_categorical, categorical, best_ah, best_ap
)
neigh = KNeighborsClassifier(
    n_neighbors=best_neighbor, metric="precomputed", leaf_size=best_leaf
)
neigh.fit(matrix_train, Y_train.astype("int").ravel())
y_pred = neigh.predict(matrix)

# calculate the score
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
    "Accuracy: ",
    accs,
    "se:",
    best_results["se_accs"],
    "\n Sensitivity:",
    sens,
    "se:",
    best_results["se_sens"],
    "\n Specificity",
    specs,
    "se:",
    best_results["se_specs"],
    "\n Precision",
    precs,
    "se:",
    best_results["se_precs"],
    "\n F1 score:",
    f1,
    "se:",
    best_results["se_f1"],
)

# ----------------saving the information into a file-------
info = []
info.append(
    """The best params were alpha_h:{}, alpha_p:{} ,leaf:{},neighbor:{}""".format(
        best_ah, best_ap, best_leaf, best_neighbor
    )
)
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

info.append("The parameters was selected from {}".format("np.logspace(-2, 0, 30)"))
info.append("The leaf was selected from {}".format(args.leaf_ls))


str2file(info, args.output)
