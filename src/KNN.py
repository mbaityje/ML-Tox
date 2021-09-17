from helper_model import (
    recall_score,
    precision_score,
    accuracy_score,
    mean_squared_error,
    f1_score,
    categorical,
    non_categorical,
    train_test_split,
    KNeighborsClassifier,
    load_data,
    select_alpha,
    dist_matrix,
    MinMaxScaler,
    confusion_matrix,
)
import numpy as np
from time import ctime
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
        "-l", "--leaf_ls", dest="leaf_list", required=True, nargs="+", type=int
    )
    parser.add_argument("-invitro", "--invitro", dest="invitro_file", default=False)
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()

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
    "invitro_label",
]

if args.invitro:
    categorical = ["class", "tax_order", "family", "genus", "species"]
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
    "MorganDensity",
    "LogP",
    "mol_weight",
    "water_solubility",
    "melting_point",
]


if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# loading data & splitting into train and test dataset
print("loading dataset...", ctime())
X, Y = load_data(
    args.inputFile,
    encoding=encoding,
    categorical_columns=categorical,
    encoding_value=encoding_value,
    seed=42,
)
# X = X[:1000]
# Y = Y[:1000]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# using 5-fold cross validation to choose the alphas with best accuracy

sequence_alpha = np.logspace(-5, 0, 30)
# sequence_alpha = np.logspace(-2, 0, 50)

print(ctime())
best_alpha_h, best_alpha_p, best_leaf, best_neighbor, best_results = select_alpha(
    X_train,
    sequence_alpha,
    Y_train,
    categorical,
    non_categorical,
    args.leaf_list,
    args.neighbors,
    encoding,
)
print(ctime())

# validate on the test dataset
minmax = MinMaxScaler().fit(X_train[non_categorical])
X_train[non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
X_test[non_categorical] = minmax.transform(X_test.loc[:, non_categorical])

matrix = dist_matrix(
    X_test, X_train, non_categorical, categorical, best_alpha_h, best_alpha_p
)
matrix_train = dist_matrix(
    X_train, X_train, non_categorical, categorical, best_alpha_h, best_alpha_p
)
neigh = KNeighborsClassifier(
    n_neighbors=best_neighbor, metric="precomputed", leaf_size=best_leaf
)
neigh.fit(matrix_train, Y_train.astype("int").ravel())
y_pred = neigh.predict(matrix)

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

# saving the information into a file
info = []
info.append(
    """The best params were alpha_h:{}, alpha_p:{} ,leaf:{},neighbor:{}""".format(
        best_alpha_h, best_alpha_p, best_leaf, best_neighbor
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
info.append("The leaf was selected from {}".format(args.leaf_list))

filename = args.outputFile
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)
with open(filename, "w") as file_handler:
    for item in info:
        file_handler.write("{}\n".format(item))
