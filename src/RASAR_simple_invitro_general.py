from helper_model import *
from sklearn.model_selection import train_test_split, ParameterSampler
import argparse


def getArguments():
    parser = argparse.ArgumentParser(
        description="Simple rasar model on invivo dataset with adding one invitro dataset."
    )
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument(
        "-iv", "--input_vitro", help="input invitro file position", required=True
    )
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
        "-ths",
        "--threshold",
        help="threshold for invitro experiment classification",
        required=True,
        nargs="?",
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

# python .../RASAR_simple_invitro_general.py -i .../lc50_processed.csv -iv .../invitro.csv -il label -wi "True" -ah 0.1 -ap 0.1 -o .../s_rasar_vitro_label_general.txt

args = getArguments()


# -------------------loading data & preprocessing--------------------
args = getArguments()

if args.encoding == "binary":
    X, Y, db_invitro = load_invivo_invitro(
        args.input, args.input_vitro, "binary", encoding_value=1, seed=42
    )
    db_invitro["invitro_label"] = np.where(
        db_invitro["invitro_conc"].values > args.ths, 0, 1
    )
elif args.encoding == "multiclass":
    X, Y, db_invitro = load_invivo_invitro(
        args.input,
        args.input_vitro,
        "multiclass",
        encoding_value=[0.1, 1, 10, 100],
        seed=42,
    )
    db_invitro["invitro_label"] = multiclass_encoding(
        db_invitro["invitro_conc"], [args.ths[0], args.ths[1], args.ths[2], args.ths[3]]
    )


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print("calcultaing distance matrix..", ctime())

matrix_euc, matrix_h, matrix_p = cal_matrixs(
    X_train, X_train, categorical, non_categorical
)

matrix_euc_x_invitro, matrix_h_x_invitro, matrix_p_x_invitro = cal_matrixs(
    X_train, db_invitro, categorical_both, non_categorical
)

print("successfully calculated distance matrix..", ctime())

if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]


# -------------------training --------------------
hyper_params_tune = {
    "max_depth": [i for i in range(10, 20, 2)],
    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=2)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

params_comb = list(ParameterSampler(hyper_params_tune, n_iter=30, random_state=2))

best_accs = 0
best_p = dict()

j = 1
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in tqdm(range(0, len(params_comb))):
            # print("*" * 50, j/(len(sequence_ap)**2 * len(params_comb) ), ctime(), end="\r")
            j = j + 1
            model = RandomForestClassifier(random_state=10)
            # model = LogisticRegression(random_state=10, n_jobs=60)
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
                db_invitro_matrix=(
                    matrix_h_x_invitro,
                    matrix_p_x_invitro,
                    matrix_euc_x_invitro,
                ),
                w_invitro=args.w_invitro,
                n_neighbors=args.n_neighbors,
                invitro_form=args.invitro_label,
                db_invitro=db_invitro,
                encoding=args.encoding,
                model=model,
            )

            if best_results["avg_accs"] > best_accs:
                best_p = params_comb[i]
                best_accs = best_results["avg_accs"]
                best_results = best_results
                best_ah = ah
                best_ap = ap

# -------------------tested on test dataset--------------------
for k, v in best_p.items():
    setattr(model, k, v)


minmax = MinMaxScaler().fit(X_train[non_categorical])
X_train[non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
X_test[non_categorical] = minmax.transform(X_test.loc[:, non_categorical])
db_invitro[non_categorical] = minmax.transform(db_invitro.loc[:, non_categorical])

matrix_test = dist_matrix(
    X_test, X_train, non_categorical, categorical, best_ah, best_ap
)
matrix_train = dist_matrix(
    X_train, X_train, non_categorical, categorical, best_ah, best_ap
)

db_invitro_matrix_train = dist_matrix(
    X_train, db_invitro, non_categorical, categorical_both, best_ah, best_ap
)
db_invitro_matrix_test = dist_matrix(
    X_test, db_invitro, non_categorical, categorical_both, best_ah, best_ap
)
train_index = X_train.index
test_index = X_test.index

train_rf, test_rf = cal_data_simple_rasar(
    matrix_train, matrix_test, Y_train, args.n_neighbors, args.encoding
)

invitro_form = args.invitro_label
invitro = args.w_invitro


if invitro == "own":
    train_rf = pd.DataFrame()
    test_rf = pd.DataFrame()


if (invitro != "False") & (invitro_form == "number"):
    ls = np.array(db_invitro_matrix_train.idxmin(axis=1))
    dist = np.array(db_invitro_matrix_train.min(axis=1))
    conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
    train_rf["invitro_conc"] = np.array(conc)
    train_rf["invitro_dist"] = dist

    ls = np.array(db_invitro_matrix_test.idxmin(axis=1))
    dist = np.array(db_invitro_matrix_test.min(axis=1))
    conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
    test_rf["invitro_conc"] = np.array(conc)
    test_rf["invitro_dist"] = dist

elif (invitro != "False") & (invitro_form == "label"):
    dist = np.array(db_invitro_matrix_train.min(axis=1))
    ls = np.array(db_invitro_matrix_train.idxmin(axis=1))
    label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
    train_rf["invitro_label"] = np.array(label)
    train_rf["invitro_dist"] = dist

    dist = np.array(db_invitro_matrix_test.min(axis=1))
    ls = np.array(db_invitro_matrix_test.idxmin(axis=1))
    label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
    test_rf["invitro_label"] = np.array(label)
    test_rf["invitro_dist"] = dist

elif (invitro != "False") & (invitro_form == "both"):

    dist = np.array(db_invitro_matrix_train.min(axis=1))
    ls = np.array(db_invitro_matrix_train.idxmin(axis=1))
    conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
    label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
    train_rf["invitro_conc"] = np.array(conc)
    train_rf["invitro_label"] = np.array(label)
    train_rf["invitro_dist"] = dist

    dist = np.array(db_invitro_matrix_test.min(axis=1))
    ls = np.array(db_invitro_matrix_test.idxmin(axis=1))
    conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
    label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)
    test_rf["invitro_conc"] = np.array(conc)
    test_rf["invitro_label"] = np.array(label)
    test_rf["invitro_dist"] = dist


model.fit(train_rf, Y_train)
y_pred = model.predict(test_rf)

if args.encoding == "binary":

    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
    specs = tn / (tn + fp)
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")

elif args.encoding == "multiclass":
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
info.append(
    "Alpha_h:{}, Alpha_p: {}, n:{},hyperpatameters:{}".format(
        best_ah, best_ap, args.n_neighbors, best_p
    )
)

str2file(info, args.outputFile)
