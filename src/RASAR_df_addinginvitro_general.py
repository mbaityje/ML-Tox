from helper_model import *
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
from tqdm import tqdm
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description="Running DataFusion RASAR model on the invivo dataset with the invitro dataset.)"
    )
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-i2", "--input_df", dest="inputFile_df", required=True)
    parser.add_argument(
        "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-ah", "--alpha_h", dest="alpha_h", required=True)
    parser.add_argument("-ap", "--alpha_p", dest="alpha_p", required=True)
    parser.add_argument(
        "-n", "--n_neighbors", dest="n_neighbors", nargs="?", default=1, type=int
    )
    parser.add_argument("-label", "--train_label", dest="train_label", required=True)
    parser.add_argument("-effect", "--train_effect", dest="train_effect", required=True)
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

if args.db_invitro != "noinvitro":
    db_mortality, db_datafusion, db_invitro = load_datafusion_datasets_invitro(
        args.inputFile,
        args.inputFile_df,
        args.db_invitro,
        categorical_columns=categorical,
        encoding=encoding,
        encoding_value=encoding_value,
    )
    db_invitro["invitro_label"] = np.where(db_invitro["invitro_conc"].values > 1, 0, 1)
else:
    db_mortality, db_datafusion = load_datafusion_datasets(
        args.inputFile,
        args.inputFile_df,
        categorical_columns=categorical,
        encoding=encoding,
        encoding_value=encoding_value,
    )
    db_invitro = args.db_invitro

# db_mortality=db_mortality[:200]
# db_datafusion=db_datafusion[:500]

X = db_mortality.drop(columns="conc1_mean").copy()
Y = db_mortality.conc1_mean.values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(ctime())
matrix_euc, matrix_h, matrix_p = cal_matrixs(
    X_train, X_train, categorical, non_categorical
)
matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
    X_train,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)

matrix_euc_x_invitro, matrix_h_x_invitro, matrix_p_x_invitro = cal_matrixs(
    X_train, db_invitro, categorical_both, non_categorical
)


print("distance matrix successfully calculated!", ctime())
del db_mortality
if encoding == "binary":
    model = RandomForestClassifier(random_state=10)
    hyper_params_tune = {
        "max_depth": [i for i in range(10, 20, 2)],
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=2)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
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
rand = 2
params_comb = list(ParameterSampler(hyper_params_tune, n_iter=40, random_state=rand))

best_accs = 0
best_p = dict()

if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]


for ah in sequence_ah:
    for ap in sequence_ap:
        for i in tqdm(range(0, len(params_comb))):
            for k, v in params_comb[i].items():
                setattr(model, k, v)

            results = cv_datafusion_rasar(
                matrix_euc,
                matrix_h,
                matrix_p,
                matrix_euc_df,
                matrix_h_df,
                matrix_p_df,
                db_invitro_matrix=(
                    matrix_h_x_invitro,
                    matrix_p_x_invitro,
                    matrix_euc_x_invitro,
                ),
                ah=ah,
                ap=ap,
                X=X_train,
                Y=Y_train,
                db_datafusion=db_datafusion,
                train_label=args.train_label,
                train_effect=args.train_effect,
                model=model,
                n_neighbors=args.n_neighbors,
                invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                db_invitro=db_invitro,
                encoding=encoding,
            )

            if results["avg_accs"] > best_accs:
                best_p = params_comb[i]
                best_accs = results["avg_accs"]
                best_results = results
                best_ah = ah
                best_ap = ap


# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_p.items():
    setattr(model, k, v)

train_index = X_train.index
test_index = X_test.index

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

db_datafusion_matrix = dist_matrix(
    X, db_datafusion, non_categorical, categorical, best_ah, best_ap
)

del (matrix_euc, matrix_h, matrix_p, matrix_euc_df, matrix_h_df, matrix_p_df)

simple_rasar_train, simple_rasar_test = cal_data_simple_rasar(
    matrix_train, matrix_test, Y_train, args.n_neighbors, args.encoding
)

datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
    train_index,
    test_index,
    X_train,
    X_test,
    db_datafusion,
    db_datafusion_matrix,
    args.train_label,
    args.train_effect,
    encoding,
)

train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)

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
        train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
        test_rf["invitro_conc"] = X_test.invitro_conc.reset_index(drop=True)
        train_rf["invitro_label"] = X_train.invitro_label.reset_index(drop=True)
        test_rf["invitro_label"] = X_test.invitro_label.reset_index(drop=True)
else:
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


del (
    datafusion_rasar_test,
    datafusion_rasar_train,
    simple_rasar_test,
    simple_rasar_train,
)

if encoding == "binary":

    model.fit(train_rf, Y_train)
    y_pred = model.predict(test_rf)

    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
    specs = tn / (tn + fp)
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")
elif encoding == "multiclass":

    train_rf.loc[:, "target"] = Y_train
    test_rf.loc[:, "target"] = Y_test

    train_rf_h2o = h2o.H2OFrame(train_rf)
    test_rf_h2o = h2o.H2OFrame(test_rf)

    for col in train_rf.columns:
        if "label" in col:
            train_rf_h2o[col] = train_rf_h2o[col].asfactor()
            test_rf_h2o[col] = test_rf_h2o[col].asfactor()

    train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
    test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

    model.train(y="target", training_frame=train_rf_h2o)
    y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]

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

info.append("Alpha_h:{}, Alpha_p: {},n:{}".format(best_ah, best_ap, args.n_neighbors))
info.append("Random state".format(rand))
filename = args.outputFile
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)

with open(filename, "w") as file_handler:
    for item in info:
        file_handler.write("{}\n".format(item))
