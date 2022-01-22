import argparse
import pandas as pd
from time import ctime
import subprocess


folder = ".../"

MODEL_PATH = "src/model/"
PREP_PATH = "src/preprocess/"

DATA_LC50_PATH = "data/processed/lc_db_processed.csv"
DATA_DF_PATH = "data/processed/df_db_processed.csv"


def check_encoding(x):
    if x == "binary":
        print("Binary classification chosen.")
        RESULT_SAVE_PATH_SUFFIX = "_bi.txt"
        return RESULT_SAVE_PATH_SUFFIX
    elif x == "multiclass":
        print("Multiclass classification chosen.")
        RESULT_SAVE_PATH_SUFFIX = "_mul.txt"
        return RESULT_SAVE_PATH_SUFFIX
    else:
        raise Exception("Flag problem with single/multiclass classification")


def seq_model(input_info, cate_encoding, encoding, RESULT_SAVE_PATH):

    RESULT_SAVE_PATH_SUFFIX = check_encoding(encoding)

    # KNN model
    print(input_info + ": KNN model running...", ctime())
    subprocess.call(
        [
            "python",
            folder + MODEL_PATH + "KNN.py",
            "-i",
            folder + DATA_LC50_PATH,
            "-e",
            encoding,
            "-if",
            input_info,
            "-ce",
            cate_encoding,
            "-l",
            "10",
            "-o",
            folder + RESULT_SAVE_PATH + "knn" + RESULT_SAVE_PATH_SUFFIX,
        ]
    )

    # LR model
    print(input_info + ": Logistic regression model running...", ctime())
    subprocess.call(
        [
            "python",
            folder + MODEL_PATH + "LR.py",
            "-i",
            folder + DATA_LC50_PATH,
            "-e",
            encoding,
            "-if",
            input_info,
            "-ce",
            cate_encoding,
            "-o",
            folder + RESULT_SAVE_PATH + "lr" + RESULT_SAVE_PATH_SUFFIX,
        ]
    )

    # RF model
    print(input_info + ": Random forest model running...", ctime())
    subprocess.call(
        [
            "python",
            folder + MODEL_PATH + "RF.py",
            "-i",
            folder + DATA_LC50_PATH,
            "-e",
            encoding,
            "-if",
            input_info,
            "-ce",
            cate_encoding,
            "-o",
            folder + RESULT_SAVE_PATH + "rf" + RESULT_SAVE_PATH_SUFFIX,
        ]
    )

    # Simple RASAR model using logistic regression

    print(input_info + ": Simple rasar (LR) model running...", ctime())
    subprocess.call(
        [
            "python",
            folder + MODEL_PATH + "RASAR_simple.py",
            "-i",
            folder + DATA_LC50_PATH,
            "-e",
            encoding,
            "-if",
            input_info,
            "-ce",
            cate_encoding,
            "-m",
            "lr",
            "-o",
            folder + RESULT_SAVE_PATH + "s_rasar_lr" + RESULT_SAVE_PATH_SUFFIX,
        ]
    )

    # Simple RASAR model using random forest

    print(input_info + ": Simple rasar (RF) model running...", ctime())
    subprocess.call(
        [
            "python",
            folder + MODEL_PATH + "RASAR_simple.py",
            "-i",
            folder + DATA_LC50_PATH,
            "-e",
            encoding,
            "-if",
            input_info,
            "-ce",
            cate_encoding,
            "-m",
            "rf",
            "-o",
            folder + RESULT_SAVE_PATH + "s_rasar_rf" + RESULT_SAVE_PATH_SUFFIX,
        ]
    )

    # Datafusion RASAR model
    print(input_info + ": Datafusion rasar model running...", ctime())
    subprocess.call(
        [
            "python",
            folder + MODEL_PATH + "RASAR_df.py",
            "-i",
            folder + DATA_LC50_PATH,
            "-idf",
            folder + DATA_DF_PATH,
            "-e",
            encoding,
            "-if",
            input_info,
            "-ce",
            cate_encoding,
            "-o",
            folder + RESULT_SAVE_PATH + "df_rasar" + RESULT_SAVE_PATH_SUFFIX,
        ]
    )
    return 0


def main(preproc, encoding, c, cte, cte_wa):

    # Data loading
    if preproc:
        # Cleaning data
        print("Starting cleaning process...", ctime())
        subprocess.call(
            [
                "python",
                folder + PREP_PATH + "data_preprocessing.py",
                "-o",
                folder + DATA_LC50_PATH,
            ]
        )
        print("Starting cleaning process for datafusion dataset...", ctime())
        subprocess.call(
            [
                "python",
                folder + PREP_PATH + "data_preprocessing_df.py",
                "-o",
                folder + DATA_DF_PATH,
            ]
        )
    else:
        # Loading preprocessed data sets
        print("Loading already processed dataset")

    # ------------------------------Models without alphas---------------------
    if c:
        RESULT_SAVE_PATH = "output/c/"
        input_info = "c"
        cate_encoding = "onehot"
        print("Run models with chemical information only.\n")
        seq_model(input_info, cate_encoding, encoding, RESULT_SAVE_PATH)
    if cte:
        RESULT_SAVE_PATH = "output/cte/"
        input_info = "cte"
        cate_encoding = "onehot"
        print("Run models with chemical, taxonomy and experiment information.\n")
        seq_model(input_info, cate_encoding, encoding, RESULT_SAVE_PATH)

    # ------------------------------Models with alphas---------------------
    if cte_wa:
        RESULT_SAVE_PATH = "output/cte_walphas/"
        input_info = "cte"
        cate_encoding = "ordinal"
        RESULT_SAVE_PATH_SUFFIX = check_encoding(encoding)
        print("CTE_wa: Run models including alphas.\n")

        # KNN model
        best_accs = 0
        for k_num in [1, 3, 5]:
            print("CTE_wa: {}NN model running...".format(k_num), ctime())
            subprocess.call(
                [
                    "python",
                    folder + MODEL_PATH + "KNN.py",
                    "-i",
                    folder + DATA_LC50_PATH,
                    "-e",
                    encoding,
                    "-if",
                    input_info,
                    "-ce",
                    cate_encoding,
                    "-k",
                    str(k_num),
                    "-l",
                    "10",
                    "-o",
                    folder
                    + RESULT_SAVE_PATH
                    + "{}nn".format(k_num)
                    + RESULT_SAVE_PATH_SUFFIX,
                ]
            )
            results = pd.read_csv(
                folder
                + RESULT_SAVE_PATH
                + "{}nn".format(k_num)
                + RESULT_SAVE_PATH_SUFFIX
            )

            # get the best alphas value
            if results[results.series_name == "test"].accuracy.values[0] > best_accs:
                ah = results[results.series_name == "test"].best_ah.values[0]
                ap = results[results.series_name == "test"].best_ap.values[0]

        # Simple RASAR model
        print("CTE_wa: Simple rasar model running...", ctime())
        subprocess.call(
            [
                "python",
                folder + MODEL_PATH + "RASAR_simple.py",
                "-i",
                folder + DATA_LC50_PATH,
                "-e",
                encoding,
                "-if",
                input_info,
                "-ce",
                cate_encoding,
                "-m",
                "rf",
                "-ah",
                str(ah),
                "-ap",
                str(ap),
                "-o",
                folder + RESULT_SAVE_PATH + "s_rasar_rf" + RESULT_SAVE_PATH_SUFFIX,
            ]
        )

        # Datafusion RASAR model
        print("CTE_wa: Datafusion rasar model running...", ctime())
        subprocess.call(
            [
                "python",
                folder + MODEL_PATH + "RASAR_df.py",
                "-i",
                folder + DATA_LC50_PATH,
                "-idf",
                folder + DATA_DF_PATH,
                "-e",
                encoding,
                "-if",
                input_info,
                "-ce",
                cate_encoding,
                "-ah",
                str(ah),
                "-ap",
                str(ap),
                "-o",
                folder + RESULT_SAVE_PATH + "df_rasar" + RESULT_SAVE_PATH_SUFFIX,
            ]
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ML Toxicity Prediction implementation."
    )

    # Loading algorithm params
    parser.add_argument(
        "-preproc",
        action="store_true",
        default=False,
        help="Execute all the preprocessing steps from the raw dataset. If not set, \
            an already preprocessed dataset will be loaded.",
    )
    parser.add_argument(
        "-encoding",
        default="binary",
        help="Classification type: binary, multiclass",
    )
    parser.add_argument(
        "-c", action="store_true", help="Run models using only chemical (c) information"
    )
    parser.add_argument(
        "-cte",
        action="store_true",
        help="Run models using chemical,taxanomy, experiment condition (cte) information",
    )
    parser.add_argument(
        "-cte_wa",
        action="store_true",
        help="Run models using cte with alphas",
    )
    args = parser.parse_args()
    main(args.preproc, args.encoding, args.c, args.cte, args.cte_wa)
