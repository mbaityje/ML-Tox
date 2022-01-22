from helpers.helper_dataprocess import *
import argparse


def getArguments():
    parser = argparse.ArgumentParser(description="Running data cleaning.")
    parser.add_argument(
        "-o", "--output", help="outputFile", default="df_db_processed.csv"
    )
    return parser.parse_args()


args = getArguments()
# -----------------------Step 1: Load Data---------------------------------

# We used three experiment records files from Ecotox with version of
# 09/15/2020 from website (https://cfpub.epa.gov/ecotox/):
# [i] species.txt
# [ii] tests.txt
# [iii] results.txt

# For information on molecular weight, melting Point, water Solubility
# and Smiles, we used data from CompTox (DSSTox_Predicted_NCCT_Model.zip:
# DSSToxQueryWPred1-4.xlsx).

# In the first file, species.txt, there are information regarding the
# species of animals: all the taxonomic denomination (from domain to variety),
# other typical denominations (Latin name or common name), and also the ecotox
# group (fish, mammalians, birds, algae , ...).
# In the second file, tests.txt, there are information regarding the tests:
# laboratory conditions (exposure, control, application frequency, ...),
# CASRN of the chemical compound and the tested animal. Each cross-table
# reference is based on internal values that are unnecessary for the models.
# The last file, results.txt, contains experiment results records: LC50, LOEC,
# EC50, etc.
# Aggregation of information tables on chemicals, species and tests is based
# on internal keys.

DATA_RESULTS_PATH = r"data/raw/results.txt"
DATA_TEST_PATH = r"data/raw/tests.txt"
DATA_SPECIES_PATH = r"data/raw/species.txt"
DATA_PROPERTY_PATH = [
    "data/raw/DSSToxQueryWPred1.xlsx",
    "data/raw/DSSToxQueryWPred2.xlsx",
    "data/raw/DSSToxQueryWPred3.xlsx",
    "data/raw/DSSToxQueryWPred4.xlsx",
]

tests, species, results, properties = load_raw_data(
    DATA_TEST_PATH, DATA_RESULTS_PATH, DATA_SPECIES_PATH, DATA_PROPERTY_PATH
)

# ---------------------Step 2: Prefiltering--------------------------------

# Once loaded the data, we restrict the animal kingdom to fish only and
# and the results as format of concentration.

results_prefiltered = prefilter(species, tests, results, label="datafusion")


# --------------Step 3: Effect & Endpoint Selection; Imputation--------------

# Once loaded the data, we filter the effect and endpoint only if they
# have enought experiment records, and we remove all mortality experiments

results_imputated = crosstab_rep_exp(results_prefiltered, effect="MOR")


# ----Step 4: Extraction of PubChem2D and molecular descriptors from CASRN----

# We use the file to extract all information from the CASRN.

# If you want to avoid calculating all pubchems and molecular descriptors
# (about 2 hours), you can use the alternative function "process_chemicals"
# which takes as input a dataset with this info already extracted.


if 0:
    # Option 1: get the properties
    results_pub = smiles_to_pubchem(results_imputated)
else:
    # Option 2: use the processed file
    # The smiles was extracted from the raw property datasets
    # (DSSToxQueryWPred1-4) for each chemical in our invivo dataset.
    # The pubchem2d was generated using the function smiles_to_pubchem()
    # in helper_dataprocessing.py on all the chemicals in our in vivo dataset.
    # The function gets pubchem2d from smiles using the PubChemPy package.

    pubchem = pd.read_csv("../data/raw/cas_pub_tot.csv")
    results_pub = results_imputated.merge(pubchem[["smiles", "pubchem2d"]], on="smiles")


# extract other molecular properties
results_chem = extract_mol_properties(results_pub)

# -------------------Step 5: Transformation of chemical features-------------------

# Some variables need transformations to regularize their distributions.
# The transformed features are: "bonds_number", "atom_number", "mol_weight" and
# "WaterSolubility". Transformation is logarithmic and then MinMax. For
# "WaterSolubility", we used the Box-Cox transformation to normalize the distribution.

final_results = process_features(results_chem)

final_results.to_csv(args.output)


# The website was ftp://newftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard
