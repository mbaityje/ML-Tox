from helper_dataprocessing import *

# -----------------------Step 1: Load Data--------------------------------

# We used three experiment records files from Ecotox with version of 06/11/2020
# from website (https://cfpub.epa.gov/ecotox/):
# [i] species.txt
# [ii] tests.txt
# [iii] results.txt

# For information on molecular weight, melting Point, water Solubility and Smiles we
# used data from CompTox (DSSTox_Predicted_NCCT_Model.zip: DSSToxQueryWPred1.xlsx,
# DSSToxQueryWPred2.xlsx, DSSToxQueryWPred3.xlsx, DSSToxQueryWPred4.xlsx).
# The website was ftp://newftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard

# In the first file there are information regarding the species of animals: all the
# taxonomic denomination (from domain to variety), other typical denominations
# (Latin name or common name), and also the ecotox group (fish, mammalians, birds,
# algae , ...).
# In the second file, tests.txt, there are information regarding the tests: laboratory
# conditions (exposure, control, application frequency, ...), CASRN of the chemical
# compound and the tested animal. Each cross-table reference is based on internal
# values that are unnecessary for the models.
# The last file contains experiment results records: LC50, ACC, EC50, etc.
# Aggregation of information tables on chemicals, species and tests is based on internal keys.

DATA_RESULTS_PATH = r"data/raw/results.txt"
DATA_TEST_PATH = r"data/raw/tests.txt"
DATA_SPECIES_PATH = r"data/raw/species.txt"
DATA_PROPERTY_PATH = [
    "data/DSSToxQueryWPred1.xlsx",
    "data/DSSToxQueryWPred2.xlsx",
    "data/DSSToxQueryWPred3.xlsx",
    "data/DSSToxQueryWPred4.xlsx",
]

tests, species, results, properties = load_raw_data(
    DATA_TEST_PATH, DATA_RESULTS_PATH, DATA_SPECIES_PATH, DATA_PROPERTY_PATH
)

# ---------------------Step 2: Prefiltering--------------------------------

# Once loaded the data, we filter the results on endpoint (we take only EC50 and LC50)
# and effects (we take only mortality (MOR)). We restrict the animal kingdom to fish only.
# Also, we removed embryos tests.

results_prefiltered = prefilter(
    species, tests, results, endpoint="LC50|EC50", effect="MOR"
)

# merging with the properties
results_prefiltered = results_prefiltered.merge(properties, on="test_cas")

# ---------------------Step 3: Feature Selection and Imputation------------

# After the selection, we proceeded to impute or remove the missing values.

results_imputed = select_impute_features(results_prefiltered)

# ---------------------Step 4: Repeated Experiment Aggregation------------

# For repeated experiments, we merged them into one and set the median concentration
# as the results. We decided the two experiments are repeated if they share the same
# taxonomy, chemcial and experiment conditions. (We selected 'exposure_type','control_type',
# 'media_type', 'application_freq_unit')

results = repeated_experiments(results_imputed)

# ---------------------Step 5: Extraction of PubChem2D and molecular descriptors from CASRN-----

# We use the file to extract all information from the CASRN.

# If you want to avoid calculating all pubchems and molecular descriptors (about 2 hours)
# you can use the alternative function "process_chemicals" which takes as input a dataset
# with these info already extracted.

if 0:
    # Option 1: get the properties
    results_pub = smiles_to_pubchem(results)
else:
    # Option 2: use the saved file
    # The smiles column was extracted from the raw property datasets (DSSToxQueryWPred1-4) for each chemical in our invivo dataset.
    # The pubchem2d column was generated using the function smiles_to_pubchem() in helper_dataprocessing.py on all the chemicals in our in vivo dataset.
    # The function gets pubchem2d from smiles using the PubChemPy package.

    pubchem = pd.read_csv("../data/processed/cas_pub_tot.csv")
    results_pub = results.merge(pubchem[["smiles", "pubchem2d"]], on="smiles")

# extract other molecular properties
results_chem = extract_mol_properties(results_pub)

# ----------------------Step 6: Transformation of chemical features----------------

# Some variables need transformations to regularize their distributions.
# The transformed features are: "bonds_number", "atom_number", "mol_weight" and "WaterSolubility".
# Transformation is logarithmic and then MinMax. For "WaterSolubility" we used the Box-Cox
# transformation to normalize the distribution.

final_results = process_features(results_chem)

final_results.to_csv("lc_db_processed.csv")

