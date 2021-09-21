# ML-Tox

A code developed and maintained by J. Wu, S. D'Ambrosi, and M. Baity-Jesi.

## Installations

#### Python
Version: 3.6.10

Required Packages: numpy, pandas, rdkit, tqdm, urllib3, pubchempy, openpyxl

## Files

* `data_download.sh`: This script downloads the necessary data. The urls are hardcoded and sometimes the websites change the filenames, so manual edits might be required.
The data on the experiments is downloaded from Ecotox. Documentation on the dataset can be found at [https://cfpub.epa.gov/ecotox/](https://cfpub.epa.gov/ecotox/).
We used three experiment records files from Ecotox (version of 06 Nov 2020). Within the database, we used the following files:

    * `species.txt`: contains information regarding the species of animals: all the
 taxonomic denomination (from domain to variety), other typical denominations
 (Latin name or common name), and also the ecotox group (fish, mammalians, birds,
 algae , ...).

    * `tests.txt`: contains information regarding the tests, such as laboratory
 conditions (exposure, control, application frequency, ...), CASRN of the chemical
 compound and the tested animal. Each cross-table reference is based on internal
 values that are unnecessary for the models.

    * `results.txt`: contains experiment results records: LC50, ACC, EC50, etc.

    Aggregation of information tables on chemicals, species and tests is based on internal keys.


    For information on molecular weight, melting Point, water Solubility and Smiles we used data from CompTox (DSSTox_Predicted_NCCT_Model.zip: DSSToxQueryWPred1.xlsx,
 DSSToxQueryWPred2.xlsx, DSSToxQueryWPred3.xlsx, DSSToxQueryWPred4.xlsx).
 The website was ftp://newftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard

* `data_preprocessing.py`: performs the data cleaning necessary for all models except the Data Fusion RASAR (since the DF-RASAR requires information on several endpoints).



