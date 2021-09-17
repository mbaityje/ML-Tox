# ML-Tox

A code developed and maintained by J. Wu, S. D'Ambrosi, and M. Baity-Jesi.

## Installations

#### Python
Version: 3.6.10

Required Packages: numpy, pandas, rdkit, tqdm, urllib3, pubchempy, openpyxl


## Data download

### Main database (ECOTOX Knowledge)
The data on the experiments is downloaded from Ecotox. Documentation on the dataset can be found at https://cfpub.epa.gov/ecotox/

#### Download of Ecotox

Download the entire database as an ASCII file from the website.
To download from command line, you can run the following command.
```
wget https://gaftp.epa.gov/ecotox/ecotox_ascii_03_15_2021.zip # This is for the version of 15 March 2021
unzip ecotox_ascii_03_15_2021.zip # Decompress
mv ecotox_ascii_03_15_2021 data/ # Move it to the data directory
```
If having problems with certificates, you can try adding the `--no-check-certificate` flag to the `wget` command. 
Sometimes we noticed that older versions are removed from the database.

#### Description of raw Ecotox

We used three experiment records files from Ecotox (version of 06 Nov 2020). Within the database, we used the following files:

 [i] species.txt 

 [ii] tests.txt 

 [iii] results.txt 

 In the first file there are information regarding the species of animals: all the
 taxonomic denomination (from domain to variety), other typical denominations
 (Latin name or common name), and also the ecotox group (fish, mammalians, birds,
 algae , ...).
 In the second file, tests.txt, there are information regarding the tests: laboratory
 conditions (exposure, control, application frequency, ...), CASRN of the chemical
 compound and the tested animal. Each cross-table reference is based on internal
 values that are unnecessary for the models.
 The last file contains experiment results records: LC50, ACC, EC50, etc.
 Aggregation of information tables on chemicals, species and tests is based on internal keys.


### Chemical descriptors


For information on molecular weight, melting Point, water Solubility and Smiles we used data from CompTox (DSSTox_Predicted_NCCT_Model.zip: DSSToxQueryWPred1.xlsx,
 DSSToxQueryWPred2.xlsx, DSSToxQueryWPred3.xlsx, DSSToxQueryWPred4.xlsx).
 The website was ftp://newftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard






### Chemicals

From the CAS identifiers provided in ECOTOX, we need to extract the SMILES codes.


SMILES:


From the SMILES, we extract Pubchem2D and several chemical properties.

PUBCHEM2D:

Properties:



