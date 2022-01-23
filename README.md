# Machine Learning for Toxicological Testing
## Introduction
In our project, we use a subset of the ECOTOX database and we use k-Nearest Neighbors (k-NN), logistic regression, random forest, two kinds of read-across structure relationships models (simple RASAR and DataFusion RASAR) and Multilayer perceptron to predict the effect of untested chemicals on tested species and vice versa. 


## Prerequites
### Basic Prerequites
- `Python` (tested on version **_3.6.2_**)
- [pip](https://pip.pypa.io/en/stable/) (tested on version *21.1.2*) (For package installation, if needed)
- `Anaconda` (Test on version *4.2.9*) (More information about Anaconda installation on your OS [here](https://docs.anaconda.com/anaconda/install/)) (For package installation)
- `numpy` (tested on version *1.19.1*)
- `scikit-learn` (tested on version *0.23.2*)
- `pandas` (tested on version *1.1.3*)
- `h2o` (tested on version *3.32.1.3*) (Only needed for multiclass datafusion model)

**NOTE**: with this configuration, the `run.py` will run without **preprocessing**

### Preprocess prerequisites 
To run the preprocessing phase the : `rdkit` (Tested on version *2017.09.1*) package and `pubchempy` (*1.0.4*) are needed.  

To install `rdkit` in your environment use the command
```bash/CMD
conda install -c rdkit rdkit
```
**NOTE**: the `run.py` will work also without `rdkit` and `pubchempy` if no preprocessing is used. An already preprocessed dataset will be used.

## Main database (ECOTOX Knowledge)
The data on the experiments is downloaded from Ecotox. Documentation on the dataset can be found at https://cfpub.epa.gov/ecotox/. For information on molecular weight, melting Point, water Solubility and Smiles we used data from CompTox (DSSTox_Predicted_NCCT_Model.zip: DSSToxQueryWPred1.xlsx, DSSToxQueryWPred2.xlsx, DSSToxQueryWPred3.xlsx, DSSToxQueryWPred4.xlsx).

Download the entire database as an ASCII file from the website, or do it from command line through
```
wget https://gaftp.epa.gov/ecotox/ecotox_ascii_09_15_2020.zip # This is for the version of 15 September 2020
unzip ecotox_ascii_09_15_2020.zip # Decompress
mv ecotox_ascii_09_15_2020 data/raw/ # Move it to the raw data directory
wget https://gaftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard/DSSTox_Predicted_NCCT_Model.zip  # chemical properties
unzip DSSTox_Predicted_NCCT_Model.zip # Decompress
mv DSSTox_Predicted_NCCT_Model data/raw/ # Move it to the raw data directory
```
If having problems with certificates, you can try adding the --no-check-certificate flag.



## Usage instruction



1. Open CMD/Bash
2. Activate the environment with needed packages installed
3. Move to the root folder, where the `run.py` is located
4. Execute the command ```python run.py``` with one or more of the following arguments:
```
Mandatory arguments:
  - encoding:
      Select either one of the two possible classification:
        -binary: Execute binary classification
        -multiclass: Execute multiclass (5 class) classification
Optional arguments:    
  -h, --help: show arguments help message and exit
  -preproc:  Execute all the preprocessing steps from the raw dataset. If not set, an already preprocessed dataset will be loaded.
  -c:  Execute all the models using only chemical (c) information. If not set, skip this step.
  -cte:  Execute all the models using chemical, taxanomy and experiment (cte) information. If not set, skip this step.
  -cte_wa:  Execute all the models using cte information and alphas. If not set, skip this step.
```
## Folder structure

    .
    ├── data 
    |   ├── processed                            # Already preprocessed data directly usable
    |   |    ├── df_db_processed.csv         
    │   |    └── lc_db_processed.csv  
    │   └── raw
    |        ├── cas_pub_tot.csv                                
    |        ├── results.txt                      # Raw data (need preprocessing)
    |        ├── species.txt 
    │        └── tests.txt 
    ├── output 
    |    ├── c                                    # Folder to store c results
    |    ├── cte                                  # Folder to store cte results
    |    ├── cte_wa                               # Folder to store cte_wa results
    |    └── mlp                                  # Folder to store mlp results
    |    
    ├── src                                    # Source files
    |    ├── helpers                           # algorithm helpers
    |    |    ├── DataScipy.py              
    │    |    ├── helper_chemproperty.py
    |    |    ├── helper_dataprocess.py          
    |    |    ├── helper_mlp.py    
    │    |    └── herlpe_model.py
    |    ├── model  
<<<<<<< HEAD
    │    |    ├── helpers.py
    │    |    |    ├── DataScipy.py 
    │    |    |    ├── helper_mlp.py 
    │    |    |    └── herlpe_model.py 
=======
>>>>>>> c999eb2ea07981374aa85bfd3b3934e1cc3d2f28
    │    |    ├── KNN.py  
    |    |    ├── LR.py     
    |    |    ├── MLP.py    
    |    |    ├── RASAR_simple.py    
    |    |    ├── RASAR_df.py    
    │    |    └── RF.py
<<<<<<< HEAD
    |    └── preprocessing                        # Preprocessing algorithm helpers and algorithms
    │         ├── helpers.py
    │         |    ├── helper_chemproperty.py 
    │         |    └── helper_dataprocess.py                     
    |         ├── data_preprocess_df.py          
    │         └── data_preprocess.py
=======
    |    ├── preprocessing                     # Preprocessing algorithm helpers and algorithms
    |         ├── data_preprocess.py          
    │         └── data_preprocess_df.py
>>>>>>> c999eb2ea07981374aa85bfd3b3934e1cc3d2f28
    ├── run.py                                 # Main entry point for the algorithms
    └── README.md

## Authors
- Jimeng Wu
- Simone D'Ambrosi
- Lorenz Ammann
- *Supervisor*: Marco Baity-Jesi