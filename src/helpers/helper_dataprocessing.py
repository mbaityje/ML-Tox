import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split

# from tqdm import tqdm
import pubchempy as pcp
from time import ctime
from helpers.helper_chemproperty import *

import helpers.DataSciPy as dsp


def to_cas(num):
    s = str(num)
    s = s[:-3] + "-" + s[-3:-1] + "-" + s[-1]
    return s


def load_raw_data(
    DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES, DATA_PROPERTY_PATH
):

    tests = pd.read_csv(DATA_PATH_TESTS, sep="\|", engine="python")
    print("tests loaded")
    print("Tests table dimensions: ", tests.shape)

    species = pd.read_csv(DATA_PATH_SPECIES, sep="\|", engine="python")
    print("species loaded")
    print("Species table dimensions: ", species.shape)

    results = pd.read_csv(DATA_PATH_RESULTS, sep="\|", engine="python", error_bad_lines=False)
    print("results loaded")
    print("Results table dimensions: ", results.shape)

    print("start loading the properties data...")
    df_property = pd.DataFrame()
    for i in DATA_PROPERTY_PATH:
        df_property = pd.concat(
            [
                df_property,
                pd.read_excel(
                    open(i, "rb"),
                    usecols=[
                        "Substance_CASRN",
                        "Structure_SMILES",
                        "Structure_MolWt",
                        "NCCT_MP",
                        "NCCT_WS",
                    ],
                    engine="openpyxl",
                ),
            ],
            axis=0,
        )
        print(str(i) + " loaded", ctime())
    df_property = df_property.rename(
        {
            "Substance_CASRN": "test_cas",
            "Structure_SMILES": "smiles",
            "Structure_MolWt": "mol_weight",
            "NCCT_MP": "melting_point",
            "NCCT_WS": "water_solubility",
        },
        axis=1,
    )
    return tests, species, results, df_property


# ---------------------------------------------------------------------
def prefilter(
    species,
    tests,
    results,
    endpoint=None,
    label="simple",
    all_property="no",
    effect="MOR",
):
    results.loc[:, "effect"] = results.effect.apply(
        lambda x: x.replace("/", "") if "/" in x else x
    )
    results.loc[:, "effect"] = results.effect.apply(
        lambda x: x.replace("~", "") if "~" in x else x
    )
    results.loc[:, "endpoint"] = results.endpoint.apply(
        lambda x: x.replace("/", "") if "/" in x else x
    )
    results.loc[:, "endpoint"] = results.endpoint.apply(
        lambda x: x.replace("*", "") if "*" in x else x
    )
    if label == "datafusion":
        # filter on only concentration endpoint
        resc = results[results.endpoint.str.contains("C")].copy()
        if all_property == "no":
            rconc = resc.loc[~(results.effect.str.contains(effect)), :]
        elif all_property == "itself":
            rconc = resc.loc[
                ~(
                    (~results.endpoint.str.contains(endpoint))
                    & (results.effect.str.contains(effect))
                ),
                :,
            ]
        elif all_property == "all":
            rconc = resc
        elif all_property == "other":
            rconc = resc.loc[
                ~(
                    results.endpoint.str.contains(endpoint)
                    & results.effect.str.contains(effect)
                ),
                :,
            ]
        print("There are", rconc.shape[0], "tests.")
        test = tests.copy()
    elif label == "simple":
        resc = results[(results.endpoint.str.contains(endpoint))]
        print("There are", resc.shape[0], "tests.(only " + endpoint + ")")
        rconc = resc[resc.effect.str.contains(effect)]
        print("There are", rconc.shape[0], "tests consideing about " + effect + ".")
        test = tests[tests.organism_lifestage != "EM"]
    else:
        rconc = results[results.endpoint.str.contains("C")].copy()
        test = tests[tests.organism_lifestage != "EM"]

    # focus on fishes
    species = species[~species.ecotox_group.isnull()]
    sp_fish = species[species.ecotox_group.str.contains("Fish")]

    # merging tests and tested fishes
    test_fish = test.merge(sp_fish, on="species_number")
    print("There are", test_fish.shape[0], "tests on these fishes.")

    # merging experiments and their relative results
    results_prefilter = rconc.merge(test_fish, on="test_id")
    print(
        "The unique chemical number was", len(results_prefilter.test_cas.unique()), "."
    )
    print("All merged into one dataframe. Size was", results_prefilter.shape, ".")

    results_prefilter["test_cas"] = results_prefilter["test_cas"].apply(to_cas)
    return results_prefilter


# -------------------------------------------------------------------
def impute_conc(results_prefiltered):
    """
    impute or remove the missing value about concentration
    """

    keep_conc_unit = [
        "ppb",
        "ppm",
        "ug/L",
        "ng/L",
        "mg/L",
        "ng/ml",
        "mg/dm3",
        "umol/L",
        "mmol/L",
        "ug/ml",
        "g/L",
        "ng/ml",
        "nmol/L",
        "mol/L",
        "g/dm3",
        "ug/dm3",
    ]

    db = results_prefiltered.copy()

    # treat the concentration unit with "AI" (Active ingredient) inside
    # the same as others.
    db.loc[:, "conc1_unit"] = db.conc1_unit.apply(
        lambda x: x.replace("AI ", "") if "AI" in x else x
    )

    # keep the main units since others only took a small precentages.
    db = db[db.conc1_unit.isin(keep_conc_unit)].copy()

    # remove the results are Not coded (NC), Not reported (NR) or null value.
    to_drop_conc_mean = db[
        (db.conc1_mean == "NC") | (db.conc1_mean == "NR") | db.conc1_mean.isnull()
    ].index
    db_filtered_mean = db.drop(to_drop_conc_mean).copy()

    # remove the asterisk inside the concentration value
    db_filtered_mean.loc[:, "conc1_mean"] = db_filtered_mean.conc1_mean.apply(
        lambda x: x.replace("*", "") if "*" in x else x
    ).copy()

    # remove the concentration that is higher than 100000 mg/L and equals 0 mg/L
    to_drop_invalid_conc = db_filtered_mean[
        (db_filtered_mean.conc1_mean == ">100000") | (db_filtered_mean.conc1_mean == 0)
    ].index
    db_filtered_mean.drop(index=to_drop_invalid_conc, inplace=True)

    db_filtered_mean.loc[:, "conc1_mean"] = db_filtered_mean.conc1_mean.astype(
        float
    ).copy()

    # convert all units into mg/L
    db_filtered_mean.loc[
        (db_filtered_mean.conc1_unit == "ppb")
        | (db_filtered_mean.conc1_unit == "ug/L")
        | (db_filtered_mean.conc1_unit == "ng/ml")
        | (db_filtered_mean.conc1_unit == "ug/dm3"),
        "conc1_mean",
    ] = db_filtered_mean.conc1_mean[
        (db_filtered_mean.conc1_unit == "ppb")
        | (db_filtered_mean.conc1_unit == "ug/L")
        | (db_filtered_mean.conc1_unit == "ng/ml")
        | (db_filtered_mean.conc1_unit == "ug/dm3")
    ] / (
        10 ** (3)
    )

    db_filtered_mean.loc[
        db_filtered_mean.conc1_unit == "ng/L", "conc1_mean"
    ] = db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == "ng/L"] / (10 ** (6))

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == "umol/L", "conc1_mean"] = (
        db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == "umol/L"]
        / (10 ** (3))
        * db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == "umol/L"]
    )

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == "mmol/L", "conc1_mean"] = (
        db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == "mmol/L"]
        * db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == "mmol/L"]
    )

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == "nmol/L", "conc1_mean"] = (
        db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == "nmol/L"]
        / (10 ** (6))
        * db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == "nmol/L"]
    )

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == "mol/L", "conc1_mean"] = (
        db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == "mol/L"]
        * (10 ** (3))
        * db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == "mol/L"]
    )

    db_filtered_mean.loc[
        (db_filtered_mean.conc1_unit == "g/L")
        | (db_filtered_mean.conc1_unit == "g/dm3"),
        "conc1_mean",
    ] = db_filtered_mean.conc1_mean[
        (db_filtered_mean.conc1_unit == "g/L")
        | (db_filtered_mean.conc1_unit == "g/dm3")
    ] * (
        10 ** (3)
    )

    db_filtered_mean.drop(columns=["conc1_unit"], inplace=True)

    # remove the experiments with Not Coded or Not Reported concentration type

    to_drop_type = db_filtered_mean.loc[
        (db_filtered_mean.conc1_type == "NC") | (db_filtered_mean.conc1_type == "NR")
    ].index
    db_filtered_mean.drop(index=to_drop_type, inplace=True)

    return db_filtered_mean


def impute_test_feat(results_prefiltered):
    """
    Impute the test features.
    The variables to impute are: "exposure_type", "test_location",
    "control_type", "media_type" and "application_freq_unit"."""
    db = results_prefiltered.copy()
    # For exposure_type
    # remove the "/" symbol
    db.loc[:, "exposure_type"] = db.exposure_type.apply(lambda x: x.replace("/", ""))
    # change the "AQUA-NR" into "AQUA"
    db.loc[:, "exposure_type"] = db.exposure_type.apply(
        lambda x: x.replace("AQUA - NR", "AQUA") if "AQUA" in x else x
    )

    # The Not Reported values (about 5000, ie 0.089% of the total)
    # are attributed with the generic value "AQUA" ie exposure in water.
    db.loc[:, "exposure_type"] = db.exposure_type.apply(
        lambda x: "AQUA" if "NR" in x else x
    )
    # For test_location:
    # This variable is removed because it is unbalanced: more than 98%
    # of the total experiments were performed in the laboratory,
    # while the remaining 2% were performed on "artificial" or "natural" soil.
    db.drop(columns=["test_location"], inplace=True)

    # For control_type:
    # remove the "/" symbol

    db.loc[:, "control_type"] = db.control_type.apply(lambda x: x.replace("/", ""))
    db.loc[:, "control_type"] = db.control_type.apply(
        lambda x: "Unknown" if "NR" in x else x
    )
    # For media_type:
    # remove the "/" symbol
    # Remove the records with media_type value of ['NR', 'CUL', 'NONE', 'NC'],
    # where CUL stands for Culture.

    db.loc[:, "media_type"] = db.media_type.apply(lambda x: x.replace("/", ""))
    to_drop_media = db[db.media_type.isin(["NR", "CUL", "NONE", "NC"])].index
    db.drop(to_drop_media, inplace=True)

    # For application_freq_unit:
    # The missing values, Not Reported and Not Coded, are imputed with the
    # most frequent class: "X", ie the concentration of the compound is
    # unitary (the compound is given all at once to the fish).

    db.loc[:, "application_freq_unit"] = db.application_freq_unit.apply(
        lambda x: "X" if ("NR" in x) | ("NC" in x) else x
    )

    return db


def impute_duration(results_prefiltered):
    """
    With the help of obs_duration_unit, we change the experiments all
    last between [24, 48, 72, 96] hours.
    """
    # convert all units into hour
    db = results_prefiltered.copy()

    keep_obs_units = ["h", "d", "mi", "wk", "mo"]
    db_filtered_unit = db[db.obs_duration_unit.isin(keep_obs_units)].copy()

    to_drop_obs_mean = db_filtered_unit[
        db_filtered_unit.obs_duration_mean == "NR"
    ].index
    db_filtered_unit.drop(to_drop_obs_mean, inplace=True)
    db_filtered_unit.obs_duration_mean = db_filtered_unit.obs_duration_mean.astype(
        float
    )

    db_filtered_unit.loc[
        db_filtered_unit.obs_duration_unit == "d", "obs_duration_mean"
    ] = db_filtered_unit.obs_duration_mean[
        db_filtered_unit.obs_duration_unit == "d"
    ].apply(
        lambda x: x * 24
    )
    db_filtered_unit.loc[
        db_filtered_unit.obs_duration_unit == "mi", "obs_duration_mean"
    ] = db_filtered_unit.obs_duration_mean[
        db_filtered_unit.obs_duration_unit == "mi"
    ].apply(
        lambda x: x / 60
    )
    db_filtered_unit.loc[
        db_filtered_unit.obs_duration_unit == "wk", "obs_duration_mean"
    ] = db_filtered_unit.obs_duration_mean[
        db_filtered_unit.obs_duration_unit == "wk"
    ].apply(
        lambda x: x * 7 * 24
    )
    db_filtered_unit.loc[
        db_filtered_unit.obs_duration_unit == "mo", "obs_duration_mean"
    ] = db_filtered_unit.obs_duration_mean[
        db_filtered_unit.obs_duration_unit == "mo"
    ].apply(
        lambda x: x * 30 * 24
    )

    db_filtered_unit.drop(columns=["obs_duration_unit"], inplace=True)

    db_processed_duration = db_filtered_unit[
        db_filtered_unit.obs_duration_mean.isin([24, 48, 72, 96])
    ].copy()

    return db_processed_duration


def impute_species(results_prefiltered):
    """
    Remove records with null value.
    """
    db = results_prefiltered.copy()
    # Dropping missing values relative to species (same values are missing for genus)
    # to_drop_spec = db[db.species.isnull()].index
    # db.drop(to_drop_spec, inplace=True)
    db = db[~db.species.isnull()]

    # Dropping missing values relative to family
    # to_drop_fam = db[db.family.isnull()].index
    # db.drop(to_drop_fam, inplace=True)

    db = db[~db.family.isnull()]
    db = db[~db.genus.isnull()]
    db = db[~db.tax_order.isnull()]
    db = db[~db["class"].isnull()]

    return db


def select_impute_features(prefiltered_results):
    keep_columns = [
        "obs_duration_mean",
        "obs_duration_unit",
        "endpoint",
        "effect",
        "measurement",
        "conc1_type",
        "conc1_mean",
        "conc1_unit",
        "test_cas",
        "test_location",
        "exposure_type",
        "control_type",
        "media_type",
        "application_freq_unit",
        "class",
        "tax_order",
        "family",
        "genus",
        "species",
        "smiles",
        "mol_weight",
        "melting_point",
        "water_solubility",
    ]

    db = prefiltered_results.copy()
    db = db[keep_columns]

    db = impute_conc(db)

    db = impute_test_feat(db)

    db = impute_duration(db)

    db = impute_species(db)

    return db


# -------------------------------------------------------------------
def repeated_experiments(imputed_db):
    """
    Remove the repeated experiments
    """
    db = imputed_db.copy()
    db["fish"] = (
        db["class"]
        + " "
        + db["tax_order"]
        + " "
        + db["family"]
        + " "
        + db["genus"]
        + " "
        + db["species"]
    )

    db_species = db[["class", "tax_order", "family", "genus", "species", "fish"]]
    db_species = db_species.groupby("fish").first()

    final_db = (
        db.groupby(
            by=[
                "test_cas",
                "smiles",
                "obs_duration_mean",
                "conc1_type",
                "fish",
                "exposure_type",
                "control_type",
                "media_type",
                "application_freq_unit",
            ]
        )
        .agg("median")
        .reset_index()
    )
    final_db = final_db.merge(db_species, on="fish")

    return final_db


# -----------------------------------------------------------


def crosstab_rep_exp(dataframe):
    ct = pd.crosstab(dataframe.effect, dataframe.endpoint)

    # keep only when the dataset is large enough
    new_ct = ct.loc[ct.sum(axis=1) > 200, ct.sum(axis=0) > 200]

    dfr = pd.DataFrame()

    for j in new_ct.columns:
        for i in new_ct.index:
            # remove all other experiments about selected effect
            if new_ct.loc[i, j] > 100:
                pp = dataframe[dataframe.endpoint == j]
                pp = pp[pp.effect == i]

                best_db = select_impute_features(pp)
                try:
                    # only keep daaset is large enough
                    if repeated_experiments(best_db).shape[0] > 100:
                        rep_exp_db = repeated_experiments(best_db)
                        rep_exp_db.loc[:, "endpoint"] = pd.Series(
                            np.repeat(j, rep_exp_db.shape[0])
                        )
                        rep_exp_db.loc[:, "effect"] = pd.Series(
                            np.repeat(i, rep_exp_db.shape[0])
                        )

                        dfr = pd.concat([dfr, rep_exp_db])
                except:
                    continue
    return dfr


def extract_mol_properties(features):
    features = features[~features.smiles.isnull()]
    features = features[~features.pubchem2d.isnull()]
    chem_feat = adding_smiles_features(features)
    to_drop_nofeat = chem_feat[chem_feat["bonds_number"] == np.nan].index
    chem_feat.drop(to_drop_nofeat, inplace=True)
    to_drop_null = chem_feat[chem_feat.isnull().any(axis=1)].index
    chem_feat.drop(index=to_drop_null, inplace=True)
    return chem_feat


def process_features(chemical_features):

    db = chemical_features.copy()

    db.bonds_number = db.bonds_number.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(db[["bonds_number"]])
    db[["bonds_number"]] = minmax.transform(db[["bonds_number"]])

    db.atom_number = db.atom_number.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(db[["atom_number"]])
    db[["atom_number"]] = minmax.transform(db[["atom_number"]])

    db.mol_weight = db.mol_weight.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(db[["mol_weight"]])
    db[["mol_weight"]] = minmax.transform(db[["mol_weight"]])

    pt = PowerTransformer(method="box-cox")

    pt.fit(db.water_solubility.values.reshape(-1, 1))
    db[["water_solubility"]] = pt.transform(
        db.water_solubility.values.reshape(-1, 1)
    ).ravel()

    return db


# ----------------------------------------------------------


def smiles_to_pubchem(df):

    pubchem = pd.DataFrame()
    pubchem["smiles"] = df["smiles"].unique()
    pubchem["pubchem2d"] = np.nan
    for i in range(pubchem.shape[0]):
        try:
            pubchem.loc[i, "pubchem2d"] = pcp.get_compounds(
                pubchem["smiles"][i], "smiles"
            )[0].cactvs_fingerprint
        except:
            pubchem.loc[i, "pubchem2d"] = np.nan
    df_pubchem = df.merge(pubchem, on="smiles")

    return df_pubchem


def null_output_counts(dataframe):

    # Find columns that start with the interesting feature
    features_interested = list(dataframe.columns)

    df_nan = pd.DataFrame(
        index=features_interested, columns=["null_values_inc_NC_NR%", "#outputs"]
    )

    # Count total NaN + NR + NC
    for i in features_interested:
        df_nan["null_values_inc_NC_NR%"][i] = (
            (
                sum(dataframe[i].isnull())
                + len(dataframe[dataframe[i] == "NR"])
                + len(dataframe[dataframe[i] == "NC"])
            )
            / len(dataframe)
            * 100
        )
        df_nan["#outputs"][i] = len(dataframe[i].unique())
    return df_nan

# MLP data preprocessing ------------------------------------------------------

def scores_cat(conc):
    '''Return score of the concentration passed as input
        Inputs:
            - conc (float): concentration value
        Outputs:
            - score (int): score given to the concentration, from 1 to 5 (see report for detail)'''
    
    if (conc <= 0.1):
        return 4
    elif ((conc>0.1) and (conc<=1)):
        return 3
    elif ((conc>1) and (conc<=10)):
        return 2
    elif ((conc>10) and (conc<=100)):
        return 1
    else:
        return 0


def binary_score(final_db):
    '''Set label to 0 or 1 based on the selected threshold of 1 mg/L
        Inputs:
            - final_db (Pandas dataframe): cleaned dataframe 
        Outputs:
            - final_db_scored (Pandas dataframe): dataframe with column score containing the labels'''
            
    # Computing score
    final_db_scored = final_db.copy()
    final_db_scored["score"] = np.where(final_db_scored.conc1_mean > 1, 0, 1)
    # Drop conc1_mean (it is replaced by score)
    final_db_scored = final_db_scored.drop(columns=['conc1_mean'])
    
    # Info
    num0 = len(final_db_scored[final_db_scored.score==0])
    num1 = len(final_db_scored[final_db_scored.score==1])
    print("There are {} datapoints with label 0 and {} datapoints with label 1".format(num0, num1))
    return final_db_scored


def multi_score(final_db):
    '''Set label to 0 or 1 based on the selected threshold of 1 mg/L
        Inputs:
            - final_db (Pandas dataframe): cleaned dataframe 
        Outputs:
            - final_db_scored (Pandas dataframe): dataframe with column score containing the labels'''
            
    # Computing score
    final_db_scored = final_db.copy()
    final_db_scored["score"] = final_db_scored.conc1_mean.apply(lambda x: scores_cat(x))
    # Drop conc1_mean (it is replaced by score)
    final_db_scored = final_db_scored.drop(columns=['conc1_mean'])
    
    # Info
    for i in range(0, 5):
        num = len(final_db_scored[final_db_scored.score==i])
        print("There are {} datapoints with label {}".format(num, i))

    return final_db_scored


def prepare_data(path_data, setup, numerical, multiclass=False):
    # ------------------------------------------------------------------------
    # Prepare data for use as features in MLP models
    # ------------------------------------------------------------------------
    final_db = pd.read_csv(path_data)
    final_db = final_db.drop(columns=['Unnamed: 0'])
    
    if setup=='fathead' or setup=='rainbow':
        if setup=='fathead':
            fish_curr = 'Actinopterygii Cypriniformes Cyprinidae Pimephales promelas'
        elif setup=='rainbow':
            fish_curr = 'Actinopterygii Salmoniformes Salmonidae Oncorhynchus mykiss'
        # keep only fathead minnow or rainbow trout in trainval and remove all but chemical features and collapse duplicated rows to median conc
        tmp = numerical.copy()
        tmp.extend(['test_cas','conc1_mean'])
        X_trainval = final_db[final_db['fish']==fish_curr]
        X_cllps_trainval = X_trainval.loc[:,tmp]
        X_cllps_trainval = X_cllps_trainval.groupby(by = 'test_cas').agg('median').reset_index()
        X_cllps_trainval.index=X_cllps_trainval.loc[:,'test_cas']
        X_test = final_db[final_db['fish']!=fish_curr]
        X_cllps_test = X_test.loc[:,tmp]
        X_cllps_test = X_cllps_test.groupby(by = 'test_cas').agg('median').reset_index()
        X_cllps_test.index=X_cllps_test.loc[:,'test_cas']
        
        X_trainval = X_trainval.loc[:,['test_cas','pubchem2d']].drop_duplicates().join(X_cllps_trainval,on=['test_cas'],lsuffix='_lft')
        X_trainval = X_trainval.drop(columns=['test_cas_lft'])
        X_test = X_test.loc[:,['test_cas','pubchem2d']].drop_duplicates().join(X_cllps_test,on=['test_cas'],lsuffix='_lft')
        X_test = X_test.drop(columns=['test_cas_lft'])
        
        fp = pd.DataFrame(X_trainval.pubchem2d.apply(dsp.splt_str))
        fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
        fp.columns = ['pub'+ str(i) for i in range(1,882)]
        # data with pubchem2d
        X_trainval = X_trainval.drop(columns=['pubchem2d']).join(fp)
        X_trainval = X_trainval.drop(columns=['test_cas'])
        
        fp = pd.DataFrame(X_test.pubchem2d.apply(dsp.splt_str))
        fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
        fp.columns = ['pub'+ str(i) for i in range(1,882)]
        # data with pubchem2d
        X_test = X_test.drop(columns=['pubchem2d']).join(fp)
        X_test = X_test.drop(columns=['test_cas'])
        
        # Prepare binary or multiclass problem
        if(multiclass):
            X_trainval = multi_score(X_trainval)
            X_test = multi_score(X_test)
        else:
            X_trainval = binary_score(X_trainval)
            X_test = binary_score(X_test)

        sclr = MinMaxScaler()
        dummy = dsp.Dataset()
        dummy.setup_data(X=X_trainval.drop(columns=['score']),
                         y=X_trainval.loc[:,['score']],
                         split_test=0)

        y_trainval = X_trainval.loc[:,['score']]
        X_trainval = X_trainval.drop(columns=['score'])
        y_test = X_test.loc[:,['score']]
        X_test = X_test.drop(columns=['score'])
        # Tensorflow does not accept boolean (or int) data, so convert to float
        X_trainval = X_trainval.astype(np.float64)
        y_trainval = y_trainval.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_test = y_test.astype(np.float64)

        sclr = MinMaxScaler()
        sclr.fit(X_trainval[numerical])
        xtmp = X_trainval.copy()
        xtmp.loc[:,numerical] = sclr.transform(X_trainval.loc[:,numerical])
        X_trainval = xtmp.copy()
        xtmp = X_test.copy()
        xtmp.loc[:,numerical]  = sclr.transform(X_test.loc[:,numerical])
        X_test = xtmp.copy()
        return X_trainval, y_trainval, X_test, y_test, dummy
        
    else:
        if setup=='onlychem':
            # remove all but chemical features and collapse duplicated rows to median conc
            tmp = numerical.copy()
            tmp.extend(['test_cas','conc1_mean'])
            final_db_cllps = final_db.loc[:,tmp]
            final_db_cllps = final_db_cllps.groupby(by = 'test_cas').agg('median').reset_index()
            final_db_cllps.index=final_db_cllps.loc[:,'test_cas']
            final_db = final_db.loc[:,['test_cas','pubchem2d']].drop_duplicates().join(final_db_cllps,on=['test_cas'],lsuffix='_lft')
            final_db = final_db.drop(columns=['test_cas_lft'])
        else:
            final_db = final_db.drop(columns=['fish','smiles'])
        
        fp = pd.DataFrame(final_db.pubchem2d.apply(dsp.splt_str))
        fp = pd.DataFrame(fp.pubchem2d.tolist(), index=fp.index)
        fp.columns = ['pub'+ str(i) for i in range(1,882)]
        
        # data with pubchem2d
        final_db = final_db.drop(columns=['pubchem2d']).join(fp)
        final_db = final_db.drop(columns=['test_cas'])
        
        # Prepare binary or multiclass problem
        if(multiclass):
            final_db = multi_score(final_db)
        else:
            final_db = binary_score(final_db)
        
        # use min max scaler (similar to Simone, and it seems reasonable looking at
        # the distribution of the data)
        #sclr = MinMaxScaler(feature_range=(-1,1))
        sclr = MinMaxScaler()
        
        dummy = dsp.Dataset()
        dummy.setup_data(X=final_db.drop(columns=['score']),
                         y=final_db.loc[:,['score']],
                         split_test=0)
        
        
        if setup != 'onlychem':
            encode_these = ['species','conc1_type','exposure_type','obs_duration_mean',
                            'family','genus','tax_order','class',
                            'application_freq_unit', 'media_type', 'control_type']
            dummy.encode_categories(variables=encode_these, onehot=True)
        
        # Tensorflow does not accept boolean (or int) data, so convert to float
        X = dummy.X_train.astype(np.float64)
        y = dummy.y_train.astype(np.float64)
                
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, 
                                                          test_size = 0.2, 
                                                          shuffle=True, 
                                                          random_state=42,
                                                          stratify=y)
        
        sclr = MinMaxScaler()
        sclr.fit(X_trainval[numerical])
        xtmp = X_trainval.copy()
        xtmp.loc[:,numerical] = sclr.transform(X_trainval.loc[:,numerical])
        X_trainval = xtmp.copy()
        xtmp = X_test.copy()
        xtmp.loc[:,numerical]  = sclr.transform(X_test.loc[:,numerical])
        X_test = xtmp.copy()
        return X_trainval, y_trainval, X_test, y_test, dummy

        
