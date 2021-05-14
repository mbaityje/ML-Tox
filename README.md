# ML-Tox

A code developed and maintained by S. D'Ambrosi, J. Wu and M. Baity-Jesi.

## Installations

#### Python
Version: xxx

Required Packages: numpy, 


## Data download

### Main database (ECOTOX Knowledge)
The data on the experiments is downloaded from Ecotox. Documentation on the dataset can be found at https://cfpub.epa.gov/ecotox/

Download the entire database as an ASCII file from the website, or do it from command line through

```
wget https://gaftp.epa.gov/ecotox/ecotox_ascii_03_15_2021.zip # This is for the version of 15 March 2021
unzip ecotox_ascii_03_15_2021.zip # Decompress
mv ecotox_ascii_03_15_2021 data/ # Move it to the data directory
```
If having problems with certificates, you can try adding the `--no-check-certificate` flag.






### Chemicals

From the CAS identifiers provided in ECOTOX, we need to extract the SMILES codes.


SMILES:


From the SMILES, we extract Pubchem2D and several chemical properties.

PUBCHEM2D:

Properties:



