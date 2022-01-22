#!/bin/bash

# This script downloads the data from the repositories and creates the necessary directory structure.



ETname='ecotox_ascii_09_15_2021' # name of the downloaded EcoTox file

cd ..
mkdir -p data/raw/
cd data 



########################
# DOWNLOAD FROM ECOTOX #
########################


# If the data from EcoTox has not been downloaded yet, download it
if [ "$(ls $ETname 2>/dev/null)" == "" ]
then
	wget --no-check-certificate https://gaftp.epa.gov/ecotox/${ETname}.zip
	unzip $ETname
fi

# We assume that there is only one

cp $ETname/results.txt raw/
cp $ETname/tests.txt raw/
cp $ETname/validation/species.txt raw/



#########################
# DOWNLOAD FROM COMPTOX #
#########################

wget ftp://newftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard/DSSTox_Predicted_NCCT_Model.zip
unzip DSSTox_Predicted_NCCT_Model.zip



