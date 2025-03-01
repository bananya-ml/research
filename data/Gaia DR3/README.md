# Description

Contains all data sourced from Gaia DR3 divided into relevant sub-directories.

# Content

1. Directories: 
    * combined: Multiple files of data combined for ease of usage and stored to avoid repeated computation
    * external: Data from external publications, where only data or some results may be relevant
    * queries: Results from Gaia ADQL queries
    * spectra: XP and RVS spectra data from the Datalink service
    * splits: train and test splits for scripts that might require it

2. Files:
    * eda.ipynb: data exploration and cleaning of gaia data, usually placed into the 'combined' directory after processing
    * README.md: This file