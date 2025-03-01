# Description:

Data combined from multiple files to avoid repeated computation.

# Content:

1. Files:
    * gaia_lm_m_stars.parquet: Mixed dataset of massive and low mass stars with their XP spectra. The dataset is of the shape (17627, 7) and has the features ['source_id', 'teff_gspphot', 'logg_gspphot', 'mh_gspphot',
       'spectraltype_esphs', 'Cat', 'flux']
    * LM_mass_flame.parquet: Dataset of 10,000 low-mass stars sourced from the Gaia archive. The dataset has the shape (10000, 4) and has the features ['source_id', 'flux', 'spectraltype_esphs', 'mass_flame']
    * LM_stars.parquet: Same as LM_mass_flame without the 'mass_flame' feature.