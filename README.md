## WAPCN

This repository contains the core code of WAPCN.

WAPCN.py: WAPCN model with DF as the feature extractor.

DF_Model.py: DF model.

lmmd.py: MMD of each website class. 

main.py: Model training.



Data preprocessing is not included in the code. In the main.py, it is assumed that the network traces and labels (both Numpy arrays) for training have been obtained.  For information about the dataset and data preprocessing, see https://github.com/jpcsmith/wf-in-the-age-of-quic.

