# eegClassifier
Classifier for hand vs. foot motion of continuous EEG data

Compile/Run Instructions:

1.  Ensure that scikit-learn and numpy are installed on your machine
    http://scikit-learn.org/stable/install.html 
    https://www.scipy.org/scipylib/download.html
    (Note: see report for specifics regarding what was borrowed and what was implemented)
2.  Ensure that both parts of src folder (from separated submissions) have been placed into the same folder
3.  Navigate to the single, combined src folder in command line
4.  Run k_fold_crossvalidation.py in the following manner:
       >> python2.7 k_fold_crossvalidation.py
5.  Wait for command line UI to appear and follow prompts:
       1.  Perform parameter tuning 
           (Note: this takes a while for most of the classifiers)
       2.  Get classification results 
       3.  Preprocess and fold data 
           (Note: this has already been done and saved in text files, though you could run it again if you'd like)


Description of data files:
• All files in BCICIV_1_asc are original data files from http://www.bbci.de/competition/iv/desc_1.html (aside from the two files ending in _1channel.txt, which are the result of calculating the mean channel for each of the two participants)
• The data has been pre-folded into the 3 variations of channel configurations in the following files: all_channels_folded.txt, mean_channels_folded.txt and reduced_channels_folded.txt
• The results of calculating dynamic time warping distance (DTW) for each pair of windows can be found in dtw_matrix.txt

