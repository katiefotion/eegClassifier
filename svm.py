# Provides classification and parameter tuning functions for support vector machine (SVM)

# Uses scikit-learns svm implementation

import k_fold_crossvalidation
from sklearn import svm as svm_classifier
import numpy

__author__ = "katie"
__date__ = "$April 19, 2017 4:04:21 PM$"

# Build list of class labels from training data (for training SVM)
def build_class_labels(train):
    
    class_labels = []
    
    # For each training window
    for i in range(0, len(train)):
        
        # Access its class label and add it to the list
        class_label = train[i][0][1]
        class_labels.append(class_label)
        
    # This list is a parameter of sklearn's SVM implementation
    return class_labels

# Train SVM using training samples, class labels, parameters, and specification of feature set
def train_svm(train, class_labels, C, gamma, feature_set):
    
    # If raw signal was chosen as feature set
    if feature_set == 1:
        
        # Initialize an empty matrix for all training samples
        training_samples = numpy.empty(shape=(len(train), len(train[0][0][0])))
        
        # Copy training data into matrix to be used as input into sklearn's SVM implementation
        i = 0
        for t in train:
            training_samples[i, :] = t[0][0]
            i += 1
    
    # If DTW matrix chosen as feature set
    elif feature_set == 2:
        
        # Initialize empty matrix for all elements of DTW matrix (passed in as train)
        training_samples = numpy.empty(shape=(len(train), len(train[0])))
        
        # Copy all rows of DTW matrix in as each training sample
        i = 0
        for t in train:
            training_samples[i, :] = t
            i += 1
    
    # Build SVM classifier
    svc = svm_classifier.SVC(C=C, gamma=gamma)
    svm = svc.fit(training_samples, class_labels)
    
    return svm

# Get statistics of classification when raw signal was feature set
def get_statistics_raw(train, test, C, gamma):
    
    # Build class labels to be used as input into train_svm
    class_labels = build_class_labels(train)
    
    # Train svm based on features in train and class labels
    svm = train_svm(train, class_labels, C, gamma, 1)
    
    # Initialize true/false positives and true/false negatives to 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # For every testing sample
    for i in range(0, len(test)):
        
        # Access true label of testing sample
        true_label = test[i][0][1]
        
        # Create 1D matrix consising of just signal 
        test_sample = numpy.matrix(test[i][0][0])
        
        # Predict class label of test_sample using trained svm
        pred_label = svm.predict(test_sample)           
        
        # Calculate true/false positives and true/false negatives
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 1 and pred_label == -1:
            fn += 1
        elif true_label == -1 and pred_label == 1: 
            fp += 1
        elif true_label == -1 and pred_label == -1:
            tn += 1
    
    # Calculate final accuracy, precision, and recall
    acc = float(tp+tn+1)/float(tp+tn+fp+fn+1)
    prec = float(tp+1)/float(tp+fp+1)
    rec = float(tp+1)/float(tp+fn+1)
    
    return [acc, prec, rec]

# Get statistics of classification when DTW matrix or DTW+metrics was feature set
def get_statistics_extracted(train, test, train_features, test_features, C, gamma):
    
    # Build class labels to be used as input into train_svm
    class_labels = build_class_labels(train)
    
    # Train svm based on train_features and class labels
    svm = train_svm(train_features, class_labels, C, gamma, 2)
    
    # Initialize true/false positives and true/false negatives to 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # For each testing sample
    for i in range(0, len(test)):
        
        # Access true label of testing sample
        true_label = test[i][0][1]
        
        # Create 1D matrix consising of just signal in test_features
        test_sample = numpy.matrix(test_features[i])
        
        # Predict class label of test_sample using trained svm
        pred_label = svm.predict(test_sample)           
        
        # Calculate true/false positives and true/false negatives
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 1 and pred_label == -1:
            fn += 1
        elif true_label == -1 and pred_label == 1: 
            fp += 1
        elif true_label == -1 and pred_label == -1:
            tn += 1
    
    # Calculate final accuracy, precision, and recall
    acc = float(tp+tn+1)/float(tp+tn+fp+fn+1)
    prec = float(tp+1)/float(tp+fp+1)
    rec = float(tp+1)/float(tp+fn+1)
    
    return [acc, prec, rec]

# Parameter tune random svm for C and gamma (no specified ranges because there is standard way of doing it)
def param_tune(dtw_mat, k_folds, partitioned_data, feature_set):
    
    # Initialize variables to very small values
    optimal_C = -1
    optimal_gamma = -1
    largest_f1 = -1.0
    largest_acc = -1.0
    largest_prec = -1.0
    largest_rec = -1.0
    
    # Define search ranges for C and gamma
    C_range = numpy.logspace(-2, 10, 13)
    gamma_range = numpy.logspace(-9, 3, 13)
    
    # Perform grid search on parameters
    for C in C_range:
        for gamma in gamma_range:
        
            # Print current iteration since it takes so long
            print('Trying C = ' + str(C) + ', gamma = ' + str(gamma))
            
            # Perform k-fold cross validation using those particular parameters
            [acc, prec, rec] = k_fold_crossvalidation.evaluate_folds(partitioned_data, dtw_mat, 3, [C, gamma], k_folds, feature_set)
            
            # Calculate resulting f1-measure
            f1 = (2*prec*rec)/(prec+rec)
            
            # If new f1-measure is larger than previously largest f1, replace every statistic with this iteration
            if f1 > largest_f1:
                largest_f1 = f1
                largest_acc = acc
                largest_prec = prec
                largest_rec = rec
                optimal_C = C
                optimal_gamma = gamma
            
    # Return statistics from optimal parameters
    return [largest_acc, largest_prec, largest_rec, optimal_C, optimal_gamma]