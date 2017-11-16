# Provides classification and parameter tuning functions for random forest

# Uses scikit-learns RandomForestClassifier implementation

import k_fold_crossvalidation
from sklearn.ensemble import RandomForestClassifier 
import numpy

__author__ = "katie"
__date__ = "$April 19, 2017 4:04:21 PM$"

# Build list of class labels from training data (for training the forest)
def build_class_labels(train):
    
    class_labels = []
    
    # For each training window
    for i in range(0, len(train)):
        
        # Access its class label and add it to the list
        class_label = train[i][0][1]
        class_labels.append(class_label)
     
    # This list is a parameter of sklearn's RF implementation
    return class_labels

# Train forest using training samples, class labels, parameters, and specification of feature set
def train_forest(train, class_labels, ntree, mtry, feature_set):
    
    # If raw signal was chosen as feature set
    if feature_set == 1:
        
        # Initialize an empty matrix for all training samples
        training_samples = numpy.empty(shape=(len(train), len(train[0][0][0])))
        
        # Copy training data into matrix to be used as input into sklearn's RF implementation
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
        
    # Train forest using specified parameters
    forest = RandomForestClassifier(n_estimators = ntree, max_features=mtry)
    forest = forest.fit(training_samples, class_labels)
    
    return forest

# Get statistics of classification when raw signal was feature set
def get_statistics_raw(train, test, ntree, mtry):
    
    # Build class labels to be used as input into train_forest
    class_labels = build_class_labels(train)
    
    # Train forest based on features in train and class labels
    forest = train_forest(train, class_labels, ntree, mtry, 1)
    
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
        
        # Predict class label of test_sample using trained forest
        pred_label = forest.predict(test_sample)           
        
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
def get_statistics_extracted(train, test, train_features, test_features, ntree, mtry):
    
    # Build class labels to be used as input into train_forest
    class_labels = build_class_labels(train)
    
    # Train forest based on train_features and class labels
    forest = train_forest(train_features, class_labels, ntree, mtry, 2)
    
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
        
        # Predict class label of test_sample using trained forest
        pred_label = forest.predict(test_sample)           
        
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

# Parameter tune random forest for ntree and mtry in specified ranges
def param_tune(min_ntree, max_ntree, min_mtry, max_mtry, dtw_mat, k_folds, partitioned_data, feature_set):
    
    # Initialize variables to very small values
    optimal_ntree = -1
    optimal_mtry = -1
    largest_f1 = -1.0
    largest_acc = -1.0
    largest_prec = -1.0
    largest_rec = -1.0
    
    # Perform grid search on parameters
    for ntree in range(min_ntree, max_ntree+1):
        for mtry in range(min_mtry, max_mtry+1):
        
            # Print current iteration since it takes so long
            print('Trying ntree = ' + str(ntree) + ', mtry = ' + str(mtry))
            
            # Perform k-fold cross validation using those particular parameters
            [acc, prec, rec] = k_fold_crossvalidation.evaluate_folds(partitioned_data, dtw_mat, 2, [ntree, mtry], k_folds, feature_set)
            
            # Calculate resulting f1-measure
            f1 = (2*prec*rec)/(prec+rec)

            # If new f1-measure is larger than previously largest f1, replace every statistic with this iteration
            if f1 > largest_f1:
                largest_f1 = f1
                largest_acc = acc
                largest_prec = prec
                largest_rec = rec
                optimal_ntree = ntree
                optimal_mtry = mtry
            
    # Return statistics from optimal parameters
    return [largest_acc, largest_prec, largest_rec, optimal_ntree, optimal_mtry]