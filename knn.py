# Provides classification and parameter tuning functions for k-nearest neighbors

# Implemented by hand!

from k_fold_crossvalidation import *

__author__ = "katie"
__date__ = "$Apr 10, 2017 2:33:19 PM$"

# Classification of all samples in testing with respect to train and dtw_mat
def get_statistics(train, test, dtw_mat, k):
    
    # Set true positives, true negatives, false positives, false negatives to 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # Iterate through each test sample
    for i in range(0, len(test)):
        
        # Access original index of test sample (for DTW matrix)
        test_index = test[i][1]
        
        # Access true label of test sample
        true_label = test[i][0][1]
        
        # Keep dictionary of k (or fewer) nearest neighbors
        nearest_neighbors = {}
        
        # For every training sample
        for j in range(0, len(train)):          # j tracks index of training samples
            
            # Access original index of training sample (for DTW matrix)
            train_index = train[j][1]
            
            # Ensure you are accessing the upper diagonal of the matrix
            if test_index < train_index:
                dist = dtw_mat[test_index][train_index]
            elif train_index < test_index: 
                dist = dtw_mat[train_index][test_index]
                
            # If for some reason you get test_index = train_index, make them very different from one another 
            else:
                dist = 10000000.0
                
            # If dist is less than any of the nearest neighbors being tracked
            if dist < any(nearest_neighbors.values()):

                # If there are already k near neighbors
                if len(nearest_neighbors) == k:
                    
                    # Delete the smallest 
                    min_key = min(nearest_neighbors, min_key=nearest_neighbors.get)
                    del nearest_neighbors[min_key]

                # Enter dist into nearest neighbors, where j represents the test index
                nearest_neighbors[j] = dist
        
        pos_votes = 0       # votes for class 1
        neg_votes = 0       # votes for class -1 
        
        # For every final near neighbor
        for neighbor_key in nearest_neighbors:
            
            # Check if neighbor is voting for 1 or -1
            if train[neighbor_key][0][1] == 1:
                pos_votes += 1
            elif train[neighbor_key][0][1] == -1:
                neg_votes +=1 
                
        # If more neighbors vote for class 1, pred_label = 1
        if pos_votes > neg_votes:
            pred_label = 1
        # If more neighbors vote for class -1, pred_label = -1
        else:
            pred_label = -1
        
        # Calculate true positives, true negatives, false positives and false negatives
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 1 and pred_label == -1:
            fn += 1
        elif true_label == -1 and pred_label == 1: 
            fp += 1
        elif true_label == -1 and pred_label == -1:
            tn += 1
            
    # Calculate accuracy, precision, and recall for that set of train and test
    acc = float(tp+tn+1)/float(tp+tn+fp+fn+1)
    prec = float(tp+1)/float(tp+fp+1)
    rec = float(tp+1)/float(tp+fn+1)
    
    return [acc, prec, rec]

# Tune KNN for parameter k based on largest F1-measure
def param_tune(min_k, max_k, k_folds, partitioned_data, dtw_matrix):
    
    # Initialize variables to very small values
    optimal_k = -1
    largest_f1 = -1.0
    largest_acc = -1.0
    largest_prec = -1.0
    largest_rec = -1.0
    
    # For every k that you are trying 
    for k in range(min_k, max_k+1):
        
        # Print which parameter you are at since it takes a while
        print('Trying k = ' + str(k))
        
        # Evaluate all folds of partitioned_data using that k
        [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 1, [k], k_folds, 0)
        
        # Calculate the f1 measure as a result
        f1 = (2*prec*rec)/(prec+rec)
        
        # If f1 from this k is larger than largest_f1, replace it
        if f1 > largest_f1:
            largest_f1 = f1
            largest_acc = acc
            largest_prec = prec
            largest_rec = rec
            optimal_k = k
            
    # Return final statistics from optimal k
    return [largest_acc, largest_prec, largest_rec, optimal_k]