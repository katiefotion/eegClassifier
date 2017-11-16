# Performs all aspects necessary for k-fold cross validation

__author__ = "katie"
__date__ = "$Apr 10, 2017 2:33:19 PM$"

from eeg_windowing import *
from dtw_dist import *
from extract_features import *
import knn
import random_forest
import svm
import svd as svd_classifier
import random 
import numpy

# Partitions windowed data into k_folds folds
def partition_data(data, k_folds):
    
    partitioned_data = []
    
    # Build structure to hold the folds
    for k in range(0, k_folds):
        empty_list = []
        partitioned_data.append(empty_list)

    # Randomly assign each window of data to a fold
    i = 0
    for data_pt in data:
        partitioned_data[random.randint(0,k_folds-1)].append([data_pt, i])
        i += 1
    
    return partitioned_data

# Reads DTW matrix file and builds matrix structure to hold information
def read_dtw_matrix():
    
    # Read DTW distance file
    with open('dtw_matrix.txt', 'r') as f:
        lines = f.readlines()
        
    # Create empty numpy matrix of correct shape
    dtw_matrix = numpy.empty(shape=(len(lines), len(lines)))
    
    # For each line in the DTW file
    i = 0
    for line in lines:
        
        # Isolate each distance
        elements = line.split('\t')
        
        # Add 0's to beginning of matrix since upper diagonal suffices
        row = []
        for j in range(0, len(lines) - len(elements) + 1):
            row.append(0.0)
        
        # For each distance in file
        for element in elements:
            
            # Make sure \n isn't at end
            el = element.strip('\n')
            
            # If inf, then set to a large distance
            if (el == 'inf'):
                row.append(1000000000.0)
                
            # Otherwise append to matrix
            elif (el != ''):
                row.append(float(el))
             
        # Add newly accessed row to dtw matrix
        dtw_matrix[i,:] = row

        i += 1
            
    return dtw_matrix

# Evaluate the folds of partitioned_data according to classifier selection and feature_set selection
# NOTE:  classifier = 1 KNN
#                   = 2 RF
#                   = 3 SVM
#       feature_set = 1 Raw
#                   = 2 DTW 
#                   = 3 DTW+metrics
def evaluate_folds(partitioned_data, dtw_mat, classifier, parameters, k_folds, feature_set):
    
    # Keep running sum for every fold
    acc_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    
    # For each fold in partitioned_data
    for k_fold in range(0, k_folds):
        
        # Access testing data fold
        testing_data = partitioned_data[k_fold]
        
        # Build training data fold
        training_data = []
        i = 0
        for partition in partitioned_data: 
            if i != k_fold:
                training_data.extend(partition)
            i += 1
            
        # If KNN, access parameter k and get statistics for that fold
        if classifier == 1:
            k = parameters[0]
            [acc, prec, rec] = knn.get_statistics(training_data, testing_data, dtw_mat, k)
            
        # If RF or SVM, and if raw feature set
        elif feature_set == 1:
        
            # If RF, access parameters ntree and mtry and get statistics for that fold
            if classifier == 2:
                ntree = parameters[0]
                mtry = parameters[1]
                [acc, prec, rec] = random_forest.get_statistics_raw(training_data, testing_data, ntree, mtry)

            # If SVM, access parameters C and gamma and get statistics for that fold
            elif classifier == 3:
                C = parameters[0]
                gamma = parameters[1]
                [acc, prec, rec] = svm.get_statistics_raw(training_data, testing_data, C, gamma)            

            else:
                break
        
        # If RF or SVM and if DTW feature set 
        elif feature_set == 2:
            
            # Extract DTW features of training set 
            training_features = extract_dtw_features(training_data, dtw_mat)
            
            # Extract DTW features of testing set
            testing_features = extract_dtw_features(testing_data, dtw_mat)
            
            # If RF, access parameters ntree and mtry and get statistics on DTW feature set for that fold
            if classifier == 2:
                ntree = parameters[0]
                mtry = parameters[1]
                [acc, prec, rec] = random_forest.get_statistics_extracted(training_data, testing_data, training_features, testing_features, ntree, mtry)

            # If SVM, access parameters C and gamma and get statistics on DTW feature set for that fold
            elif classifier == 3:
                C = parameters[0]
                gamma = parameters[1]
                [acc, prec, rec] = svm.get_statistics_extracted(training_data, testing_data, training_features, testing_features, C, gamma)            

            else:
                break
        
        # If RF or SVM, and if DTW+metrics feature set
        elif feature_set == 3:
            
            #Extract DTW+metrics feature set of training and testing data
            training_features = extract_all_features(training_data, dtw_mat)
            testing_features = extract_all_features(testing_data, dtw_mat)
            
            # If RF, access parameters and calculate statistics on DTW+metrics feature set for fold
            if classifier == 2:
                ntree = parameters[0]
                mtry = parameters[1]
                [acc, prec, rec] = random_forest.get_statistics_extracted(training_data, testing_data, training_features, testing_features, ntree, mtry)

            # If SVM, access parameters and calculate statistics on DTW+metrics feature set for fold
            elif classifier == 3:
                C = parameters[0]
                gamma = parameters[1]
                [acc, prec, rec] = svm.get_statistics_extracted(training_data, testing_data, training_features, testing_features, C, gamma)            

            else:
                break
        
        # Accumulate accuracy, precision and recall
        acc_sum += acc
        prec_sum += prec
        rec_sum += rec
      
    # Return averaged accuracy, precision and recall across all k_folds folds
    return [acc_sum/float(k_folds), prec_sum/float(k_folds), rec_sum/float(k_folds)]

if __name__ == "__main__":
        
    # Access mrk file for participant A
    mrk_filename = './BCICIV_1_asc/BCICIV_calib_ds1a_mrk.txt'
    
    # Set number of folds to 10
    k_folds = 10

    # Find times and class labels for each cue
    [times, labels] = find_labeled_times(mrk_filename)

    # Read in DTW matrix
    dtw_matrix = read_dtw_matrix()

    # Start UI
    print('===========================================')
    print('             Classify EEG data             ')
    print('===========================================')
    print('')
    print('Select an option from below:')
    print('1. Perform parameter tuning')
    print('2. Get classification results')
    print('3. Preprocess and fold data')
    print('4. Quit')

    # Get user input regarding options above
    option = raw_input('>>')

    # While user has not selected quit
    while not option.startswith('4'):
        
        # If the user wants to parameter tune or get classification results
        if not option.startswith('3'):
            
            # Ask what channel option they want
            print('')
            print('Would you like to use all channels, only the mean signal across channels, or use SVD to reduce the number of channels?')
            print('1. All channels')
            print('2. Mean channel')
            print('3. SVD on channels')

            # Get user input regarding channel options above
            channel_choice = raw_input('>>')
            
            # Set data file to appropriate name based on channel choice
            # NOTE: files are already folded!
            if channel_choice.startswith('1'):
                filename = 'all_channels_folded.txt'
            elif channel_choice.startswith('2'):
                filename = 'mean_channels_folded.txt'
            elif channel_choice.startswith('3'):
                filename = 'reduced_channels_folded.txt'
            
            # Read that data file
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Detect when folds were written in file and build partitioned_data accordingly
            partitioned_data = []
            fold = []
            for line in lines:
                
                # Detect change in fold
                if line.startswith('\n'):
                    partitioned_data.append(fold)
                    fold = []

                # Otherwise keep writing signals to current fold
                else:
                    values = line.split('\t')
                    
                    # Access class label and true index (for DTW matrix)
                    class_label = int(float(values[len(values)-2]))
                    index = int(float(values[len(values)-1].strip('\n')))
                    
                    # Build signal list
                    signal_str = values[0].strip('[].').split(', ')
                    signal = []
                    for s in signal_str:
                        signal.append(float(s))

                    # Append all information to fold
                    fold.append([[signal, class_label], index])
                   
        # If user wants to parameter tune
        if option.startswith('1'):

            # Ask what classifier they want to tune
            print('')
            print('What classifier would you like to tune?')
            print('1. K-Nearest-Neighbors')
            print('2. Random Forest')
            print('3. Support Vector Machine')

            classifier = raw_input('>>')
            
            # If KNN
            if classifier.startswith('1'):
                
                # Tell them they can't use all 59 channels or 10 SVD channels with KNN (must be mean channel)
                if channel_choice.startswith('1') or channel_choice.startswith('3'):
                    
                    tmp_classifier = classifier
                    
                    # Ask them for a new classifier selection
                    while tmp_classifier.startswith('1'):
                        print('')
                        print('KNN is not an option when more than 1 channel is being used')

                        print('')
                        print('What classifier would you like to tune?')
                        print('1. K-Nearest-Neighbors')
                        print('2. Random Forest')
                        print('3. Support Vector Machine')

                        tmp_classifier = raw_input('>>')
                        
                    classifier = tmp_classifier
             
                # If they did select mean channel
                else:

                    # Ask for range of k for parameter tuning
                    print('')
                    print('What range of k (for KNN) would you like to try? (ex: 1 100)')

                    # Access user input for min k and max k
                    k_neighbors = raw_input('>>').split(' ')
                    min_k = int(float(k_neighbors[0]))
                    max_k = int(float(k_neighbors[1]))

                    # Get parameter tuning results 
                    [acc, prec, rec, k] = knn.param_tune(min_k, max_k, k_folds, partitioned_data, dtw_matrix)

                    # Print optimal k value based on highest F1-measure
                    print('')
                    print('Optimal k: ' + str(k))

            # If want to tune RF
            if classifier.startswith('2'):

                # If mean channel approach
                if channel_choice.startswith('2'):
            
                    # Ask for desired feature set (since any are possible)
                    print('')
                    print('What set of features would you like to use?')
                    print('1. Raw signal')
                    print('2. Dynamic time warped distances')
                    print('3. All extracted features')

                    features = raw_input('>>')
                   
                # If all 59 channels or SVD 10 channels, only raw input is option as feature set
                else:
                    
                    features = '1'

                # Ask for parameter tuning range
                print('')
                print('What range of ntree and mtry (for RF) would you like to try? (ex: 1 100 1 10)')

                # Access input for parameter tuning range
                parameters = raw_input('>>').split(' ')
                min_ntree = int(float(parameters[0]))
                max_ntree = int(float(parameters[1]))
                min_mtry = int(float(parameters[2]))
                max_mtry = int(float(parameters[3]))

                # Depending on feature set selection, parameter tune with appropriate feature set
                if features.startswith('1'):
                    [acc, prec, rec, ntree, mtry] = random_forest.param_tune(min_ntree, max_ntree, min_mtry, max_mtry, dtw_matrix, k_folds, partitioned_data, 1)
                elif features.startswith('2'):
                    [acc, prec, rec, ntree, mtry] = random_forest.param_tune(min_ntree, max_ntree, min_mtry, max_mtry, dtw_matrix, k_folds, partitioned_data, 2)
                elif features.startswith('3'):
                    [acc, prec, rec, ntree, mtry] = random_forest.param_tune(min_ntree, max_ntree, min_mtry, max_mtry, dtw_matrix, k_folds, partitioned_data, 3)

                # Print optimal parameters
                print('')
                print('Optimal ntree: ' + str(ntree) + ', mtry: ' + str(mtry))

            # If want to tune SVM
            elif classifier.startswith('3'):
                
                # If mean channel approach
                if channel_choice.startswith('2'):

                    # Ask for desired feature set (since any are possible)
                    print('')
                    print('What set of features would you like to use?')
                    print('1. Raw signal')
                    print('2. Dynamic time warped distances')
                    print('3. All extracted features')

                    features = raw_input('>>')
                    
                # If all 59 channels or SVD 10 channels, only raw input is option as feature set
                else:
                    features = '1'

                # Depending on feature set selection, parameter tune with appropriate feature set
                if features.startswith('1'):
                    [acc, prec, rec, C, gamma] = svm.param_tune(dtw_matrix, k_folds, partitioned_data, 1)
                elif features.startswith('2'):    
                    [acc, prec, rec, C, gamma] = svm.param_tune(dtw_matrix, k_folds, partitioned_data, 2)
                elif features.startswith('3'):    
                    [acc, prec, rec, C, gamma] = svm.param_tune(dtw_matrix, k_folds, partitioned_data, 3)

                # Print optimal parameters
                print('')
                print('Optimal C: ' + str(C) + ', gamma: ' + str(gamma))

        # If user wants to get classification results
        elif option.startswith('2'):

            # Ask what classifier they want to use
            print('')
            print('What classifier would you like to use?')
            print('1. K-Nearest-Neighbors')
            print('2. Random Forest')
            print('3. Support Vector Machine')

            # Get classifier selection
            classifier = raw_input('>>')

            # If KNN
            if classifier.startswith('1'):
                
                # Ask for value of k that is desired 
                print('')
                print('What value of k (for KNN) would you like to use?')

                # Access input
                k_neighbors = int(float(raw_input('>>')))

                # Evaluate each fold of data using DTW matrix as feature set 
                [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 1, [k_neighbors], k_folds, 0)

            # If RF
            elif classifier.startswith('2'):

                # Ask for ntree and mtry 
                print('')
                print('What value of ntree and mtry (for RF) would you like to use? Ex: 100 10')

                # Access input for ntree and mtry
                parameters = raw_input('>>').split(' ')
                ntree = int(float(parameters[0]))
                mtry = int(float(parameters[1]))

                # If mean channel approach
                if channel_choice.startswith('2'):

                    # Ask for feature set since any are possible
                    print('')
                    print('What set of features would you like to use?')
                    print('1. Raw signal')
                    print('2. Dynamic time warped distances')
                    print('3. All extracted features')

                    features = raw_input('>>')
                   
                # If all 59 channels or 10 SVD channels selected, only raw input is option
                else:
                    features = '1'

                # Evaluate folds appropriately based on feature set selection
                if features.startswith('1'):
                    [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 2, [ntree, mtry], k_folds, 1)
                elif features.startswith('2'):
                    [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 2, [ntree, mtry], k_folds, 2)
                elif features.startswith('3'):
                    [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 2, [ntree, mtry], k_folds, 3)

            # If SVM
            elif classifier.startswith('3'):

                # Ask for C and gamma
                print('')
                print('What value of C and gamma (for SVM) would you like to use? Ex: 100 0.01')

                # Access input for C and gamma
                parameters = raw_input('>>').split(' ')
                C = float(parameters[0])
                gamma = float(parameters[1])

                # If mean channel approach
                if channel_choice.startswith('2'):

                    # Ask for feature set since any are possible
                    print('')
                    print('What set of features would you like to use?')
                    print('1. Raw signal')
                    print('2. Dynamic time warped distances')
                    print('3. All extracted features')

                    features = raw_input('>>')
                    
                # If all 59 channels or 10 SVD channels, only raw input is option
                else:
                    features = '1'

                # Evaluate each fold according to feature set selected
                if features.startswith('1'):
                    [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 3, [C, gamma], k_folds, 1)
                elif features.startswith('2'):
                    [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 3, [C, gamma], k_folds, 2)
                elif features.startswith('3'):
                    [acc, prec, rec] = evaluate_folds(partitioned_data, dtw_matrix, 3, [C, gamma], k_folds, 3)
        
        # If user wants to refold the data
        elif option.startswith('3'):
                        
            # Read all 59 channels data
            all_channels_filename = './BCICIV_1_asc/BCICIV_calib_ds1a_cnt.txt'
            all_channels_data = read_data(all_channels_filename)

            # Read mean channel data
            mean_channels_filename = './BCICIV_1_asc/BCICIV_calib_ds1a_cn_1channel.txt'
            mean_channels_data = read_data(mean_channels_filename)

            # Ask for n_components for SVD
            print('')
            print('How many features would you like in the reduced signal?')
            n_components = int(float(raw_input('>>')))

            # Print warning about time it takes
            print('')
            print('Performing singular value decomposition. This may take a moment...')

            # Perform SVD on channels
            expanded_data = svd_classifier.build_samplings_matrix('./BCICIV_1_asc/BCICIV_calib_ds1a_cnt.txt')
            [svd_data, svd] = svd_classifier.reduce_dimensionality(expanded_data, n_components)

            # Set 3 data sources to be iterable
            data_sources = [all_channels_data, mean_channels_data, svd_data]
            
            # Iterate over each form of data
            i = 0
            for data in data_sources:
                
                # Set filename to appropriate name 
                if i == 0:
                    filename = 'all_channels_folded.txt'
                elif i == 1:
                    filename = 'mean_channels_folded.txt'
                elif i == 2:
                    filename = 'reduced_channels_folded.txt'
                
                # Window the data
                windowed_data = window_data(data, times, labels)

                # Fold the data
                partitioned_data = partition_data(windowed_data, k_folds)

                # Save results to list
                lines = []
                for fold in partitioned_data:
                    for sampling in fold:
                        lines.append(str(sampling[0][0]) + '\t' + str(sampling[0][1]) + '\t' + str(sampling[1]) + '\n')
                    lines.append('\n')

                # Write results to file
                with open(filename, 'w') as f:
                    for line in lines:
                        f.write(line)
                        
                i += 1
                
        # As long as the user did not want to refold the data
        if not option.startswith('3'):
        
            # Print statistics from whatever was done previously
            print('Accuracy: ' + str(acc))
            print('Precision: ' + str(prec))
            print('Recall: ' + str(rec))
            print('F1-Measure: ' + str((2.0*prec*rec)/(prec+rec)))

        # Redisplay main menu options
        print('')
        print('Select an option from below:')
        print('1. Perform parameter tuning')
        print('2. Get classification results')
        print('3. Preprocess and fold data')
        print('4. Quit')

        option = raw_input('>>')