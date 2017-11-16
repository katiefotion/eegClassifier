# Extract features to be used in classifiers 

__author__ = "katie"
__date__ = "$April 20, 2017 9:36:18 AM$"

# Calculate min and max of signal
def get_min_max(sample):
    
    # Access signal
    signal = sample[0][0]
    min = 10000000
    max = -10000000
    
    # For each value in signal
    for val in signal:
        
        # Calculate max and min
        if val > max:
            max = val
        elif val < min:
            min = val
        
    return [min, max]

# Calculate sample mean of signal
def get_sample_mean(sample):
    
    # Access signal
    signal = sample[0][0]
    
    mean_sum = 0.0
    
    # For each value in signal
    for sampling in signal:
        
        # Calculate mean sum
        mean_sum += sampling

    return mean_sum/float(len(signal))

# Use DTW matrix + measurements of signal as features
def extract_all_features(data, dtw_mat):
    
    all_features = []
    
    # For each data sampling
    for sample in data:
        
        # Access original index 
        # (not in folds, but in original file because that is where it will be found in the DTW matrix)
        original_index = sample[1]
        
        # Access row of dtw_matrix corresponding to sample
        features = dtw_mat[original_index].tolist()
        
        # Calculate min and max of sample
        [min, max] = get_min_max(sample)
        
        # Calculate sample mean of sample
        sample_mean = get_sample_mean(sample)
        
        # Append all features 
        features.append(min)
        features.append(max)
        features.append(sample_mean)
        all_features.append(features)
        
    return all_features
        
# Use only DTW matrix as features
def extract_dtw_features(data, dtw_mat):
    
    dtw_features = []
    
    # For each data sample
    for sample in data:
        
        # Access original index 
        # (not in folds, but in original file because that is where it will be found in the DTW matrix)
        original_index = sample[1]
        
        # Access row of dtw_matrix corresponding to sample
        features = dtw_mat[original_index].tolist()
    
        # Append DTW results to features list
        dtw_features.append(features)
        
    return dtw_features
        
