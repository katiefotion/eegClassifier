# This program splits the EEG data into windows,
# starting at the time step when the cue is given
# ending at the time step before the next cue

__author__ = "katie"
__date__ = "$Apr 10, 2017 12:52:48 PM$"

# Finds times and labels for each cue
def find_labeled_times(filename):
    
    # Read mrk file
    with open(filename, 'r') as f:
        
        lines = f.readlines()
        
    times = [] 
    labels = []
    
    # For each sampling
    for line in lines:
        
        # If line is not null
        if line: 
        
            # Get info about that cue
            line_info = line.split('\t')

            # Get time and class label
            time = int(float(line_info[0]))
            label = int(float(line_info[1]))
            times.append(time)
            labels.append(label)
            
    return [times, labels]

# Reads data and saves in matrix
def read_data(filename):
    
    # Read data file
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    data = []
    
    # For every sample point
    for line in lines:
        
        # Get all channel values
        values = line.split('\t')
        
        sampling = []
        
        # For each channel value, add it to sampling list
        for val in values:
            sampling.append(float(val.strip('\n')))
            
        # Append row of 59 channel data to matrix of all data
        data.append(sampling)
        
    return data
    
# Window data beginning at time of cue and ending one sampling before next cue
def window_data(data, times, labels):

    windows = []
     
    # Access first time
    prev_time = times[0]
    
    # Access number of samplings that occur between the first two cues
    num_sample_points = times[1] - prev_time
    
    # For every cue
    for j in range(1, len(times)):
        
        # Access time of cue and class label of cue
        time = times[j]
        label = labels[j-1]
        
        signal = []
        
        # For signal starting one sampling after end of last cue and running for num_sample_points
        for i in range(prev_time+1, prev_time+num_sample_points):
            
            # Access and save signal values
            signal_values = data[i]
            for val in signal_values:
                signal.append(val)
            
        # Set prev_time to time of current cue
        prev_time = time
        
        # Append signal and class label to list of windows
        windows.append([signal, label])
    
    signal = []
    
    # Add the last segment of signal (after last cue)
    for i in range(prev_time+1, prev_time+num_sample_points):
        
        signal_values = data[i]
        for val in signal_values:
            signal.append(val)
        
    # Append last segment to windows
    windows.append([signal, labels[len(labels)-1]])
    
    return windows

        