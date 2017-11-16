# This program creates new text files to be used for visualization in 
# Tableau by combining all information into a single file  (ONLY RUN ONCE)
# Makes data file include time of cue, class of cue, and all signals

__author__ = "katie"
__date__ = "$Mar 27, 2017 6:42:28 PM$"

# Get time at which each cue was given
def get_time_labels(mrk_filename):
    
    # Read mrk file
    with open(mrk_filename, 'r') as f:
        lines = f.readlines()
        
    labels = []
    
    # For every sampling
    for line in lines:
        
        # Get the time the cue was given 
        labels.append(line.split('\t')[0])
        
    return labels

# Get english names of two classes for each file (was used to determine which participants did hand/foot) 
def determine_classes(filename):
    
    # Read nfo file
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # Get the two true labels for the experiment
    class_line = lines[1].rstrip('\n').split(' ')
    
    return [class_line[1].rstrip(','), class_line[2]]
    

# Get true class label of each cue (1 and -1)
def get_class_labels(mrk_filename, nfo_filename):
    
    # First determine the two classes of the experiment
    classes = determine_classes(nfo_filename)
    
    # Then read the class labels (in form 1 and -1)
    with open(mrk_filename, 'r') as f:
        lines = f.readlines()
    
    labels = []
    
    # For each sample point
    for line in lines: 
        
        # Read if the cue is 1 or -1
        class_indicator = line.rstrip('\n').split('\t')[1]
        
        # If -1, refer back to result from determine_classes to see what that means in English
        if int(float(class_indicator)) == -1:
            labels.append(classes[0])
        
        # If 1, do the same
        else:
            labels.append(classes[1])
            
    return labels

# Create new data file to include times and class labels (for visualization in Tableau)
def add_labels(filename, times, classes):
    
    # Read data file
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # Access time of first sampling
    curr_time_label = int(float(times[0]))
    
    i = 0
    j = 0
    
    outfilename = filename.rstrip('.txt') + '_labeled.txt'
    
    # Write time and class labels to file along side data
    with open(outfilename, 'w') as f:
        
        # For every sampling
        for line in lines:

            # Check if time label corresponds to line number (since that is what time means)
            if curr_time_label == i:

                # If we have reached the time label, write the time and cue's class next to signal
                f.write(str(i) + '\t' + classes[j] + '\t' + line)

                j += 1

                # Update curr_time_label to reflect the time of next cue
                if j < len(times)-1:
                    curr_time_label = int(float(times[j]))
                # If we've reached the last cue, set curr_time_label to 0
                else:
                    curr_time_label = 0

            # If this is not the time of a cue
            else:

                # Write ? as true class label
                f.write(str(i) + '\t?\t' + line)

            i += 1
    
if __name__ == "__main__":
    
    # For each participants' data
    postfixes = ['a', 'f']
    
    for postfix in postfixes:
        
        # Access data filenames
        cnt_filename = './BCICIV_1_asc/BCICIV_calib_ds1' + postfix + '_cnt.txt'
        mrk_filename = './BCICIV_1_asc/BCICIV_calib_ds1' + postfix + '_mrk.txt'
        nfo_filename = './BCICIV_1_asc/BCICIV_calib_ds1' + postfix + '_nfo.txt'
        
        # Get time labels of cues
        time_labels = get_time_labels(mrk_filename)
        
        # Get class labels of cues
        class_labels = get_class_labels(mrk_filename, nfo_filename)
        
        # Create new file with time and class labels included with data
        add_labels(cnt_filename, time_labels, class_labels)
        
    