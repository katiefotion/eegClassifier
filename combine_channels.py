# Finds the mean channel over all 59 channels to produce a single signal per sampling

__author__ = "katie"
__date__ = "$Apr 10, 2017 12:55:50 PM$"

# Writes mean channel data to file with ending '_1channel.txt'
def write_mean(filename):
    
    # Read data
    with open(filename, 'r') as f:
        
        lines = f.readlines()
        
    means = []
 
    # Iterate through each sampling
    for line in lines: 
        
        # Divide data into the 59 channels
        channels = line.strip('\n').split('\t')
        
        mean = 0
        
        # Compute the mean over all 59 channels
        for channel in channels:
            mean += int(channel)    
        mean = mean / len(channels)
        
        # Keep list of all means
        means.append(mean)
    
    # Write 1 channel signal to file
    outfilename = filename.rstrip('.txt') + '_1channel.txt'
    with open(outfilename, 'w') as f:
        for mean in means: 
            f.write(str(mean) + '\n')

if __name__ == "__main__":
    
    # For both participants' data
    postfixes = ['a', 'f']
    for postfix in postfixes:
        
        # Get filename of original data (with 59 channels)
        cnt_filename = './BCICIV_1_asc/BCICIV_calib_ds1' + postfix + '_cnt.txt'
        
        # Calculate mean channel and write to new file
        write_mean(cnt_filename)
        
        
