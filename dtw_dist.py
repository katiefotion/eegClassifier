# Calculates the distance between signals using dynamic time warping

__author__ = "katie"
__date__ = "$Apr 10, 2017 1:44:03 PM$"

from eeg_windowing import *
from math import sqrt

# Calculate distance between two signals s1 and s2
# This method was adapted from http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
def dtw_dist(s1, s2):
    dtw={}

    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
    for i in range(len(s2)):
        dtw[(-1, i)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            dtw[(i, j)] = dist + min(dtw[(i-1, j)],dtw[(i, j-1)], dtw[(i-1, j-1)])

    return sqrt(dtw[len(s1)-1, len(s2)-1])

# Writes result of all pairwise DTW distances to a file 
# NOTE: each pair is only computed once, so matrix is an upper diagonal matrix
def write_dtw_matrix(windowed_data):
    
    dtw_mat = []
    
    # For every window of data (after cue)
    for i in range(0, len(windowed_data)):
        
        row = []
        
        # Calculate the pairwise DTW distance between window i and all other windows 
        for j in range(i+1, len(windowed_data)):
            
            row.append(dtw_dist(windowed_data[i][0], windowed_data[j][0]))
               
            # Print out progress (ends at 200 / 200)
            print(str(i) + '/' +  str(j))
        
        # Append all distances from window i
        dtw_mat.append(row)
        
    # Write results to file
    with open('dtw_matrix.txt', 'w') as f:
        for row in dtw_mat:
            row_str = ''
            for dist in row:
                row_str += '\t' + str(dist)
            f.write(row_str + '\n')

if __name__ == "__main__":

    # The two participants' data files
    postfixes = ['a', 'f']
    
    for postfix in postfixes:
        
        # Access data filenames
        mrk_filename = './BCICIV_1_asc/BCICIV_calib_ds1' + postfix + '_mrk.txt'
        onech_filename = './BCICIV_1_asc/BCICIV_calib_ds1' + postfix + '_cn_1channel.txt'
        
        # Read mrk.txt file and find out when cues were given
        [times, labels] = find_labeled_times(mrk_filename)

        # Based on when cues were given, window the continuous data appropriately
        windowed_data = window_data(onech_filename, times, labels)
        
        # Write DTW distance matrix to file to be accessed later
        write_dtw_matrix(windowed_data)