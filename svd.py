# Reduces dimensionality of channels using singular value decomposition (SVD) 
# from 59 channels to n (usually 10)

import numpy
from sklearn.decomposition import TruncatedSVD

__author__ = "katie"
__date__ = "$April 20, 2017 2:06:39 PM$"

# Create a matrix for raw data (each row is a sampling and each column is a channel)
def build_samplings_matrix(filename):

    # Read file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize empty matrix of appropriate shape
    samplings_matrix = numpy.empty(shape=(len(lines), len(lines[0].split('\t'))))
    
    # For every sampling in data
    i = 0
    for line in lines:
        
        # Access values from all 59 channels
        samplings_str = line.split('\t')
        samplings = []
        for val in samplings_str:
            samplings.append(float(val.strip('\n')))

        # Add that row to samplings_matrix
        samplings_matrix[i] = samplings
        i += 1

    return samplings_matrix

# Reduce dimensionality of samplings matrix from 59 channels to n using SVD
def reduce_dimensionality(matrix, n):
    
    # Perform dimensionality reduction
    svd = TruncatedSVD(n_components=n)
    reduced_mat = svd.fit_transform(matrix) 
        
    return [reduced_mat, svd]