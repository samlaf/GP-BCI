import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd

mat = loadmat('EmgResponse.mat')  # load mat-file
mdata = mat['EmgResponse'][0,0]  # variable in mat file
mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
# * SciPy reads in structures as structured NumPy arrays of dtype object
# * The size of the array is the size of the structure array, not the number
#   elements in any particular field. The shape defaults to 2-dimensional.
# * For convenience make a dictionary of the data using the names from dtypes
# * Since the structure has only one element, but is 2-D, index it at [0, 0]
ndata = {n: mdata[n][0, 0] for n in mdtype.names}
# Reconstruct the columns of the data table from just the time series
# Use the number of intervals to test if a field is a column or metadata
columns = [n for n, v in ndata.items()]
# now make a data frame, setting the time stamps as the index
df = pd.DataFrame(np.concatenate([ndata[c].T for c in columns], axis=1),
                  index=ndata['Time'][0],
                  columns=columns)
# df is (640x10), because 640 different stimulations, and 8 emg
# responses (+ 2 header columns)


# Build our dataset
# For each stimulation channel, we have around 20 stimulations
# With each a different emg response
# Each has 31 points of response (taken at ~100hz, this represents
# [-150ms,150ms] window)
# We take the max over these emg response windows for each of
# stim, and take the mean of these maxs
means = np.zeros(32)
for schan in range(32):
    # schan+1 because channels are indexed starting from 1
    responses = df[df['StimChan'] == schan+1]['chan3']
    # We can change these to take max.max, max.mean, mean.max, etc.
    mean = responses.apply(lambda x: x[0].max()).mean()
    means[schan] = mean
plt.plot(means)
plt.show()
