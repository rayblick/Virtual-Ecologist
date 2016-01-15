"""
Virtual-Ecologist

This module evaluates plant survey data collected in plots along a transect.
First the error rate that is expected between two people is evaluated for
different lifeforms (e.g. grass, shrubs, trees). The model is then used to
examine a large dataset collected by one person to calculate the minimum
detectable difference as plots are reduced from each transect.

--- You need two datasets ---
1) Pilot data: A csv file with NO header.
    3 columns:
    observer 1 estimate (e.g. 54),
    observer 2 estimate (e.g. 60),
    functional group name (e.g. grass, tree)

2) Full dataset: A csv file collected by one observer WITH header.
    5 columns:
    Wetland name (header title: Wetland)
    Functional group name (header title: FuncGroup) <= names should be in file1
    Transect identity (header title: Transect_ID)
    Plot identity (header title: Plot_ID)
    Percentage cover estimate (header title: Cover) <= units should match file1

--- The virtual ecologist ---
File one is used to train the virtual ecologist. If no training data is
available you can skip the initial stage and accept an error rate between
participants of 10% (One standard deviation).

--- The functional groups ---
If you have pilot data for one functional group, you can assign the same error
rate for all other functional groups in the large dataset. This might not
be sensible (i.e. estimating grass cover may be more accurate than canopy
cover of trees). If you have pilot data for several functional groups but not
all of them are represented in the full dataset, the ones without data are
assigned the average value of the error rate of all other groups.

Created on Thu Mar 19 12:30:56 2015
Last updated Fri Jan 15 2016
Author: Ray Blick
"""

# import modules
import csv

def train_observer(csv_filename):
    """
    Returns a dictionary containing Mean Square Error of estimates.
    Input is a csv with 3 columns: observer 1 estimates, observer 2
    estimates and functional group names.

    # Test normal case
    >>> train_observer("ve_testdata.csv")
    {'grass': 13.090909090909092, 'shrubs': 27.2, 'trees': 13.153846153846153}

    # Test no arguments
    >>> train_observer()
    Traceback (most recent call last):
    ...
    TypeError: train_observer() takes exactly 1 argument (0 given)

    # Test numeric argument
    >>> train_observer(23)
    Traceback (most recent call last):
    ...
    TypeError: coercing to Unicode: need string or buffer, int found
    """

    # store count of functional groups
    fg_dict = dict()
    # store Mean Square Error output
    mse_output = dict()

    with open(csv_filename, 'r') as f:
        file_reader = csv.reader(f)
        for row in file_reader:

            # assignment of columns in each row
            est_one = row[0]
            est_two = row[1]
            fg_key = row[2]

            # record the number of entries to the dictionary
            if fg_key not in fg_dict:
                fg_dict[fg_key] = 1
            else:
                fg_dict[fg_key] += 1

            # Calculate square difference between observers
            square_difference = (float(est_one) - float(est_two)) ** 2

            # add to global dictionary
            mse_output[fg_key] = mse_output.get(fg_key, 0) + square_difference

        # Calculate mean of error between observers
        # divide one dictionary by another
        for entry in mse_output:
            if entry in fg_dict:
                mse_output[entry] = mse_output.get(entry, 0) / fg_dict[entry]

        return mse_output

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    test_output = train_observer("ve_testdata.csv")
    print test_output
