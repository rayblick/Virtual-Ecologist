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
import pandas as pd

# global variable

class VirtualEcologist:
    """
    A class of methods to reduce the number of plots along transect lines.
    """

    def __init__(self, pilot_data, full_data):
        self.pilot_data = pilot_data # input file 1
        self.full_data = full_data # input file 2
        self.mse_output = {} # built with train_observer
        self.data_output = {} # built with describe_dataset


    def train_observer(self):
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

        with open(self.pilot_data, 'r') as f:
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
                self.mse_output[fg_key] = self.mse_output.get(fg_key, 0) + square_difference

            # Calculate mean of error between observers
            # divide one dictionary by another
            for entry in self.mse_output:
                if entry in fg_dict:
                    self.mse_output[entry] = self.mse_output.get(entry, 0) / fg_dict[entry]

            # returns a dictionary of MSE
            return self.mse_output


    def match_full_dataset(self):
        """
        Updates the dictionary of Mean Square Error rates. If all functional
        groups (FG) are missing, each FG is assigned 10%. If only some FG's
        are missing, each FG is assigned a mean value based on the pilot data.
        """
        # local count dictionary
        count_dict = dict()
        # holds functional group names not in the training dataset
        list_of_groups = []

        # import dataset
        self.dataset = pd.read_csv(self.full_data)

        # count the frequency each functional group occurs in the full dataset
        for row in self.dataset['FuncGroup']:
            if row not in count_dict:
                count_dict[row] = 1
            else:
                count_dict[row] += 1
        number_of_groups = len(count_dict)

        # populate a list with groups not trained
        # matches keys of two dictionaries
        for entry in count_dict:
            if entry not in self.mse_output:
                # hold missing functional groups in a list
                list_of_groups.append(entry)

        # Update MSE values for untrained functional groups
        if list_of_groups == []:
            print("All functional groups have been trained.")
        else:
            # calculate average MSE across functional groups
            dictionary_value = 0
            dictionary_iteration = 0
            for key in self.mse_output:
                # adds up total mse
                dictionary_value += self.mse_output[key]
                dictionary_iteration += 1

            # If list lengths match then there is no training data
            # for each functional group: assign 10% error rate
            # e.g 10**2 ==> 100
            if len(list_of_groups) == len(count_dict):
                for item in list_of_groups:
                    PseudoObserver.output[item] = 100

            # otherwise give missing functional groups average MSE
            # and add the value into the main output directory
            # which is called self.mse_output
            else:
                for item in list_of_groups:
                    if item not in self.mse_output:
                        self.mse_output[item] = dictionary_value / dictionary_iteration





        # PRINT FUNCTION
        ## Provide an output describing import attributes of the dataset
        #print("Data loaded successfully. ", "There are {0} functional groups in your data.".format(number_of_groups))
        #print("\nThe functional groups include:")

        #iteration = 0
        #t = PrettyTable(['Number','Functional group', 'Count'])
        #for group in counts:
        #    iteration += 1
        #    t.add_row([iteration, group, counts[group]])

        #print(t.get_string(sortby="Number"))
        #print('')

        #

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    test = VirtualEcologist("ve_testdata.csv", "ve_fulldataset.csv")
    print test.mse_output
    test.train_observer()
    print test.mse_output
    test.match_full_dataset()
    print test.mse_output
