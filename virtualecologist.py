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
    Site name (header title: site)
    Functional group name (header title: lifeform) <= entries same as file1
    Transect identity (header title: transect)
    Plot identity (header title: plot)
    Percentage cover estimate (header title: cover) <= units should match file1

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
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class VirtualEcologist:
    """
    A class of methods to reduce the number of plots along transect lines.
    """
    def __init__(self, pilot_data, full_data):
        self.pilot_data = pilot_data # input file 1
        self.full_data = full_data # input file 2
        self.mse_output = {} # built with train_observer
        self.dataset = {} # built with match_full_dataset


    def print_table(self, data_dictionary):
        """
        Prints a data_dictionary in table form.
        """
        # number to iterate over lifeforms in the table
        iteration = 0
        # header
        t = PrettyTable(['ID', 'Life form', 'MSE', 'Pilot data'])
        # loop over data_dictionary
        for group in data_dictionary:
            # look in dictionary created by train_observer
            if group in self.fg_dict:
                pilot_data =  'yes'
            else:
                pilot_data = 'no'
            # increase ID counter
            iteration += 1
            # add each row to be printed
            t.add_row([iteration, group, round(data_dictionary[group], 3), pilot_data])
        # print to console
        print(t.get_string(sortby = "ID"))


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
        self.fg_dict = dict()

        with open(self.pilot_data, 'r') as f:
            file_reader = csv.reader(f)
            for row in file_reader:

                # assignment of columns in each row
                est_one = row[0]
                est_two = row[1]
                fg_key = row[2]

                # record the number of entries to the dictionary
                if fg_key not in self.fg_dict:
                    self.fg_dict[fg_key] = 1
                else:
                    self.fg_dict[fg_key] += 1

                # Calculate square difference between observers
                square_difference = (float(est_one) - float(est_two)) ** 2

                # add to global dictionary
                self.mse_output[fg_key] = self.mse_output.get(fg_key, 0) + square_difference

            # Calculate mean of error between observers
            # divide one dictionary by another
            for entry in self.mse_output:
                if entry in self.fg_dict:
                    self.mse_output[entry] = self.mse_output.get(entry, 0) / self.fg_dict[entry]

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
        for row in self.dataset['lifeform']:
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


    def calc_mmd(self, site, lifeform, trigger = 10,
        iterations = 100, min_plot = 4, figure = True):
        """
        Returns number of plots to reduce per transect. Takes two
        arguments; site name (e.g. shrubswamp) and lifeform (e.g. shrub).
        These arguments must be in the dataset that is being used.

        Three default values are added:
            1) a 10% trigger value
            2) 100 iterations
            3) minimum plot reduction of 4 per transect
            4) save figure to local directory

        These can be altered by the user.
        E.g.
        # increase trigger level
        test.calc_mmd('forestA', 'tree', trigger = 20)

        # turn off plotting
        test.calc_mmd('forestB', 'shrub', figure = False)
        """
        # variables
        self.ttest_results = [] # holds t-test results
        self.plot_data = [] # holds plot data
        self.trigger_points = [] # holders trigger value info for plotting
        counter = 0  # stores the number of iterations to bootstrap
        self.trigger = trigger # used for plotting
        self.site = site    # used for plotting
        self.lifeform = lifeform # used for plotting

        for i in range(iterations):
            counter += 1
            # add empty list to hold virtual_ecologist estimates
            ve_estimates = []
            # loop through the dataset
            for row in np.array(self.dataset):

                # match FG from dataset with a key in MSE dictionary
                if row[1] in self.mse_output:
                    # assign cover scores to variable "observer_estimate"
                    observer_estimate = row[4]

                    # The MSE is equal to the sum of the variance
                    # calculate SD and draw a random number that is
                    # centered on the real observers estimate.
                    # Assumes a normal distribution of error.
                    sd = math.sqrt(self.mse_output[row[1]])
                    virtual_ecologist = np.random.normal(observer_estimate, sd)

                    # maximum cover is 100%
                    if virtual_ecologist >= 100:
                        virtual_ecologist = 100
                    # negative values are missed observations
                    elif virtual_ecologist <= 0:
                        virtual_ecologist = 0

                    # add to array
                    ve_estimates.append(virtual_ecologist)

            # Add column to the main dataframe
            self.dataset['virtual_ecologist'] = ve_estimates

            # Start process of self.dataset
            temp_data_holder = (self.dataset[self.dataset['site'].str.contains(self.site)])
            find_longest_transect = len(temp_data_holder['plot'].unique())

            # subset data to evaluate each transect
            subset_data = dict(list(temp_data_holder.groupby(['site','transect'])))

            # Track plot reductions
            plot_iterator = 0

            for i in range(find_longest_transect):
                # add place holder list for plot names in sequential order
                plotnames_list = []

                # subset is data for all transects in the wetland
                for subset in subset_data:
                    # transect length is measured by number of plots
                    # plot iterator reduces transect length during each loop
                    transect_length = (len(subset_data[subset]['plot'].unique()) - plot_iterator)

                    # transect length cannot be less than minimum plot number
                    if transect_length <= min_plot:
                        reduce_transect_length = min_plot
                    else:
                        reduce_transect_length = transect_length

                    # sort plots in order
                    sorted_data = (subset_data[subset]['plot'])
                    sorted_data = sorted((sorted_data).unique())
                    sorted_data = (sorted_data)[:reduce_transect_length]

                    # add plot names to empty list
                    for plot_name in sorted_data:
                        if plot_name not in plotnames_list:
                            plotnames_list.append(plot_name)

                # increase plot_iterator reduces transect length
                plot_iterator += 1

                # subset data matching the list of plot names
                reduced_transect = (temp_data_holder[temp_data_holder['plot'].isin(plotnames_list)])

                # extract functional group from the data AFTER transects are reduced
                lifeform_data = (reduced_transect[reduced_transect['lifeform'].str.contains(self.lifeform)])
                lifeform_data = dict(list(lifeform_data.groupby(['site','transect','lifeform'])))

                # place holder for the next for loop
                group_data_array = []

                # calculate sums
                for group in lifeform_data:
                    # calculate observer estimate
                    real_observer = lifeform_data[group]['cover'].sum() \
                        / len(lifeform_data[group]['cover'])

                    # calculate virtual_ecologist estimate
                    virtual_observer = lifeform_data[group]['virtual_ecologist'].sum() \
                        / len(lifeform_data[group]['virtual_ecologist'])

                    # calculate plot_occupancy
                    plot_occupancy = len(lifeform_data[group]['plot'].unique())

                    # concatenate data
                    output = (group[0], group[1], group[2], real_observer, \
                    virtual_observer, plot_occupancy)

                    #append the output to the empty list
                    group_data_array.append(output)

                # convert the list of lists to a Pandas DataFrame
                result = pd.DataFrame(group_data_array, columns = list(["site", \
                "transect", "lifeform", "cover", "virtual", "occupancy"]))
                result.sort(['site', 'transect', 'lifeform'], ascending = True, inplace = True)

                # subset data to calc MMD
                mmd_subset = dict(list(result.groupby(['site','lifeform'])))

                # calculate t-test (2 tailed)
                for subset in mmd_subset:

                    # Calculate minimum detectable difference
                    A = mmd_subset[subset]['cover']
                    B = mmd_subset[subset]['virtual']
                    number_of_transects = len(mmd_subset[subset]['cover'])

                    # subtract two lists of equal length
                    calculated_difference = [a - b for a, b in zip(A, B)]

                    # Calculate Minimum Detectable Difference <=========== MDD
                    stand_dev = np.array(calculated_difference).std()
                    min_detect_change = np.sqrt((4) * (stand_dev**2) * \
                        (1.96 + 1.28) / number_of_transects)

                    # Determine average plot occupancy across all transects
                    avg_plot_occupancy = mmd_subset[subset]['occupancy'].sum()

                    # record values beyond trigger point
                    if min_detect_change >= int(self.trigger):
                        mdc_trigger_point = counter, plot_iterator, min_detect_change, \
                        avg_plot_occupancy, number_of_transects
                        self.trigger_points.append(mdc_trigger_point)

                    # append data for plotting
                    mdc_data = (plot_iterator, subset[0], subset[1], min_detect_change, \
                    number_of_transects, plot_occupancy)
                    self.plot_data.append(mdc_data)

                    # calculate paired t-test
                    test = stats.ttest_rel(mmd_subset[subset]['cover'], mmd_subset[subset]['virtual'])
                    data_str = (plot_iterator, subset[0], subset[1], list(test)[0], \
                    round(list(test)[1], 3), number_of_transects)
                    self.ttest_results.append(data_str)

        # Plot results
        if figure == True: self._create_mdd_figure()


    def _create_mdd_figure(self):
        """
        Saves a figure in png format for site and lifeform.
        """

        # import data
        mdc_dataframe = pd.DataFrame(self.plot_data, columns = list(["dropped_plots", \
        "site", "lifeform", "mdc", "n", "occupancy"]))

        # get trigger level data
        if self.trigger_points != []:
            # trigger points
            trigger_dataframe = pd.DataFrame(self.trigger_points, columns = list(["loop", \
            "dropped_plots", "mdc", "occupancy", "n"]))

            # mean trigger value
            # min used for list ordering
            mean_trigger_point = np.mean(list(trigger_dataframe.groupby('loop')['dropped_plots'].min()))

            # mean maximum plot occupancy
            # max used for list ordering
            mean_occupancy = np.mean(list(trigger_dataframe.groupby('loop')['occupancy'].max()))

            # log to shell
            print("Max number of plots you can drop (if each transect still has 4 plots) \
is: {0}".format(round(mean_trigger_point, 2)))

            print("The trigger value was exceeded when the minimum number of plots \
per transect was less than: {0}".format(round(mean_occupancy, 2)))

        else:
            # trigger not reached
            mean_trigger_point = 0

        # set up x axis data
        mdc_x = list(mdc_dataframe['dropped_plots'].unique())

        # set up data
        mdc_n_output = (mdc_dataframe["n"][0:max(mdc_x)])
        mdc_po_output = (mdc_dataframe["occupancy"][0:max(mdc_x)])

        #  mean number of dropped plots;
        mdc_mean_output = list(mdc_dataframe.groupby('dropped_plots')['mdc'].mean())

        # calc stand. error.
        mdc_sd_output = list(mdc_dataframe.groupby('dropped_plots')['mdc'].std())
        mdc_se_output = ((mdc_sd_output / np.sqrt(mdc_n_output)) * 1.96)  # <= 95% confidence interval

        # contruct Y limits which will change for each figure
        if  max(mdc_po_output) + 10 >= max(mdc_mean_output) + max(mdc_se_output):
            set_y_axis_limits = max(mdc_po_output) + 10
        else:
            set_y_axis_limits = max(mdc_mean_output) + max(mdc_se_output) + 10

        # plot error bars representing Minimum detectable difference
        plt.errorbar(mdc_x, mdc_mean_output, yerr = mdc_se_output, color='black', \
            lw = 1.5, linestyle = '-', label = "MDD - 95% CI")
            # 95% confidence interval

        # add a horizontal line representing the trigger value
        plt.plot([0, max(mdc_x)], [int(self.trigger),int(self.trigger)], \
            color = 'grey', lw = 2, linestyle = ':')

        # set x and y axis
        plt.ylim(0, set_y_axis_limits)
        plt.xlim(0, max(mdc_x) + 1)

        # n transects
        mdc_n_transects = mdc_dataframe["n"][0]

        # title: location [lifeform]
        plt.title(self.site + ' [' + self.lifeform + ' | ' + str(mdc_n_transects) + ' transects]')

        # add number of plots that are occupied
        plt.plot(mdc_x, mdc_po_output, label = "plot occupancy", \
            color = 'grey', lw = 1, linestyle = '--')

        # plot a vertical line representing optimal replication
        plt.plot([mean_trigger_point, mean_trigger_point], [0, set_y_axis_limits], \
            color = 'grey', lw = 1, linestyle = '-')

        # add text for optimal replication
        if mean_trigger_point != 0:
            plt.text(mean_trigger_point + 0.1, max(mdc_mean_output) + max(mdc_se_output), \
                round(mean_trigger_point, 2), size = 16)

        # uncomment to add x and y labels
        plt.ylabel("Minimum detectable difference (%)",size = 14)
        plt.xlabel("Number of plots dropped from each transect",size = 14)

        # save figure
        plt.savefig('MDD_' +  self.site + '_' + self.lifeform + '.png', format='png', dpi=1000)
        #plt.show()



if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    test = VirtualEcologist("ve_testdata.csv", "ve_fulldataset.csv")
    test.train_observer()
    test.match_full_dataset()
    test.print_table(test.mse_output)
    test.calc_mmd(site = "swamp", lifeform = "herb")
