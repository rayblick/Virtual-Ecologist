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


    def calc_mmd(self, wetland,  ):
        """
        The bootstrap process is designed to analyse one location (multiple transects) and
        one functional group designated by the user input. The process runs 100 iterations
        of the same process to find the mean and sd for differences between the observer
        and the virtual ecologist (The PseudoObserver).
        """

        # plotting variables
        ResultsArray=[]
        MDCarray=[]
        TriggerValuesArray=[]

        list_wetlands = list(np.unique(PseudoObserver.data['Wetland']))
        print('List of selectable wetlands:')
        print(list_wetlands)

        # USER INPUT
        wetland_name = input("Enter a wetland name (Currently defaults to WC): ")
        if len(wetland_name) < 1: wetland_name = 'West Carne'
        print(" ")

        # base on user input find the unique functional groups that can be assessed
        possible_functional_groups = (PseudoObserver.data[PseudoObserver.data['Wetland'].str.contains(wetland_name)])
        print(possible_functional_groups['FuncGroup'].unique())

        # USER INPUT
        functional_group_name = input("Enter a functional group (Defaults to Tdr): ")
        if len(functional_group_name) < 1: functional_group_name = 'Tdr'

        print("Enter a trigger value for the wetland/functional group")
        Trigger_value= input("What you enter will be plotted as a horizontal line in the figure: ")

        pulled_in_the_data = PseudoObserver.data

        # record global iterations
        # I use this to mark the loops where MDC exceeds the specified trigger value
        global_iterator=0

        number_of_loops = 1000

        for i in range(number_of_loops):

            global_iterator = global_iterator + 1
            # add empty list to save the new pseudoobserver data
            myArray = []
            # Function to modify the PseudoObserver on each loop
            for row in np.array(pulled_in_the_data):

                if row[2] in PseudoObserver.output:
                    mean = row[5]

                    # The MSE is equal to the sum of the variance
                    # convert to SD and draw a random number using cover estimates of observer one
                    sd = math.sqrt(PseudoObserver.output[row[2]])
                    CalculatePseudoObserver = np.random.normal(mean, sd)

                    # A species can only cover a maximum area of 100%
                    if CalculatePseudoObserver >= 100:
                        CalculatePseudoObserver = 100

                    # if the pseudo observer draws negative value the species is considered missed
                    elif CalculatePseudoObserver <= 0:
                        CalculatePseudoObserver = 0

                    myArray.append(CalculatePseudoObserver)

            # Add the new column into the DataFrame
            pulled_in_the_data['pseudo_observer'] = myArray


            Data = (pulled_in_the_data[pulled_in_the_data['Wetland'].str.contains(wetland_name)])
            get_max_transect_length = len(Data['Plot_ID'].unique())

            # get data into the correct format where the data can be evaluated for each unique transect_ID
            subsets = dict(list(Data.groupby(['Wetland','Transect_ID'])))
            #print(subsets)

            # modify this line if you want to retain a minimum number of plots in a transect
            MinNumberOfPlots = 4


            # To keep track of plot reductions
            iterator = 0



            for i in range(get_max_transect_length):
                # This list will hold the plot names in sequential order
                # so that plot reductions occur deterministically from the end of each transect
                MyList = []


                # Note where subsets has come from above
                # It is a dictionary containing grouped data according to transect
                # Such that, subset refers to a grouping of data
                for subset in subsets:
                    # item is an integer and used as a slice operator 4lines below in NewData
                    item = (len(subsets[subset]['Plot_ID'].unique()) - iterator)

                    # conditional statment that checks the minimum plot number
                    # that is allowed to occur on each transect (remember this is still within
                    #the dictionary as a 'subset' of data)
                    if item <= MinNumberOfPlots:
                        SliceOperator = MinNumberOfPlots
                    else:
                        SliceOperator = item

                    # arrange plots in order necessary because dictionaries are not
                    # ordered data structures
                    NewData = (subsets[subset]['Plot_ID'])
                    NewData = sorted((NewData).unique())
                    NewData = (NewData)[:SliceOperator]

                    # the following for loop adds the plot names to the empty list
                    # created at the beginning of this process (called: MyList)
                    for plot_name in NewData:
                        if plot_name not in MyList:
                            MyList.append(plot_name)

                #print(MyList)
                #The iterator increments on every loop which decreases the length of the slice operator
                iterator += 1

                #print(MaximumTransectLength)
                # From the full dataset pull out the data that matches the list (according to the slice operator)
                CompiledData = (Data[Data['Plot_ID'].isin(MyList)])

                #-------------------------------------------------------------------------
                #For each subsect of 'CompiledData' in the previous step, extract/compile
                # to run a t-test

                # pull out the data according to user entry
                pieces = (CompiledData[CompiledData['FuncGroup'].str.contains(functional_group_name)])
                pieces = dict(list(pieces.groupby(['Wetland','Transect_ID','FuncGroup'])))

                #print(pieces)

                MyDataArray = []
                #loop through each piece of data and calculate the sum
                for group in pieces:
                    #print(pieces[group])
                    # This returns total proportional cover for each functional group along each transect for both observer and pseudo
                    cover = pieces[group]['Cover'].sum() / len(pieces[group]['Cover'])
                    pseudo = pieces[group]['pseudo_observer'].sum() / len(pieces[group]['pseudo_observer'])
                    plot_occupancy = len(pieces[group]['Plot_ID'].unique())

                    # combine data and append the results to MyDataArray
                    # To recap: the following line concatenates the wetland name, transect id, functional group
                    # along with the sum of cover and pseudo observer.
                    # This will occur as many times as the longest transect EXCEPT on each iteration
                    # it is likely that there will be fewer transects containing species from your chosen functional group
                    output = (group[0], group[1], group[2], cover, pseudo, plot_occupancy)

                    #append the output to the empty list
                    MyDataArray.append(output)

                # convert the list of lists to a Pandas DataFrame
                result = pd.DataFrame(MyDataArray, columns = list(["Wetland","Transect","FuncGroup","Cover","Pseudo","PlotOccupancy"]))
                result.sort(['Wetland','Transect','FuncGroup'], ascending = True, inplace=True)

                # subset the data again to get the required columns for calculating a paired t-test for each functional group in each wetland
                newsubset = dict(list(result.groupby(['Wetland','FuncGroup'])))
                #print(newsubset)


                # iterate through the new data dictionary called 'newsubset' and calculate dependent t-test (2 tailed)
                for subset in newsubset:

                    # Calculate minimum detectable difference
                    A = newsubset[subset]['Cover']
                    B = newsubset[subset]['Pseudo']
                    # note that n in the following line refers to transects NOT transect length
                    # which was calculate in the previous loop.
                    n= len(newsubset[subset]['Cover'])
                    # use list comprehension to subtract two lists of equal length
                    calculated_difference = [a - b for a, b in zip(A, B)]
                    stand_dev = np.array(calculated_difference).std()
                    MDC = np.sqrt((4) * (stand_dev**2) * (1.96+1.28) / n)

                    # Determine average plot occupancy across all transects
                    PO = newsubset[subset]['PlotOccupancy'].sum()

                    # save the information when MDC exceeds the set trigger value
                    # note this will record ALL iterations not just the first event when the trigger is exceeded
                    # uncomment the next two lines
                    if MDC >= int(Trigger_value):
                        MDC_trigger_breaks = global_iterator,iterator,MDC,PO,n
                        TriggerValuesArray.append(MDC_trigger_breaks)


                    # append data for plotting
                    MDC_data = (iterator, subset[0], subset[1], MDC, n, PO)
                    #print(MDC_data)
                    MDCarray.append(MDC_data)

                    # calculate paired t-test
                    test = stats.ttest_rel(newsubset[subset]['Cover'],newsubset[subset]['Pseudo'])
                    data_str = (iterator, subset[0],subset[1],list(test)[0],round(list(test)[1],3),n)
                    ResultsArray.append(data_str)


        # set plotting params
        font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

        matplotlib.rc('font', **font)
        matplotlib.rc(('xtick', 'ytick'), labelsize=14)

        # evaluate p-values
        PlotDataFrame = pd.DataFrame(ResultsArray, columns = list(["Dropped Plots", "Wetland","Functional Group", "t value", "p value","n"]))

        # extract the x axis data from PlotDataFrame
        x = list(PlotDataFrame['Dropped Plots'].unique())

        # extract n from PlotDataFrame; note it requires x in previous line
        # and is used to calculate the standard error
        n_output = (PlotDataFrame["n"][0:max(x)])

        # extract mean from PlotDataFrame;
        mean_output=list(PlotDataFrame.groupby('Dropped Plots')['p value'].mean())

        # extract sd from PlotDataFrame; not the division at the end of line - calc stand. error.
        sd_output=list(PlotDataFrame.groupby('Dropped Plots')['p value'].std())
        se_output = (sd_output / np.sqrt(n_output))

        # plot the test of p values
        plt.errorbar(x, mean_output, yerr=se_output)
        plt.ylim(0,1.1)
        plt.xlim(0,max(x) + 1)
        plt.title(wetland_name + ' [' + functional_group_name + ']')
        plt.ylabel("p value")
        plt.xlabel("Number of plots that are dropped from each transect")
        plt.savefig('MC_' +  wetland_name + '_' + functional_group_name + '.png', format='png', dpi=1000)
        plt.show()
        #-------------------------------------------------------------------------------------------------------------

        # evaluate MDC-values
        MDCDataFrame = pd.DataFrame(MDCarray, columns = list(["Dropped Plots", "Wetland","Functional Group", "MDC","n","Plot Occupancy"]))
        #print(MDCDataFrame)

        if TriggerValuesArray != []:
            # Also get the trigger values and find the average number of dropped plots where the MDC exceeds the specified trigger
            TriggerDataFrame = pd.DataFrame(TriggerValuesArray, columns = list(["Loop", "Dropped Plots","MDC","Plot Occupancy","n"]))

            MeanTriggerPoint = np.mean(list(TriggerDataFrame.groupby('Loop')['Dropped Plots'].min()))
            MeanOccupancy = np.mean(list(TriggerDataFrame.groupby('Loop')['Plot Occupancy'].max()))
            print("The max number of plots you can drop (if the transect will still have 4 plots) is: {0}".format(round(MeanTriggerPoint,2)))
            print("The minimum number of plots that must be occupied with this functional group is: {0}".format(round(MeanOccupancy,2)))

        else:
            MeanTriggerPoint = 0

        # extract the x axis data from PlotDataFrame
        mdc_x = list(MDCDataFrame['Dropped Plots'].unique())

        # extract n from PlotDataFrame; note it requires x in previous line
        # and is used to calculate the standard error
        mdc_n_output = (MDCDataFrame["n"][0:max(mdc_x)])
        mdc_po_output = (MDCDataFrame["Plot Occupancy"][0:max(mdc_x)])

        # extract mean from PlotDataFrame;
        mdc_mean_output=list(MDCDataFrame.groupby('Dropped Plots')['MDC'].mean())

        # extract sd from PlotDataFrame; not the division at the end of line - calc stand. error.
        mdc_sd_output=list(MDCDataFrame.groupby('Dropped Plots')['MDC'].std())
        mdc_se_output = ((mdc_sd_output / np.sqrt(mdc_n_output)) * 1.96)  # <= 95% confidence interval

        # contruct Y limits which will change for each figure
        if  max(mdc_po_output)+10 >= max(mdc_mean_output) + max(mdc_se_output):
            set_y_axis_limits = max(mdc_po_output)+10
        else:
            set_y_axis_limits = max(mdc_mean_output) +max(mdc_se_output) + 10

        # plot error bars representing Minimum detectable difference
        plt.errorbar(mdc_x, mdc_mean_output, yerr=mdc_se_output,color='black',lw=1.5,linestyle='-',label="MDD [95% C.I.]")

        # add a horizontal line representing the trigger value
        plt.plot([0, max(mdc_x)], [int(Trigger_value),int(Trigger_value)],color='grey',lw=2,linestyle=':')

        # set x and y axis
        plt.ylim(0,set_y_axis_limits)
        plt.xlim(0,max(mdc_x)+1)

        # add a title including the location and the functional group
        plt.title(wetland_name + ' [' + functional_group_name + ']')

        #add to the figure the number of plots that are occupied by the chosen functional group
        plt.plot(mdc_x, mdc_po_output, label="Plot Occupancy",color='grey',lw=1,linestyle='--')

        # plot the vertical line representing optimal replication
        plt.plot([MeanTriggerPoint,MeanTriggerPoint],[0,set_y_axis_limits],color='grey',lw=1,linestyle='-')


        plt.text(MeanTriggerPoint + 0.1, max(mdc_mean_output) + max(mdc_se_output), round(MeanTriggerPoint,2),size=16)

        # uncomment to add x and y labels
        #plt.ylabel("Minimum detectable difference (%)",size = 14)
        #plt.xlabel("Number of plots dropped from each transect",size = 14)

        plt.savefig('MDD_' +  wetland_name + '_' + functional_group_name + '.png', format='png', dpi=1000)
        plt.show()








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
