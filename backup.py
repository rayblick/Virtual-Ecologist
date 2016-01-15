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
from prettytable import PrettyTable
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import re
from itertools import cycle, islice

#class PseudoObserver:


# set variables

data = 0

def train_observer(filename):
    """
    Returns a dictionary containing Mean Square Error of estimates.

    Input is a csv with 3 columns: observer 1 estimates, observer 2
    estimates and functional group names.

    >>> train_observer("data.csv") # one string input
    dict
    """
    # store count of functional groups
    FG_dict = dict()
    # store Mean Square Error output
    MSE_output = {}

    with open(filename, newline = '') as f:
        file_reader = csv.reader(f)
        for row in file_reader:
            # assignment of columns in each row
            EST_one, EST_two, FG_entry = row[0], row[1], row[2]

            # record the number of entries to the dictionary
            if FG_entry not in FG_dict:
                FG_dict[FG_entry] = 1
            else:
                FG_dict[FG_entry] += 1

            # Calculate square difference between observers
            value =(float(EST_one) - float(EST_two))**2

            # add to global dictionary
            MSE_output[FG_entry] = MSE_output.get(FG_entry, 0) + value

        # Calculate mean of error between observers
        # divide one dictionary by another
        for entry in MSE_output:
            if entry in counts:
                MSE_output[entry] = MSE_output.get(entry, 0) / FG_dict[entry]

        return MSE_output

        # Print out the dictionary in table form
        #iteration = 0
        #NumberOfGroups = len(MSE_output)
        #t = PrettyTable(['Number','Functional group', 'MSE','n'])

        #print("You are training {0} functional groups:".format(NumberOfGroups))
        #for group in PseudoObserver.output:
            #iteration += 1
            #if group in counts:
                #n = counts[group]

                #t.add_row([iteration, group, round(PseudoObserver.output[group],3), n])

        #print(t.get_string(sortby="Number"))


#def UseDataset():
#    """
#    Imports a user defined table into Python environment. The table must have a column called FuncGroup. \n
#    If a training dataset is not used, then a default dataset is used. If no pilot data are available,\n
#    then each functional group is given a mean square error (MSE) of 100. This corresponds to a pseudo-observer\n
#    that can estimate cover to within 10% of the real assessor. It is the researchers responsibility to \n
#    determine if this is appropriate. If at least one functional group has been trained, then the functional\n
#    groups are assigned the average MSE from all training datasets.

#    """
#    counts = dict()
#    ListOfGroups = []

#    dname = input("Enter file name of your full dataset: ")
#    if len(dname) < 1: dname = "NP2014_vegdata_mod.csv"

    PseudoObserver.data = pd.read_csv(dname)
    for row in PseudoObserver.data['FuncGroup']: # NEED A METHOD TO IDENTIFY OTHER COLUMN NAMES
        if row not in counts:
            counts[row] = 1
        else:
            counts[row] += 1

    NumberOfGroups = len(counts)
    for entry in counts:
        if entry not in PseudoObserver.output:
            ListOfGroups.append(entry)

    # Provide an output describing import attributes of the dataset
    print("Data loaded successfully. ", "There are {0} functional groups in your data.".format(NumberOfGroups))
    print("\nThe functional groups include:")

    iteration = 0
    t = PrettyTable(['Number','Functional group', 'Count'])
    for group in counts:
        iteration += 1
        t.add_row([iteration, group, counts[group]])

    print(t.get_string(sortby="Number"))
    print('')


    # State whether any additional attributes need to be added to the global variable holding the training dataset
    if ListOfGroups == []:
        print("All functional groups have been trained.")
    else:
        print("Some groups have not been trained with pilot data. A default value will apply to these groups.")
        print("This includes {0}".format(ListOfGroups))

        # FOR ALL GROUPS WITHOUT TRAINING DATA - ADD THE MEAN OF ALL MSE VALUES
        # get the average of the trained dictionary (that is, the mean MSE value for all functional groups)
        DictionaryValue = 0
        DictionaryIteration = 0
        for value in PseudoObserver.output:
            DictionaryValue += PseudoObserver.output[value]
            DictionaryIteration += 1

        # if no training data is supplied assign every functional group a value of 100
        # 100 represents a PsuedoObserver that can estimate cover to within +/- 10% of the real assessor
        if len(ListOfGroups) == len(counts) :
            for item in ListOfGroups:
                PseudoObserver.output[item] = 100

        # However, if there is at least one functional group trained, then find the average
        # This means that the MSE is drawn from real data
        #
        else:
            for item in ListOfGroups:
                if item not in PseudoObserver.output:
                    PseudoObserver.output[item] = DictionaryValue / DictionaryIteration



def ReturnFuncGroupPlot():

    wetlands=[]
    print("Do you want to drop any functional groups?")
    dropped_groups=input("Enter a comma separated list with no spaces: ")
    dropped_groups = dropped_groups.split(',')

    FG_subset = dict(list(PseudoObserver.data.groupby(['Wetland'])))
    for wetland in FG_subset:
        #if wetland not in wetlands:
            #wetlands.append(wetland)

        wetland_dictionary={}


        #print(FG_subset[wetland])
        for row in FG_subset[wetland]['FuncGroup']:
            if row not in wetland_dictionary:
                wetland_dictionary[row] = 1
            else:
                wetland_dictionary[row] += 1

            #if key not in wetland_dictionary:
                #wetland_dictionary.get(key, 0)

        for i, j in wetland_dictionary.items():
            put_data_together=wetland, i, j
            wetlands.append(put_data_together)

    #print(wetlands)
    data =(pd.DataFrame(wetlands,columns = list(["Wetland","FuncGroup","Count"])))

    # Drop the functional groups assigned by the user
    print("Reminder: you dropped the following functional groups " + str(dropped_groups))
    data = data[~data.FuncGroup.isin(dropped_groups)]

    # modify the data structure for plotting and replace NaNs with zero
    df_data = data.groupby(['Wetland','FuncGroup']).aggregate(sum).unstack()
    df_data.fillna(0, inplace=True)

    # convert entries into percentiles
    percentiles = df_data.apply(lambda c: c / c.sum() * 100, axis=1)
    #-----------------------------------------------------------------

    # specify a colour gradient
    #my_colors = [(0.9, 0.1, x/35) for x in range(0,len(df_data)*2,2)]
    my_colors=list(islice(cycle(['DarkKhaki','Khaki','PaleGoldenrod','LightGoldenrodYellow','LightYellow','white']), None, len(percentiles)))


    # plot barchart
    ax = percentiles.plot(kind='bar', stacked = True, color=my_colors, ylim=(0,100))

    # this section modifies the labels for the barchart
    # There must be an easier way than using REGEXP
    h,l = ax.get_legend_handles_labels()

    labels =[]
    for i in l:
        i= re.sub(r'\W','',i.split(',')[1])
        labels.append(i)

    ax.legend(h,labels,loc='center left',bbox_to_anchor=(1,0.5),prop={'size':8})
    ax.set_position([0.1,0.6,0.6,0.35])
    ax.grid('off')
    fig = ax.get_figure()


    # save the figure
    fig.savefig('FunctionalGroups.png', format='png', dpi=1000)




def ReturnProbDensityFunction():
    """
    Prints a table containing the group name and sigma that will define the PseudoObserver model
    """
            # Print out the dictionary in table form
    iteration = 0
    NumberOfGroups = len(PseudoObserver.output)
    t = PrettyTable(['Number','Functional group', 'MSE'])
    print("\nThere are {0} functional groups that are trained for analysis:".format(NumberOfGroups))
    for group in PseudoObserver.output:
        iteration += 1
        t.add_row([iteration, group, round(PseudoObserver.output[group],3)])

    print(t.get_string(sortby="Number"))

    input("Hit enter to plot the prediction rate of your pseudo-observer...")
    # Plot estimators as histograms

    for group in PseudoObserver.output:
        x = np.random.normal(50, math.sqrt(PseudoObserver.output[group]),1000)
        count, bins, ignored = plt.hist(x, 30, range=[0,100],normed=True,color="White")
        #plt.ylabel('Probability Distribution')
        #plt.xlabel('Percentage cover estimate')
        plt.plot(bins, 1/(math.sqrt(PseudoObserver.output[group]) * np.sqrt(2 * np.pi)) *np.exp( - (bins - 50)**2 / (2 * math.sqrt(PseudoObserver.output[group])**2) ),linewidth=2, color='Black')

        plt.text(5,0.09, group + ' [' + str(round(np.sqrt(PseudoObserver.output[group]),2)) + ']', size=16)
        plt.ylim((0,0.1))
        plt.xlim((0,100))
        plt.ylabel("Probability of cover estimate", size=14)
        plt.xlabel("Percentage cover estimate for a single species (0-100)", size=14)

        plt.plot([50, 50], [0, 0.1], 'Grey', lw=2, linestyle = '--')

        name = group
        name = re.sub('/','_',name)

        #print(name)
        plt.savefig(name + '.png', format='png', dpi=1000)
        plt.show()

def bootstrap():
    """
    The bootstrap process is designed to analyse one location (multiple transects) and
    one functional group designated by the user input. The process runs 100 iterations
    of the same process to find the mean and sd for differences between the observer
    and the virtual ecologist (The PseudoObserver).
    """

    # set up empty arrays to save results - used for plotting
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


# Example usage of the above code
# Uncomment as needed
#p = PseudoObserver
#p.TrainObserver()
#p.UseDataset()
#p.ReturnFuncGroupPlot()
#p.ReturnProbDensityFunction()
#p.bootstrap()
