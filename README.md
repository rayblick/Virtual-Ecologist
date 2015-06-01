# Virtual-Ecologist

## Introduction
Simulating data is a common method in ecology to evaluate pattern and process. A classic example was the approach by Connor and Simberloff (1979), who re-evaluate Jared Diamond's (1975) assembly rules of bird species on the Bismarck Archipelago. By employing a simple null model, Connor and Simberloff (1979) compared the original co-occurrence pattern of birds on islands, with a virtual dataset that would arise if bird species randomly occupied different islands. There are many examples of similar tests in ecology; however, the code I have written is geared towards the virtual ecologist approach for the purpose of optimising vegetation surveys.  

The virtual ecologist is an "approach where simulated data and observer models are used to mimic real species and how they  are virtually observed." (Zurell et al. 2009). Under this scenario, a researcher can model the entire data collection process and evaluate the effects of different sampling designs and sampling effort in different environments. The code I present here employs the virtual ecologist approach to generate a second observer (I also refer to this as the pseudo-observer) for an entire flora monitoring programme that would allow me to calculate the minimum detectable difference (MDD) that we expect to see as we reduced transect length. The overall aim is to reduce transect length to a minimum size that will allow us to detect an a priori trigger value that we determined necessary in the monitoring programme, and ulitmately, to optimise our time spent in the field.      

I have added the following section on 'data collection and data structure' because the overall results will likely differ if your study system is more appropriately sampled using a different technique. Even though I have only considered the transect sampling design in this study, it would be possible with some modification to the existing code to apply this approach to other sampling techniques, such as quadrats. 

## Data collection and data structure
The data used to construct this code has come from Newnes Plateau Shrub Swamp (NPSS) communities, situated in the Blue Mountains, Australia. NPSS are semi-isolated wetland plant communities that house a different assemblage of plants (often less than 2m tall) to the surrounding environment (30+ m eucalypt woodland), and they generally have a hydrological gradient from dry edge to a wet central drainage line. NPSS are protected by state and federal legislation because they are subject to several key threatening processes, including underground coal mining. Recent research has shown that the best practice sampling design are transects (see next paragraph), which cross the hydrological gradient. This coincides with  preliminary analyses showing that transects are required to capture the phylogenetic turnover in these wetland plant communities which occurs rapidly over 10s of meters, rather than a slow distance decay over 10s of kms. 

Thus, the overall data structure covers two important points; 1) we used scale-dependent (variable length) transects to capture the rapid turnover of vegetation across the hydorological gradient, and 2) we analysed each semi-isolated wetland independently due to having very different hydrological regimes covering different biogeographic areas (ranging in size from 0.4 to 4 hectares), at different altitudes (ranging from 950m to 1150m a.s.l). To the user of this programme, this will mean the inclusion of columns in the dataset for weltand_id and transect_id (more on this below).     

### Transect design
The NPSS are generally long (1-2km) and narrow (10-100m) with a longitudinal slope less than 10 degrees. Depending on the start position on one side of the wetland, the total original transect length will be determined by the width of the wetland crossing the hydrologically gradient at that point. It is impractical to continuously sample across the width of these wetlands in a feasible timeframe, even if you could travel in a straight line through the dense vegetation. Cutting a long story short, the best practice method was determined to be percentage cover estimates to the nearest 1% for all plant species in a 1m^2 plot that was separated by 4m along each transect (note that each plot took ~9 mins to collect). Therefore, a 40m transect would have ten 1m2 plots which would take approximately an hour and half to complete. Sampling was scale-dependent; each semi-isolated wetland contained a minimum of three transects and the largest wetland contained 10 transects. 

Importantly, all plots along each transect is pooled together to generate a proportional total of cover for each species. The replication is determined by the number of transects, not the number of plots. The number of plots simply improves the chance of detecting new species, and decreases the standard error across the transect which is associated with estimating cover (within observer bias).   

The key thing to note here is that the transects have permanent markers for the start and end points of each transect. The question is: when we return in the next season, do we need to go beyond the central drainage line to the other side, or, can we reduce the transects without sacrifficing the ability to identify any real differences that exist between seasons?

### Wetland indicator groups
The transect design has one final element requiring attention. The monitoring programme assesses the relative change in cover of each indicator group. That is, we are interested in determining if there is a significant loss in damp/wet tolerators and an increase in terestrial species, among others. The code I have written here addresses this specifically. For users of this programme it means that the real dataset requires a pre-determined list of groups that can be assigned to each and every species (i.e. a column in the dataset called FuncGroup). While I have used wetland indicator group here, you could just as easily assign life form or functional guild determined by published literature. I use the terms 'indicator group', 'functional group', 'functional guild' or 'lifeform' interchangebly because it makes no difference to this program. However, I expect that your results will differ drastically depending on your categorisation method. 

## The Virtual Ecologist
The aim of the virtual ecologist is to act as a second dataset, modelled from pilot data, across an entire monitoring programme to determine how many plots we can remove from each transect, without affecting the error variance assoicated with pooled cover abundance totals. The virtual dataset could then be used to determine the MDD expected between observers. 
That is, the virtual ecologist needs to record an estimate for every plant species that was observed and recorded by the real assessor, and in some circumstances, miss the observation of low cover species.

To acheive this goal, the code I have written has several stages...
* Import pilot data containing paired estimates from two real assessors
* Import the real dataset for an entire monitoring programme (collected by 1 person per transect)
* Summarise data
* Iteratively remove plots and calculate MDD

Data processing has been broken into methods within the class PseudoObserver. The code was written in Python 3.4 using Spyder2 and IPython. To run the code, add the python script to the same directory as the data. You may need to install several modules.

### Pilot data
The pilot data is used to determine the error expected between two real people. The file contains three columns, including 1) the cover estimate of assessor one, 2) the cover estimate of assessor two, and 3) the functional group the species belongs too. The file must have no headings. On instantiating a variable (e.g. 'p' as in my commented out section at the bottom of the code) with PseudoObserver(), run the method 'p.TrainObserver()'. You will be prompted to enter the name of your datafile (note that in the code there is a shortcut you can modify to speed up testing). Ensure that you end the file name with the extension .csv. The data will be stored for further use in other methods. The output is a table detailing the functional group, Mean Squared Error (MSE), and the number of samples that were assessed for each group.  

### Import the real dataset
The real data includes observations from one person per transect. The file contains six columns, including 1) Wetland, 2) Species, 3) FuncGroup, 4) Transect_ID, 5) Plot_ID, and 6) Cover. This file requires these headings. Run the method 'p.UseDataset'. Similar to the pilot data, you will be prompted for the file name (note that in the code there is a shortcut you can modify to speed up testing). Ensure that you end the file name with the extension .csv. This code will determine if the functional groups in the real dataset were trained from the previous step. If the dataset includes a functional group that wasn't trained then it will assign the average error from all groups that were trained in the previous step. However, if no training data were given at all, then this method will assume that two assessors will estimate cover at a rate of +/- 10%. The output is a table showing the number of observations of each functional group in your dataset. 

### Summarise data
#### Bar chart
The first method for summarising data is a barplot showing the percentage contribution of each functional group in each wetland. Run 'p.ReturnFuncGroupPlot()'. You will be prompted to enter a comma separated list of functional groups that you want to drop from the bar plot. If you want to include all groups then simply hit the enter key. The bar chart will be shown in IPython and saved to the processing directory (the location of your data and the python script). 

#### Error estimates: Probability density function (PDF)
The second method for summarising data is to generate a PDF showing the probability of the estimates for the virtual ecologist. Run 'p.ReturnProbDensityFunction()'. The figures produced are histograms indicating the virtual data surrounding a real estimate of 50%. The number of figures is determined by the number of functional groups in your dataset. The name of the functional group and the standard deviation in the top left corner. The location of this text is hard-wired on the x and y axis, and you may need to change it following the input of your data. 

### Iteratively remove plots and calculate MDD
The final process is the iterative reduction of plots. Run 'p.bootsrap'. You will be prompted for several user inputs. First, input the chosen community (I refer to them as wetlands), second enter a functional group from the list given, and third, you will be asked for a trigger value (e.g. 10%). The program will determine the max number of plots you can drop before observer error is too large between two assessors (e.g. estimation error exceeds 10%). If the method fails after this point then I suggest checking the replication for that group first before looking for alternative explanations. 

The default number of iterations is 1000 (You can change this manually in the code). For every iteration, the plots are sequentially pulled from each transect, starting at the end-post working back to the start. If the transect has 4 plots then no further reductions are made to that particular transect. This means that replication is consistent across all tests for each wetland and the only factor changing is the pooled variance for the abundance of plants in each functional group.   

The results of this test are two figures (p value and MDD) representing the difference between the virtual ecologist and the real assessor. Error bars are standard deviations. The dotted line represents the user defined trigger value.





