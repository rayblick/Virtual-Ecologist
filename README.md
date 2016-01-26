===========
Virtual Ecologist
===========

A set of tools to optimize transect length in an ecological monitoring program.
The primary goal is to reduce transect length without losing the power to
detect a difference between surveys (e.g. seasonal surveys).  

These tools assume that the data are collected from transects (which can be
 variable in length), and each transect contains sequential data collection points.
E.g. A 50 meter transect with 1m2 plots spaced 4 meters apart.   

It also assumes that you are assessing grouped data (e.g. functional groups,
lifeforms, guilds or a custom list of groupings used to cluster species) at an
individual location. For example, lifeforms (forbs, trees) in a remnant forest.


Installation
=========
```bash
pip install virtualecologist
```

Usage
=========

```python
# Import library
from virtualecologist import virtualecologist as ve
# instantiate virtualecologist
# only requires the main dataset. You must specify pilot data if it is given
example = VirtualEcologist(pilot_data="virtualecologist/data/pilotdata.csv",
                                    "virtualecologist/data/fulldata.csv")
# uses pilot data to generate mean error rate
example.train_observer()
# match main data to the mean error rate
# if pilot data not given, you assume a 10% error rate for all groups
example.match_full_dataset()
# print to console a table of groups and error rates
# execution here returns error rate
example.print_table(example.mse_output)
# Main function
# isolate the location and the lifeform to investigate
# produces a figure showing number of plots to reduce
example.calc_mmd(site="swamp", lifeform="shrub")
# generates a stacked barchart for all sites and lifeforms in the dataset
example.create_barchart()
# get the probability density function for your selected lifeform
example.create_pdf_figure()
```
