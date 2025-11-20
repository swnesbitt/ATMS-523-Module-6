# Module 6 Homework ATMS 523
## Scott Andersen

This assignment was implemented in hw6.py, assuming scikit learn and shap are installed in your python environment it should run as is.

Data taken for the tornado counts was given in the following link: https://www.spc.noaa.gov/wcm/data/1950-2021_actual_tornadoes.csv

Data taken for the ENSO, PDO, NAO, and AO oscillations were created in Module 4 notebook 1 and saved in oscillation.csv, and are included in this upload.

The final merged dataframe is saved in tornados.csv.

## Question 8 discussion

According to Scikit learn's permutation analysis, months April, May, and June are the most important for accuracy in predicting the tornado count. That is, shuffling these months data has the biggest impact on accuracy. While the NAO oscillation is the oscillation has the biggest impact on accuracy.

While SHAP analysis tells a slightly different story. The most important features are the months May and June, like in the permutation analysis, followed by the oscillations. First ENSO and then AO, followed by NAO and PDO. The high values for all of the months seem to have a significant impact on the tornado counts, while the values of the oscillation indexes have much more variance. Extreme values of ENSO seem to have substantial impact in increasing the number of tornados in Il, while higher values of AO do as well, not to the extent of ENSO.

Similar results are suggested by the bar chart data from SHAP.

Analysis of the ENSO feature shows us that higer values for the ENSO index lead to more tornados given the V shape of the graph. While ENSO neutral leads to less tornados. ENSO interacts most with the indicator of the month, June. That is a strong El niño or la niña will most impact the prediction of tornado counts in June.

Note that this is what the model has learned, and is not necessarily reflective of what the real impacts of the season and oscillation indexes have on tornado count. The signal seems noisy and the regression does not perform very well in either encoding.


## Citations
Tornado dataset is available from the storm prediction center data. Available https://www.spc.noaa.gov/wcm/data/1950-2021_actual_tornadoes.csv. Accessed November 15th, 2025.

Oscillation.csv was created in Module 4 Notebook 1, from various sources by Dr. Stephen Nesbitt.

