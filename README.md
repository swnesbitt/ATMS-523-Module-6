# Module 6 Project
This project uses seasonal tornado prediction as a means to test the uses of simple machine learning methods such as: 
1) Random forest regression
2) One-hot encoding
3) SHAP

Features in the model include low-frequency climate modes (ENSO, PDO, NAO, and AO), as well as the month that the tornado takes place in via the one-hot encoding. 

Note: To be consistent with my previous work (Graber et al. 2024), the main project only uses EF1+ reports, but in the video submission, I will go through a simulation that does EF1+ reports and all reports including EF0. 

# References
Graber, M., R. J. Trapp, and Z. Wang, 2024: The Regionality and Seasonality of Tornado Trends in the United States. Npj Clim. Atmospheric Sci., 7, https://doi.org/10.1038/s41612-024-00698-y.

McGovern, A., R. Lagerquist, D. John Gagne II, G. E. Jergensen, K. L. Elmore, C. R. Homeyer, and T. Smith, 2019: Making the Black Box more Transparent: Understanding the Physical Implications of Machine Learning. Bull. Am. Meteorol. Soc., 100, 2175–2199, https://doi.org/10.1175/BAMS-D-18-0195.1.

Storm Prediction Center: Severe Weather Database Files (1950–2021), 1950-2021_actual_tornadoes.csv, NOAA/National Weather Service, National Centers for Environmental Prediction, Storm Prediction Center [data set], https://www.spc.noaa.gov/wcm/#data, last access: 19 November 2025. 
