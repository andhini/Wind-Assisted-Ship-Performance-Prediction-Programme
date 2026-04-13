# Wind-Assisted-Ship-Performance-Prediction-Programme
This is a Performance Prediction Programme for wind-assisted ship in collaboration with the University of Southampton. This project is funded by the UK Research and Innovation (UKRI)-Innovate UK. This PPP is generated in accordance with the manuscript submitted for Ocean Engineering (DOI will be provided later). Please consult the paper for further info. <br />

Please refer to the manual of the program in a pdf file (will be provided later). Currently, master branch is the most updated version. The main code is VPP_WAShip.py, supported by the following ship files: <br />
Aframax.py,  <br />
HM_ship.py,  <br />

Users access the program by modifying (or creating new) the following example files: <br />
example_multisolve_Aframax.py, <br />
example_multisolve_Aframax_parallel.py, <br />
example_singlesolve_Aframax.py <br />

The postprocessing code for Monte Carlo, SHAP and Sobol:  <br />
example_singlesolve_Aframax_MonteCarlo_SHAP_Sensitivity.py, <br />

Regression-trained models for the postprocessing script:
vpp_surrogate_9kts_ballast.pkl , <br />
vpp_surrogate_9kts_fullyload.pkl , <br />
vpp_surrogate_13kts_ballast.pkl , <br />
vpp_surrogate_13kts_fullyload.pkl , <br />

Other additional files : <br />
Example_power_fuel.py is an example of fuel-power relations<br />
wind_probability.py is the IMO annual wind probability for global route shipping<br />

There are two collections of aerodynamic data for different wing-sails located in the directories: <br />
Fujiwara_aero is the main aero sample<br />
G_aero is another example to feed the Programme <br />

The calling arrangement of the program is described generally by the following flowchart. <br />
![Alt text](https://github.com/andhini/Wind-Assisted-Ship-Performance-Prediction-Programme/blob/master/FC.png?raw=true "Flowchart")

Inquiries and request can be addressed to my email in Reserach Repository. Enjoy the WASp PPP!