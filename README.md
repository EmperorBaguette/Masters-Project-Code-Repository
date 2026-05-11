# Masters-Project-Code-Repository
A repository for the coding project developed during my Master's Degree Project. This contains the numerical model for the driven NS Junction system I have been working with.

# Notes
- Natural units have been used at all points in the calculations wherever possible.

# Required Packages
- numpy
- matplotlib

# Instructions For Usage
There are several parameters within the code that you should familiarise yourself with. 
- Delta_SC, a parameter which controls the size of the superconducting band gap, set this to 0 for a normal-normal junction.
- eta, a parameter which controls the strength of the static barrier and the driving strength.
- N, a parameter which determines the number of allowed Floquet sidebands in the system.
- omega, a parameter which describes the driving frequency of the system, adjusting this usually adjusts the position of the kinks.
- mu, a constant defining the chemical potential in the NS junction.
- v_param, a constant defining the relative strength of W_0 vs W_1, static barrier strength vs. driving strength.
- delta_T, a parameter describing the size of the temperature gradient for the induced thermal current calculation.
- E_start_value, a parameter describing the lower bound of energy values you wish to scan through for the simulation.
- E_stop_value, a parameter describing the upper bound of energy values you wish to scan through for the simulation.
- num_intervals, a parameter describing the number of intervals you would like there to be for the energy scan.

By editing these parameters you should be able to simulate and current system for either an NN or NS junction. 

# Code Functionality
- First edit the parameters described above for the system you wish to model.
- Press run.
- The code will run, printing a statement telling you what percentage of the calculation is complete as well as the value of the current normalisation at that step.
- The code will then display graphs at sizes which you can either resize manually or leave to be determined automatically by the size of the energy window.
- To open the next graph you must close the previous one.
- The code will create .npz files for data storage and comparison for the thermal gradient induced current in the code's directory.
