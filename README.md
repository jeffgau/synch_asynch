# synch_asynch
 
This repository supports running the MATLAB simulations for the synch-asynch paper. It also provides code for plotting the simulation outputs.

Workflow:
1. Specify simulation type ('robobee', 'moth', or 'roboflapper') in ```run_analysis.m```
2. Specify simulation parameters in ```load_simulation_param.m```. The most common change is ```ntests```, which specifies the parameter grid size (```ntests x ntests```)
3. Modify system parameters. This is typically unnecessary unless the flapper or bee was modified.
4. Run ```run_analysis.m```. This script loads system and simulation parameters. Then goes through a force tuning procedure (see notes below). After force tuning, the heavy lifting is done in ```run_simulation.m```. After data is generated, visualization code is ran. Finally, data is saved for final plotting in Python.
6. Plot simulation outputs with ```plotter.py```.



MATLAB setup:
* Code is run on MATLAB 2020a with Simulink Desktop Realtime

PYTHON setup:
1. Download Anaconda: https://www.anaconda.com/
2. Navigate Command Prompt (Windows) or Terminal (Mac) to this repository.
3. Create the Conda environment by running ```conda env create```.
4. Activate the environment with ```conda activate synch_asynch```.
5. Open your preferred IDE to run ```plotter.py`.


Notes:
* When ```run_analysis.m``` is ran, it will begin with an iterative force tuning procedure. It essentially adjusts the ratio of the synchronous to asynchronous force until a user-specified metric is identical for the peak synchronous and peak asynchronous conditions. This parameter is currently named ```opt_param``` in ```run_one_simulation.m```. For example, if ```opt_param``` is set to peak oscillation amplitude, the iterative tuning procedure will adjust the synch and asynch forces until synch and asynch oscillation amplitude are sufficiently close.
