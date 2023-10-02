

# Reproducng the figures 9-11 of the paper Optimization of Sensor Locations for a Light Pollution Monitoring Network.

**Notes:**

This code has been proved with:

* Python >= 3.8.10
* pip >= 20.0.2


To run the experiment I recommend to install the virtual env ```venv``` package to avoid breaking packages dependecies.  Here the instructions to install and use Virtual env. [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)

* You need the command line to execute the full experiment. The experiment is time consuming and I recommend to use the ```screen``` command [Screen]( https://linuxhint.com/how-do-i-close-a-screen-session/). This command allows to start a command line session in run the long experiment in the background even if the terminal is closed.  

The execution of the full experiment takes around 24 hours with a computer with a microprocessor.

## Instructions

Assuming you have installed ```Python```, ```pip```, ```virtualenv``` and ```github```:

0. Create a ```screen``` session. 

```$screen -r run_experiment```

If you want to exit press ```Ctrl+a, d```

You can close the terminal, come back later to the initialized session.


1. Create a python virtual environment for the reproduction of results. In this example we make a virtual environment with the name ```reproduce_ols_results```.

```$python3 -m venv reproduce_ols_results```

It will generate the folder with the files of the new python environment.


2. Go to folder: 

```$cd reproduce_ols_results/```


3. Activate the environment: 


```$source bin/activate```

and you will see in the command line an indicator of the current environment. It looks like:

```(reproduce_ols_results) ~/reproduce_ols_results$``` 

4. Download the experiment from github

```(reproduce_ols_results) ~$git clone ```

5. cd enter to the project folder.

```cd [project_folder]```

4. Install the requirements

```$ pip3 install requirements.txt```


## Reproduce the results.

To reproduce the results you must run the experiment in three steps:

1. ```python 01_precalculate_regions_of_influence.py```. It pre-calculates the spatial semivariograms and influence regions for each possible coordinate in the study region.

2. ```python 02_run_lightPollexperiments.py```. It calculates the optimal locations for diferent number of sensors and different sensitivty.

3. ```python 03_generate_figures.py``` It recovers the optimal locations and produce the figure 9-11.


4. If the the execution is succesful, you will find the figures 9-11 in folder:

```~$/sreproduce_optimization_sensor_locations_results/experiments/paper_experiment/results/```



> Written with [StackEdit](https://stackedit.io/).
> 
> I can explore some ideas regarding the stratification strategy. If I find something interesting I let you know.
> 
>Relative Frequency approach
>
>wheter or not or decided what is important to consider.