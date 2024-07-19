

# Train models

To train all the three models you can simply run ```executable.sh``` by

```bash
/executable.sh 
```

It calls all the bash files needed to be run for the sub-goal generator, action generator conditioned on the sub-goals,
first-subgoal generator, and the swift models. 

Just to explain the details:

In ```executable```, the bash files are named as ```run_*.sh``` which are asking for slurm job allocations. All the variables for job's resources including amount of memory, number of cpus, gpus, ... set in ```run_*.sh``` files.

In each ```run_*.sh``` the main bash file named ```ds_train*.sh``` is called which uses deepspeed. 

The path of the data to fine tune the models and all the parameters of the models like epoch, learning rate, ... set in ```ds_train*.sh``` file.



## Installation

```ds_train*.sh``` files are based on `deepspeed`. To install it do the following steps:

```bash
conda create -n swiftsage python=3.8 pip
conda activate swiftsage
pip3 install scienceworld==1.1.3
pip3 install -r deepspeed_reqs.txt
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install -c conda-forge openjdk # if needed 
```