
# Generate dataset

To generate dataset do the following steps:

- First `unzip goldpaths-all.zip`. It stores the goal trajectories of all the tasks. We use the goal trajectories to generate the dataset.

- To generate the datasets you need to run `data_convert.py`. It will create a JSON file per tasks for train, test, and dev. 

- Then run `read_file.py` to generate one JSON file including all the tasks.