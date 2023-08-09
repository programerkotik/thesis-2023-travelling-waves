The preprocess script is used to prepare the data for running analysis. One should have a data/raw folder with experiments denoted with some code in separate folders. 

The script will create a data/processed folder with the following structure:

```
data/processed
    ├── experiment1
    ├── experiment2
```

To run the script, one should run the following command:

```
python processing.py --exp <experiment_code>
```

The plot.py script is used just to plot some example of raw data. 

Data parameters are stored in the data_params.yml file, it contains the information about data format and sampling rate. It could be modified but one should always check the source code of the processing.py script to make sure that the data is processed correctly. For reading the data [neo library](https://neo.readthedocs.io/en/latest/) is utilized.
