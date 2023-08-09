This folder includes a preprocessing step to get upstate regions in ECoG-generated data. Since the signals from different ECoG channels are similar, upstates are defined based on the behavior across all channels. Upstate intervals are identified in the low-pass filtered data using a thresholding method. The mean and standard deviation of each channel determine the threshold value. Obtained data about upstates are also used later in other experiments. 

Resulting upstates are saved as numpy .npy files in the data/processed folder. The structure of the data is the following:

```
data/processed
    ├── experiment1
        ├── event_times.npy
    ├── experiment2
        ├── event_times.npy
```

Find_upstates.yml file stores the parameters used for Up and Down state detection. The parameters are the following:

- **f_range** - tuple of low cutoff frequency for the filter and high cutoff frequency for the filter
- **threshold_scalar** - threshold for the detection of upstates

