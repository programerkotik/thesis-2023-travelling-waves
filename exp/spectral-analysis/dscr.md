## Oscillation analysis
In this experiment I used [FOOOF](https://fooof-tools.github.io/fooof/auto_tutorials/index.html) library to analyse oscillatory properies of the LFP signal from probe I and II. 

The spectral_analysis_by_intervals.py is the main script for this analysis. It performs the spectral analysis using FOOOF for each Up and Down state interval and save result to a csv table to the res/spectral-analysis folder:
    
```
res/spectral-analysis/<experiment_code>/spectral_analysis_by_intervals.csv
```

The script could be run with the following command:

```bash
python spectral_analysis_by_intervals.py --exp <experiment_code>
```

The csv table is analyzed and visualized in the jupyter notebook spectral_plots_intervals.ipynb. 

The script general_spectral_analysis.py computes the spectral analysis without FOOOF for the whole signal. The band_resolved_spectral_props.py and depth_resolved_spectral_props.py use csv table from res/spectral-analysis/<experiment_code>/spectral_analysis_by_intervals.csv to visualize the spectral properties of the signal. 

The line_noise_filter_test.py was used to test the success of the line noise filtering.