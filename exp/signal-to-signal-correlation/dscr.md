## ECoG-Probes and ECoG-ECoG correlation
In this analysis I used signals from ECoG and probes I and II to analyse the correlation between them. 

The ecog_ecog_correlation.py script was used to calculate the correlation between the channels from the ECoG. The ecog_probes_correlation.py script was used to calculate the correlation between the signals from the ecog channels (average is used) and each channel from Probe I and II. Both scripts could be run with the following commands:

```bash
python ecog_ecog_correlation.py --exp <experiment_code>
```

```bash
python ecog_probes_correlation.py --exp <experiment_code>
```

The upstate_downstate_pcc.py perform computation of PCC (Pearson Correlation Coefficient) and maximum PCC between the signals from the ECoG (averaged across signals) and the probe for each interval of Up and Down state. The upstate_downstate_pcc.py script could be run with the following command:

```bash
python upstate_downstate_pcc.py --exp <experiment_code>
```

The output of upstate_downstate_pcc.py is csv table with pcc, maximum pcc and lag value for each interval of Up and Down state. This csvs are saved to the res/signal-to-signal-correlation folder:
    
```
res/signal-to-signal-correlation/<experiment_code>/upstate_downstate_pcc.csv
```

The csv table is analyzed and visualized by the upstate_downstate_pcc_plot.py script. The upstate_downstate_pcc_plot.py script could be run with the following command:

```bash
python upstate_downstate_pcc_plot.py>
```
