## Spike-field coupling
In the next set of experiments I have examined spike-field coupling (SFC) between the LFP signal and the spike trains from the probe I and II. In this experiments I have used the same data as in the previous 2022-10-05 experiment but sampled with 20 kHz frequency. This sampling rate should be sufficient to catch the action potentials (AP). 

To analyse the spike-field coupling first of all spikes should have been assessed from the raw data. The strategy for spike detection was the following: first only high frequencies were filtered using band-pass [30-300] filter, then the threshold was set to mean - threshold_cutoff*std. Functions for spike detection are available in source code. The resulting spikes and control spiking data are saved to local data folder.