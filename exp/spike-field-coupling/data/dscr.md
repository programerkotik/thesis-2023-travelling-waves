Spiking data from cat's visual cortex extracted from 20 kHz recordings from 2022-09-27 spontaneous raw data. 
The spiking data are available only for two 16ch CSD probes. 

Channels are:
- ch65-80   16ch CSD probe 1

- ch97-112   16 ch CSD probe 2

The details on spike extraction can be found in the spike extraction notebook and helper functions in src/spike_extraction.py. The control is generated using [Poisson process](https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_generation/elephant.spike_train_generation.StationaryPoissonProcess.html#elephant.spike_train_generation.StationaryPoissonProcess) with rate according to the real rate of that channel.