### Experiments
This directory contains the experiments that were conducted for the project. 
The directory `data` contains the raw results of the experiments. Each group holds individual experiments in json or CSV format (one row here).
All experiments in a subfolder are plotted together on a bar plot.

The script to plot the experiments operates in 3 mode:
- "none" plots the raw results
- "delta" compares the results to the baseline (delta)
- "percent" compares the results to the baseline (%)

In order to compare to the baseline a file with a prefix `baseline` *must be present in the folder*
In the comparison modes this file will not be plotted, in "none" mode will be plotted.
This behaviour can be overridden by using the `--no_baseline_added` flag.

For more usage consult:
```bash
python plot_experiments.py -h
```

Results will appear in the `plots` directory (not synced with the repository).