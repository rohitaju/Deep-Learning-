# CSA47 – Deep Learning Lab: 80 Programs

This archive contains one Python program for each of the 80 experiments in the supplied CSA47 Deep Learning Lab document.

## Folder structure
- `Experiment_01/` through `Experiment_80/`
  - `experiment_NN.py` — runnable source code for that experiment
  - `experiment_NN_output.txt` — captured console output from running the program
  - `experiment_NN_output.png` — generated plot/visual output

## Run a program
Install the dependencies:
```bash
python -m pip install -r requirements.txt
```
Then open a terminal in the relevant experiment folder and run, for example:
```bash
python experiment_01.py
```

## Notes
- The programs use deterministic synthetic/sample data so they can run without downloading private or external datasets.
- Each script prints its computed results and saves a plot in its own folder.
- Deep-learning experiments that ordinarily require TensorFlow/Keras or large datasets are represented by lightweight, executable numerical demonstrations and workflow scaffolds. TensorFlow is not required for this pack. For full dataset-scale CNN/RNN/VAE/GAN training, adapt the scripts to your instructor-provided dataset and install the relevant framework.
- The text output files are the console outputs captured when these programs were executed in this environment.
