# Leveraging contextual events on structure-aware next activity prediction
This code is the official implementation of the paper "Leveraging contextual events on structure-aware next activity prediction" presented at the AI4PM workshop of ECML-PKDD 2026 conference. 

## Prerequisite
- Install Python 3.10
- Install requirements using pip: `pip install -r requirements.txt`

## How to run
To start code, run: `hyper_parameters_search_all_variants.py`.
According to the methodology described in the paper, the code:
- reads the IGs file (`.g` in `dataset` folder);
- discovers, for each event, its set of contextual events;
- encodes the prefix-IGs according to `var_*`;
- runs the hyper-parameter search of the GNN using Optuna.
- 
## Cite

