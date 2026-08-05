"""helpers and worker modules for the collapse pipeline

KEEP THIS FILE FREE OF IMPORTS.

The worker modules in this package are launched as `python -m utils.<name>`, which imports this
`__init__` *before* the module itself. Several of those workers have to import unsloth before
torch/transformers for its patches to land (utils/train_generation.py, utils/generate_dataset.py,
utils/calculate_perplexity.py, utils/generate_dataset_extrapolation.py, utils/calibrate_surrogate.py
all open with the unsloth import for that reason). Anything imported here that pulls in torch would
beat unsloth to it in every one of them, silently.

utils/devices.py is torch-free for the same reason: the orchestrators import it above their own
unsloth import.
"""
