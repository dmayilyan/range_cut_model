# Denoising network for range cut optimization [WIP]

Parametrization is done via [Hydra](https://hydra.cc/).

# Logging

Besides Hydra default logging mechanism, a custom logging decorator is implemented. Custom logger targets logging of train/test losses along with the configuration of a given run. Decorator logging is based on *config.py* dataclasses. Columns are created based on dataclass values and types. It combines selected model (DnCNNConfig for example) dataclass recursive parameter retrieval with LogConfig dataclass which contains train/test loss, epoch values.

I would suggest using litecli (as seen in requirements-dev.txt) to quickly look into the logged data.
