# Denoising network for range cut optimization [WIP]

Parametrization is done via [Hydra](https://hydra.cc/).

# Logging

Logging is done via a custom decorator, which by default creates a `log.db` sqlite3 database, which logs training parameters and loss over epochs.

I would suggest using litecli (as seen in requirements-dev.txt) to quickly look into the logged data.
