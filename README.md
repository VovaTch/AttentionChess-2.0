# VovaTch Pytorch Project Template

This is a fork of the Pytorch Project Template from [this link](https://github.com/victoresque/pytorch-template).

# Changes:

* Added `colorama` for text visualization, some fluff indeed.
* Changed the folder structure such that the `test.py` and `train.py` scripts are in the `scripts` folder.
* Changed the format of the config files to `*.yaml`, and the net config file is in the `config` folder.
* Added an option to use SuperConvergence with `OneCycleLR` scheduler.
* Modified the saving logic such that the model saves both checkpoints and the best models independently.

# How to use:

* Either download it as a zip file, or use `git clone https://github.com/VovaTch/pytorch-template-vovatch.git`.

# Work in Progress:

* Support for ClearML
