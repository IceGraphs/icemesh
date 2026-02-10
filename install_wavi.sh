#!/usr/bin/env bash

# Note that the following MUST be done outside any conda environment:
# Julia doesn't play nice with conda
sudo apt install curl
curl -fsSL https://install.julialang.org | sh # Prefered way of installing juliaup
. /home/tvl/.bashrc

julia install_wavi.jl

jupyter kernelspec list # This should (also) update the list of kernels


# To uninstall julia, sudo apt purge juliaup curl