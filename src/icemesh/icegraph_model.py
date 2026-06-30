"""IceGraph model, with training and testing functionality."""


import numpy as np
import os
from pathlib import Path 
import random
import time
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
import torch.optim as optim
from tqdm import trange, tqdm
import warnings



#TODO(tvl) temp imports?
from icemesh import Config



# What notebook does (broad strokes):
#
#   ARGUMENTS / CONFIG
#
# def objectview and args       [12]
# def functions for converting args <-> filename (uses [12])        [8]
#
#   NORMALIZATION
#
# def 'normalization functions' [1b]
#
#   NETWORK
#
# def building MLP (with various #in / #hidden / #out / #layers)    [2]
# def message passing ProcessorLayer {init (uses [2]), reset, forward, message, aggregate} [3]
# def MeshGraphNet torch module {init (uses [1b, 2, 3]), forward (uses [1b])}  [4]
# def CNN torch module {init, forward (uses [1b])}      [5]
# def GCN torch module {init, forward (uses [1b])}      [6]
# def function to select between modules {MGN, CNN, GCN} (uses [4, 5, 6])        [7]
#
#   DATASET
#
# def load + concatenate function       [13]        TODO(tvl) also def load (no concatenate) function
# def selecting certain k / features    [14]
#
#   TRAINING
#
# def computing stats [1a]
# script for pre-processing {collect data, split into training / validation, load data, compute normalization stats from training data, select device}
#   uses: [1a, 7, 13, 14]
#
# def building optimizer        [9]
# def validation function (uses [1b])       [10]
# def computing weights (disabled gradient calculation)     [15]
# def generic training function (uses [1b, 7, 8, 9, 10, 15])        [11]
# script for calling training and reporting results [11]
#
#   INFERENCE
#
# script for training one / multiple models for one-step-ahead prediction results (uses [1b, 7, 8, 14])
# script for checking output
# script for rollout (multi-step-ahead) prediction "until the end" (?) (uses [1b, 7, 8, 14])
#
#
# --> add scripts to a processing class that _has_ a network to use.
# for preparing, training, checking, rollout




# Things to keep track of:
# * replace args by config
# * replace strings by enumerators where possible
# * try to put model params in file (header) instead of in filename



config = Config.from_yaml("tests/icegraph_model_config.yaml")
root_dir = config.data.root_dir


