#!/usr/bin/env bash

#Provide your CUDA verion (ove of cu126, cu128, cu129, or cpu):
CUDA=$1
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/${CUDA}
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+${CUDA}.html
