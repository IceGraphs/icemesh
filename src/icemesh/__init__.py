"""Documentation about icemesh."""

import logging
from icemesh.config import Config #, DataConfig, ModelConfig
from icemesh.wavi_to_mesh import wavi_to_mesh, read_pt_file

__all__ = ['Config', 'wavi_to_mesh', 'read_pt_file']

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Team Atlas"
__email__ = "t.vanlankveld@esciencecenter.nl"
__version__ = "0.0.1"
