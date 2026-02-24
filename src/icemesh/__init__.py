"""Documentation about icemesh."""

import logging

from icemesh.config import Config
# from icemesh.wavi_to_mesh import transform_single_grid_to_mesh

__all__ = ['Config']
# __all__ = ['Config', 'transform_single_grid_to_mesh']

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Team Atlas"
__email__ = "m.grootes@esciencecenter.nl"
__version__ = "0.0.1"
