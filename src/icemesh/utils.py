"""Utility functions for MeshGraphNets.

Provided in the release of the code from the original MeshGraphNets study: #https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets.
"""

import enum
import tensorflow.compat.v1 as tf  # type: ignore


def triangles_to_edges(faces):
  """Computes mesh edges from triangles.

  Decomposes 2D triangular meshes to edges and returns the undirected graph nodes.
  """
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))



class NodeType(enum.IntEnum):
    """Define the code for the one-hot vector representing the node types.

    Subclass of enum with unique and unchanging integer valued attributes 
    over instances in order to make sure values are unchanged
    """
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
