"""
build_graph.py
Builds the road-network adjacency matrix for the N sensor nodes.

Steps:
  1. Download Austin drive network via osmnx
  2. Snap sensor lat/lon to nearest graph nodes
  3. Compute pairwise shortest-path distances (Dijkstra)
  4. Apply Gaussian kernel → sparse adjacency matrix A (N×N)

Fallback: if osmnx fails, computes Euclidean distances from lat/lon directly.

Usage:
    python src/data/build_graph.py --sensors data/raw/sensor_locations.csv
"""

# TODO: implement
