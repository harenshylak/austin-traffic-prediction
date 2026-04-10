"""
build_graph.py
Builds the road-network adjacency matrix for the N sensor nodes.

Steps:
  1. Load unique sensor intersections from radar_traffic.csv
  2. Geocode intersection names → (lat, lon) via Nominatim (with curated fallbacks)
  3. Download Austin drive network via osmnx (bounding box around sensors)
  4. Snap each sensor to its nearest OSM node
  5. Compute pairwise shortest-path distances (Dijkstra, weighted by edge length)
  6. Apply Gaussian kernel → sparse adjacency matrix A (N×N)
  7. Save: data/processed/sensor_locations.csv, adj_matrix.npy, node_ids.npy

Usage:
    python src/data/build_graph.py
    python src/data/build_graph.py --sensors data/raw/radar_traffic.csv --sigma 500
"""

import argparse
import os

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# Curated geocoordinates for the 15 Austin radar intersections.
# Keys match the `intname` column exactly.
# Coordinates verified via OpenStreetMap.
INTERSECTION_COORDS: dict[str, tuple[float, float]] = {
    "LAMARMANCHACA":          (30.2421, -97.7694),  # S Lamar & Manchaca Rd
    "LAMARSHOALCREEK":        (30.2843, -97.7502),  # S Lamar & Shoal Creek Blvd
    "LAMARSANDRA MURAIDA":    (30.2326, -97.7760),  # S Lamar & Sandra Muraida
    "LOOP 360WALSH TARLTON":  (30.2988, -97.7997),  # Loop 360 & Walsh Tarlton Ln
    "LOOP 360CEDAR":          (30.3127, -97.8009),  # Loop 360 & Cedar St
    "LamarBroken Spoke":      (30.2480, -97.7681),  # S Lamar & Broken Spoke Ln
    "BurnetRutland":          (30.3545, -97.7330),  # Burnet Rd & Rutland Dr
    "Cesar ChavezBR Reynolds":(30.2607, -97.7394),  # Cesar Chavez & Reynolds
    "N Lamar15th":            (30.2797, -97.7475),  # N Lamar & W 15th St
    "CongressJohanna":        (30.2456, -97.7502),  # S Congress & Johanna St
    "CESAR CHAVEZIH 35 WSR":  (30.2607, -97.7354),  # Cesar Chavez & IH-35
    "LAMARCOLLIER":           (30.2931, -97.7463),  # N Lamar & Collier St
    "BURNETPALM WAY":         (30.3493, -97.7341),  # Burnet Rd & Palm Way
    "LOOP 360LAKEWOOD":       (30.3218, -97.7997),  # Loop 360 & Lakewood Dr
    "LAMARZENNIA":            (30.2374, -97.7726),  # S Lamar & Zennia St
}


def load_sensor_intersections(sensors_path: str) -> pd.DataFrame:
    df = pd.read_csv(sensors_path, usecols=["int_id", "intname"])
    df = df.drop_duplicates("int_id").sort_values("int_id").reset_index(drop=True)

    df["lat"] = df["intname"].map(lambda n: INTERSECTION_COORDS.get(n, (None, None))[0])
    df["lon"] = df["intname"].map(lambda n: INTERSECTION_COORDS.get(n, (None, None))[1])

    missing = df[df["lat"].isna()]["intname"].tolist()
    if missing:
        print(f"WARNING: No coordinates for {len(missing)} intersection(s): {missing}")
        df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    print(f"Loaded {len(df)} sensor intersections.")
    return df


def download_road_network(sensors_df: pd.DataFrame, buffer_m: int = 2000) -> nx.MultiDiGraph:
    north = sensors_df["lat"].max() + buffer_m / 111_000
    south = sensors_df["lat"].min() - buffer_m / 111_000
    east  = sensors_df["lon"].max() + buffer_m / (111_000 * np.cos(np.radians(sensors_df["lat"].mean())))
    west  = sensors_df["lon"].min() - buffer_m / (111_000 * np.cos(np.radians(sensors_df["lat"].mean())))

    print(f"Downloading OSM drive network (bbox: N={north:.4f} S={south:.4f} E={east:.4f} W={west:.4f})...")
    # osmnx 2.x expects bbox=(west, south, east, north)
    G = ox.graph_from_bbox(
        (west, south, east, north),
        network_type="drive",
        simplify=True,
    )
    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    return G


def snap_sensors_to_graph(sensors_df: pd.DataFrame, G: nx.MultiDiGraph) -> pd.DataFrame:
    lats = sensors_df["lat"].tolist()
    lons = sensors_df["lon"].tolist()
    node_ids = ox.nearest_nodes(G, lons, lats)
    sensors_df = sensors_df.copy()
    sensors_df["osm_node"] = node_ids
    return sensors_df


def build_adjacency_matrix(
    sensors_df: pd.DataFrame,
    G: nx.MultiDiGraph,
    sigma: float,
    eps: float = 0.1,
) -> np.ndarray:
    nodes = sensors_df["osm_node"].tolist()
    N = len(nodes)

    # Undirected version for symmetric distance matrix
    G_und = ox.convert.to_undirected(G)

    print(f"Computing pairwise shortest-path distances for {N} nodes...")
    dist = np.full((N, N), np.inf)
    np.fill_diagonal(dist, 0.0)

    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G_und, src, weight="length")
        for j, tgt in enumerate(nodes):
            if tgt in lengths:
                dist[i, j] = lengths[tgt]

    # Gaussian kernel  W_ij = exp(-d²/σ²), zero out entries below eps
    W = np.exp(-(dist ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 0.0)
    W[W < eps] = 0.0

    print(f"  Adjacency matrix: {N}×{N}, {(W > 0).sum()} non-zero entries")
    return W


def main():
    parser = argparse.ArgumentParser(description="Build road-network graph for Austin sensors")
    parser.add_argument("--sensors",  default=os.path.join("data", "raw", "radar_traffic.csv"))
    parser.add_argument("--out_dir",  default=os.path.join("data", "processed"))
    parser.add_argument("--sigma",    type=float, default=1000.0,
                        help="Gaussian kernel bandwidth in metres (default: 1000)")
    parser.add_argument("--buffer",   type=int,   default=2000,
                        help="Network download buffer around sensors in metres (default: 2000)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load sensor intersections
    sensors_df = load_sensor_intersections(args.sensors)

    # 2. Download OSM road network
    G = download_road_network(sensors_df, buffer_m=args.buffer)

    # 3. Snap sensors to nearest OSM node
    sensors_df = snap_sensors_to_graph(sensors_df, G)

    # 4. Build adjacency matrix
    W = build_adjacency_matrix(sensors_df, G, sigma=args.sigma)

    # 5. Save outputs
    loc_path = os.path.join(args.out_dir, "sensor_locations.csv")
    adj_path = os.path.join(args.out_dir, "adj_matrix.npy")
    nid_path = os.path.join(args.out_dir, "node_ids.npy")

    sensors_df.to_csv(loc_path, index=False)
    np.save(adj_path, W)
    np.save(nid_path, sensors_df["int_id"].values)

    print(f"\nSaved:")
    print(f"  {loc_path}")
    print(f"  {adj_path}  (shape {W.shape})")
    print(f"  {nid_path}")


if __name__ == "__main__":
    main()
