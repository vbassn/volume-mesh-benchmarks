import numpy as np
import meshio
import networkx as nx
from scipy.spatial import KDTree
import heapq
from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import json

def _process_vertex_chunk(vertex_chunk, tetrahedra):
    """Process a chunk of vertices to find potentially adjacent tetrahedra."""
    candidates = set()
    
    for vertex, tet_list in vertex_chunk:
        # All pairs of tetrahedra sharing this vertex
        for i in range(len(tet_list)):
            for j in range(i + 1, len(tet_list)):
                tet_i, tet_j = tet_list[i], tet_list[j]
                if tet_i < tet_j:
                    candidates.add((tet_i, tet_j))
                else:
                    candidates.add((tet_j, tet_i))
    
    return candidates


def _check_adjacency_chunk(edge_chunk, tetrahedra, centroids):
    """Check which candidate edges are actual adjacencies."""
    edges = []
    for i, j in edge_chunk:
        shared = len(set(tetrahedra[i]).intersection(set(tetrahedra[j])))
        if shared == 3:  # Face-adjacent
            dist = np.linalg.norm(centroids[i] - centroids[j])
            edges.append((i, j, dist))
    
    return edges


def _process_tetrahedra_batch_parallel(batch, shared_data):
    """Process a batch of tetrahedra to find adjacent pairs."""
    start_idx, end_idx = batch
    centroids = shared_data['centroids']
    vertex_sets = shared_data['vertex_sets']
    mesh_points = shared_data['mesh_points']
    tetrahedra = shared_data['tetrahedra']
    
    # Rebuild KD-tree in this process
    kdtree = KDTree(centroids)
    
    edges = []
    
    for i in range(start_idx, end_idx):
        # Estimate search radius
        vertices_i = mesh_points[tetrahedra[i]]
        edge_lengths = [np.linalg.norm(vertices_i[j] - vertices_i[k]) 
                       for j in range(4) for k in range(j+1, 4)]
        search_radius = np.max(edge_lengths) * 1.5
        
        # Find candidates
        candidates = kdtree.query_ball_point(centroids[i], search_radius)
        
        # Check adjacency
        tet_i_verts = vertex_sets[i]
        
        for j in candidates:
            if j > i:  # Only check each pair once
                # Check shared vertices
                shared_vertices = len(tet_i_verts.intersection(vertex_sets[j]))
                
                if shared_vertices == 3:  # Adjacent
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    edges.append((i, j, dist))
    
    return edges


class MLEnhancedPathfinder:
    """
    A machine learning-enhanced pathfinding system for drones navigating 
    through a city represented as a tetrahedral mesh in VTU format.
    """
    
    def __init__(self, vtu_file: str, enable_ml: bool = True):
        """
        Initialize the pathfinder with a VTU mesh file.
        
        Args:
            vtu_file: Path to the VTU file containing the tetrahedral mesh
            enable_ml: Whether to enable ML features
        """
        self.mesh = None
        self.graph = nx.Graph()
        self.centroids = []
        self.kdtree = None
        self.tetrahedra = []
        self.enable_ml = enable_ml
        
        # ML components
        self.cost_predictor = None
        self.safety_predictor = None
        self.cost_scaler = StandardScaler()
        self.safety_scaler = StandardScaler()
        self.path_history = []
        self.obstacle_memory = set()
        self.wind_predictor = None
        
        # Load mesh and build graph
        self.load_mesh(vtu_file)
        self.build_navigation_graph_parallel_fast()
        
        if self.enable_ml:
            self.initialize_ml_models()
    
    #---
    def save_path_for_paraview(self, path, filename_base="drone_path"):
        """
        Save the path in formats compatible with ParaView visualization.
        
        Args:
            path: List of waypoints (3D coordinates)
            filename_base: Base filename without extension
        """
        if not path:
            print("No path to save")
            return
        
        path_array = np.array(path)
        
        # Method 1: Save as VTK PolyData (recommended)
        self._save_path_as_vtk(path_array, f"{filename_base}.vtk")
        
        # Method 2: Save as VTU (UnstructuredGrid)
        self._save_path_as_vtu(path_array, f"{filename_base}.vtu")
        
        # Method 3: Save as CSV (can be imported to ParaView)
        self._save_path_as_csv(path_array, f"{filename_base}.csv")
        
        # Method 4: Save as PLY (simple format)
        self._save_path_as_ply(path_array, f"{filename_base}.ply")

    def _save_path_as_vtk(self, path_array, filename):
        """Save path as VTK PolyData with lines."""
        with open(filename, 'w') as f:
            # VTK header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Drone Path\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            
            # Points
            n_points = len(path_array)
            f.write(f"POINTS {n_points} float\n")
            for point in path_array:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
            
            # Lines connecting the points
            f.write(f"\nLINES 1 {n_points + 1}\n")
            f.write(f"{n_points}")
            for i in range(n_points):
                f.write(f" {i}")
            f.write("\n")
            
            # Add scalar data (e.g., altitude)
            f.write(f"\nPOINT_DATA {n_points}\n")
            f.write("SCALARS altitude float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for point in path_array:
                f.write(f"{point[2]}\n")
            
            # Add distance along path
            f.write("\nSCALARS distance_along_path float 1\n")
            f.write("LOOKUP_TABLE default\n")
            distances = [0]
            for i in range(1, n_points):
                dist = distances[-1] + np.linalg.norm(path_array[i] - path_array[i-1])
                distances.append(dist)
            for d in distances:
                f.write(f"{d}\n")
        
        print(f"Saved path as VTK to {filename}")

    def _save_path_as_vtu(self, path_array, filename):
        """Save path as VTU using meshio."""
        import meshio
        
        # Create line cells connecting consecutive points
        lines = []
        for i in range(len(path_array) - 1):
            lines.append([i, i + 1])
        
        # Create mesh with lines
        cells = [("line", np.array(lines))]
        
        # Add data
        point_data = {
            "altitude": path_array[:, 2],
            "waypoint_index": np.arange(len(path_array))
        }
        
        # Calculate distance along path
        distances = [0]
        for i in range(1, len(path_array)):
            dist = distances[-1] + np.linalg.norm(path_array[i] - path_array[i-1])
            distances.append(dist)
        point_data["distance_along_path"] = np.array(distances)
        
        mesh = meshio.Mesh(
            points=path_array,
            cells=cells,
            point_data=point_data
        )
        
        meshio.write(filename, mesh)
        print(f"Saved path as VTU to {filename}")

    def _save_path_as_csv(self, path_array, filename):
        """Save path as CSV file."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z', 'WaypointIndex'])
            
            for i, point in enumerate(path_array):
                writer.writerow([point[0], point[1], point[2], i])
        
        print(f"Saved path as CSV to {filename}")
        print("  In ParaView: File > Open > Select CSV > Apply")
        print("  Then: Filters > Table To Points > Set X,Y,Z columns > Apply")

    def _save_path_as_ply(self, path_array, filename):
        """Save path as PLY file with edges."""
        with open(filename, 'w') as f:
            n_vertices = len(path_array)
            n_edges = n_vertices - 1
            
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element edge {n_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("end_header\n")
            
            # Vertices
            for point in path_array:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
            
            # Edges
            for i in range(n_edges):
                f.write(f"{i} {i+1}\n")
        
        print(f"Saved path as PLY to {filename}")

    def save_path_with_mesh_info(self, path, filename="drone_path_analysis.vtu"):
        """
        Save path with additional analysis data for ParaView.
        Includes safety scores, traversal costs, etc.
        """
        import meshio
        
        if not path or not self.enable_ml:
            self._save_path_as_vtu(np.array(path), filename)
            return
        
        path_array = np.array(path)
        
        # Create line cells
        lines = []
        for i in range(len(path_array) - 1):
            lines.append([i, i + 1])
        
        cells = [("line", np.array(lines))]
        
        # Gather analysis data
        point_data = {
            "altitude": path_array[:, 2],
            "waypoint_index": np.arange(len(path_array))
        }
        
        # Add ML predictions if available
        if self.enable_ml:
            safety_scores = []
            for waypoint in path:
                tet_idx = self.find_containing_tetrahedron(waypoint)
                if tet_idx is not None:
                    safety = self.predict_safety_score(tet_idx)
                    safety_scores.append(safety)
                else:
                    safety_scores.append(1.0)
            
            point_data["safety_score"] = np.array(safety_scores)
        
        # Distance along path
        distances = [0]
        for i in range(1, len(path_array)):
            dist = distances[-1] + np.linalg.norm(path_array[i] - path_array[i-1])
            distances.append(dist)
        point_data["distance_along_path"] = np.array(distances)
        
        # Cell data (for line segments)
        if len(lines) > 0:
            segment_costs = []
            for i in range(len(path) - 1):
                cost = np.linalg.norm(path_array[i+1] - path_array[i])
                segment_costs.append(cost)
            
            cell_data = {
                "line": {
                    "segment_cost": np.array(segment_costs)
                }
            }
        else:
            cell_data = None
        
        # Create mesh
        mesh = meshio.Mesh(
            points=path_array,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data
        )
        
        meshio.write(filename, mesh)
        print(f"Saved path with analysis data to {filename}")


##
    def add_bounding_box_obstacle(self, bbox, z_range=None, padding=0.0):
        """
        Add an obstacle in the form of a bounding box.
        
        Args:
            bbox: Tuple/list of (xmin, ymin, xmax, ymax)
            z_range: Optional tuple of (zmin, zmax). If None, covers all altitudes
            padding: Additional padding around the box (meters)
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Apply padding
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding
        
        # Determine z range
        if z_range is None:
            zmin = self.mesh.points[:, 2].min()
            zmax = self.mesh.points[:, 2].max()
        else:
            zmin, zmax = z_range
        
        print(f"Adding bounding box obstacle: X[{xmin:.1f}, {xmax:.1f}], "
            f"Y[{ymin:.1f}, {ymax:.1f}], Z[{zmin:.1f}, {zmax:.1f}]")
        
        # Find all tetrahedra inside or intersecting the box
        affected_tets = set()
        removed_edges = []
        
        for i in range(len(self.tetrahedra)):
            centroid = self.centroids[i]
            
            # Check if centroid is inside the box
            if (xmin <= centroid[0] <= xmax and
                ymin <= centroid[1] <= ymax and
                zmin <= centroid[2] <= zmax):
                affected_tets.add(i)
                
                # Remove this node from the graph
                if i in self.graph:
                    # Store edges before removal for potential restoration
                    for neighbor in list(self.graph.neighbors(i)):
                        removed_edges.append((i, neighbor, 
                                            self.graph[i][neighbor]['weight']))
                    self.graph.remove_node(i)
        
        # Also check tetrahedra that might intersect the box
        for i in range(len(self.tetrahedra)):
            if i in affected_tets:
                continue
                
            vertices = self.mesh.points[self.tetrahedra[i]]
            
            # Check if any vertex is inside the box
            for vertex in vertices:
                if (xmin <= vertex[0] <= xmax and
                    ymin <= vertex[1] <= ymax and
                    zmin <= vertex[2] <= zmax):
                    affected_tets.add(i)
                    if i in self.graph:
                        for neighbor in list(self.graph.neighbors(i)):
                            removed_edges.append((i, neighbor,
                                                self.graph[i][neighbor]['weight']))
                        self.graph.remove_node(i)
                    break
        
        # Store obstacle info
        self.bbox_obstacles = getattr(self, 'bbox_obstacles', [])
        self.bbox_obstacles.append({
            'bbox': (xmin, ymin, xmax, ymax),
            'z_range': (zmin, zmax),
            'affected_tets': affected_tets,
            'removed_edges': removed_edges
        })
        
        # Add corner points to obstacle memory for ML
        corners = [
            (xmin, ymin, (zmin + zmax) / 2),
            (xmax, ymin, (zmin + zmax) / 2),
            (xmin, ymax, (zmin + zmax) / 2),
            (xmax, ymax, (zmin + zmax) / 2)
        ]
        for corner in corners:
            self.obstacle_memory.add(corner)
        
        print(f"Removed {len(affected_tets)} tetrahedra from navigation graph")
        
        # Check connectivity
        if self.graph.number_of_nodes() > 0:
            n_components = nx.number_connected_components(self.graph)
            if n_components > 1:
                print(f"Warning: Graph now has {n_components} disconnected components!")

    def add_multiple_bbox_obstacles(self, obstacles_list):
        """
        Add multiple bounding box obstacles at once.
        
        Args:
            obstacles_list: List of obstacle definitions, each can be:
                - (xmin, ymin, xmax, ymax) for full height
                - ((xmin, ymin, xmax, ymax), (zmin, zmax)) for specific height
                - {'bbox': (xmin, ymin, xmax, ymax), 'z_range': (zmin, zmax), 'padding': float}
        """
        for obstacle in obstacles_list:
            if isinstance(obstacle, dict):
                self.add_bounding_box_obstacle(
                    obstacle['bbox'],
                    obstacle.get('z_range'),
                    obstacle.get('padding', 0.0)
                )
            elif isinstance(obstacle, tuple) and len(obstacle) == 2:
                # Format: (bbox, z_range)
                self.add_bounding_box_obstacle(obstacle[0], obstacle[1])
            else:
                # Format: just bbox
                self.add_bounding_box_obstacle(obstacle)

    def is_path_blocked_by_bbox(self, start, end, bbox, z_range=None):
        """
        Check if a line segment intersects with a bounding box.
        
        Args:
            start: Start point (x, y, z)
            end: End point (x, y, z)
            bbox: (xmin, ymin, xmax, ymax)
            z_range: Optional (zmin, zmax)
        """
        xmin, ymin, xmax, ymax = bbox
        
        if z_range:
            zmin, zmax = z_range
        else:
            zmin = min(start[2], end[2]) - 1
            zmax = max(start[2], end[2]) + 1
        
        # Simple line-box intersection test
        # This is a simplified version - for production use, implement
        # proper 3D line-box intersection
        
        # Check if line passes through box in 2D
        # ... (implement full intersection test if needed)
        
        return False  # Simplified for now

    def visualize_with_obstacles(self, path=None, save_as="path_with_obstacles.vtu"):
        """
        Save visualization data including bounding box obstacles.
        """
        import meshio
        
        # Create obstacle visualization data
        obstacle_points = []
        obstacle_cells = []
        
        if hasattr(self, 'bbox_obstacles'):
            for obs in self.bbox_obstacles:
                xmin, ymin, xmax, ymax = obs['bbox']
                zmin, zmax = obs['z_range']
                
                # Create box vertices
                base_idx = len(obstacle_points)
                box_vertices = [
                    [xmin, ymin, zmin], [xmax, ymin, zmin],
                    [xmax, ymax, zmin], [xmin, ymax, zmin],
                    [xmin, ymin, zmax], [xmax, ymin, zmax],
                    [xmax, ymax, zmax], [xmin, ymax, zmax]
                ]
                obstacle_points.extend(box_vertices)
                
                # Create box edges (12 edges of a box)
                edges = [
                    # Bottom face
                    [base_idx+0, base_idx+1], [base_idx+1, base_idx+2],
                    [base_idx+2, base_idx+3], [base_idx+3, base_idx+0],
                    # Top face
                    [base_idx+4, base_idx+5], [base_idx+5, base_idx+6],
                    [base_idx+6, base_idx+7], [base_idx+7, base_idx+4],
                    # Vertical edges
                    [base_idx+0, base_idx+4], [base_idx+1, base_idx+5],
                    [base_idx+2, base_idx+6], [base_idx+3, base_idx+7]
                ]
                obstacle_cells.extend(edges)
        
        # Combine with path if provided
        if path:
            path_array = np.array(path)
            all_points = np.vstack([path_array, np.array(obstacle_points)])
            
            # Path lines
            path_lines = [[i, i+1] for i in range(len(path)-1)]
            
            # Adjust obstacle line indices
            obstacle_lines = [[i+len(path), j+len(path)] for i, j in obstacle_cells]
            
            cells = [
                ("line", np.array(path_lines)),
                ("line", np.array(obstacle_lines))
            ]
            
            # Create point data
            point_data = {
                "is_path": np.concatenate([
                    np.ones(len(path)),
                    np.zeros(len(obstacle_points))
                ])
            }
        else:
            all_points = np.array(obstacle_points)
            cells = [("line", np.array(obstacle_cells))]
            point_data = {}
        
        # Save mesh
        mesh = meshio.Mesh(
            points=all_points,
            cells=cells,
            point_data=point_data
        )
        
        meshio.write(save_as, mesh)
        print(f"Saved obstacles and path to {save_as}")

    def remove_obstacle(self, obstacle_index):
        """
        Remove a previously added bounding box obstacle.
        
        Args:
            obstacle_index: Index of the obstacle to remove
        """
        if not hasattr(self, 'bbox_obstacles') or obstacle_index >= len(self.bbox_obstacles):
            print(f"No obstacle at index {obstacle_index}")
            return
        
        obs = self.bbox_obstacles[obstacle_index]
        
        # Restore removed edges
        for i, j, weight in obs['removed_edges']:
            if i < len(self.tetrahedra) and j < len(self.tetrahedra):
                self.graph.add_edge(i, j, weight=weight)
        
        # Remove from list
        self.bbox_obstacles.pop(obstacle_index)
        
        print(f"Removed obstacle {obstacle_index}, restored {len(obs['removed_edges'])} edges")
##
    def load_mesh(self, vtu_file: str):
        """Load the tetrahedral mesh from VTU file."""
        try:
            self.mesh = meshio.read(vtu_file)
            print(f"Loaded mesh with {len(self.mesh.points)} vertices")
            
            # Extract tetrahedral cells
            for cell_block in self.mesh.cells:
                if cell_block.type == "tetra":
                    self.tetrahedra = cell_block.data
                    print(f"Found {len(self.tetrahedra)} tetrahedra")
                    break
            
            if len(self.tetrahedra) == 0:
                raise ValueError("No tetrahedral cells found in mesh")
            
            # Print mesh bounds
            xmin, ymin, zmin = self.mesh.points.min(axis=0)
            xmax, ymax, zmax = self.mesh.points.max(axis=0)
            print(f"Mesh bounds:")
            print(f"  X: [{xmin:.2f}, {xmax:.2f}] (width: {xmax-xmin:.2f})")
            print(f"  Y: [{ymin:.2f}, {ymax:.2f}] (height: {ymax-ymin:.2f})")
            print(f"  Z: [{zmin:.2f}, {zmax:.2f}] (depth: {zmax-zmin:.2f})")
                
        except Exception as e:
            print(f"Error loading mesh: {e}")
            raise


    def __init__(self, vtu_file: str, enable_ml: bool = True, 
                subdomain: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize the pathfinder with a VTU mesh file.
        
        Args:
            vtu_file: Path to the VTU file containing the tetrahedral mesh
            enable_ml: Whether to enable ML features
            subdomain: Optional dictionary defining subdomain bounds
                    e.g., {'x': (100, 500), 'y': (200, 600), 'z': (0, 100)}
                    If provided, only tetrahedra within these bounds are used
        """
        self.mesh = None
        self.graph = nx.Graph()
        self.centroids = []
        self.kdtree = None
        self.tetrahedra = []
        self.enable_ml = enable_ml
        self.subdomain = subdomain
        
        # ML components
        self.cost_predictor = None
        self.safety_predictor = None
        self.cost_scaler = StandardScaler()
        self.safety_scaler = StandardScaler()
        self.path_history = []
        self.obstacle_memory = set()
        self.wind_predictor = None
        
        # Load mesh and build graph
        self.load_mesh(vtu_file)
        
        # Filter mesh if subdomain is specified
        if self.subdomain:
            self.filter_mesh_to_subdomain()
        
        self.build_navigation_graph_parallel_fast()
        
        if self.enable_ml:
            self.initialize_ml_models()


    def filter_mesh_to_subdomain(self):
        """Filter tetrahedra to only include those within the specified subdomain."""
        if not self.subdomain:
            return
        
        print(f"\nFiltering mesh to subdomain:")
        print(f"  X: {self.subdomain.get('x', 'all')}")
        print(f"  Y: {self.subdomain.get('y', 'all')}")
        print(f"  Z: {self.subdomain.get('z', 'all')}")
        
        # Get bounds
        x_bounds = self.subdomain.get('x', (-np.inf, np.inf))
        y_bounds = self.subdomain.get('y', (-np.inf, np.inf))
        z_bounds = self.subdomain.get('z', (-np.inf, np.inf))
        
        # Filter tetrahedra
        filtered_tetrahedra = []
        
        for i, tet in enumerate(self.tetrahedra):
            # Get centroid of tetrahedron
            vertices = self.mesh.points[tet]
            centroid = np.mean(vertices, axis=0)
            
            # Check if centroid is within bounds
            if (x_bounds[0] <= centroid[0] <= x_bounds[1] and
                y_bounds[0] <= centroid[1] <= y_bounds[1] and
                z_bounds[0] <= centroid[2] <= z_bounds[1]):
                filtered_tetrahedra.append(tet)
        
        # Update tetrahedra
        original_count = len(self.tetrahedra)
        self.tetrahedra = np.array(filtered_tetrahedra)
        
        print(f"Filtered from {original_count} to {len(self.tetrahedra)} tetrahedra")
        print(f"Reduction: {(1 - len(self.tetrahedra)/original_count)*100:.1f}%")
        
        if len(self.tetrahedra) == 0:
            raise ValueError("No tetrahedra found in specified subdomain!")



    #---

    def load_mesh_old(self, vtu_file: str):
        """Load the tetrahedral mesh from VTU file."""
        try:
            self.mesh = meshio.read(vtu_file)
            print(f"Loaded mesh with {len(self.mesh.points)} vertices")
            
            # Extract tetrahedral cells
            for cell_block in self.mesh.cells:
                if cell_block.type == "tetra":
                    self.tetrahedra = cell_block.data
                    print(f"Found {len(self.tetrahedra)} tetrahedra")
                    break
            
            if len(self.tetrahedra) == 0:
                raise ValueError("No tetrahedral cells found in mesh")
                
        except Exception as e:
            print(f"Error loading mesh: {e}")
            raise
    
    def initialize_ml_models(self):
        """Initialize machine learning models for path optimization."""
        # Cost prediction model - predicts actual traversal cost based on features
        self.cost_predictor = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=1000,
            random_state=42
        )
        
        # Safety prediction model - predicts collision risk
        self.safety_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Wind prediction model - predicts wind conditions
        self.wind_predictor = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=1000,
            random_state=42
        )
        
        # Initialize with synthetic training data if no history exists
        if len(self.path_history) == 0:
            self.generate_synthetic_training_data()
    
    def extract_tetrahedron_features(self, tet_idx: int) -> np.ndarray:
        """
        Extract ML features from a tetrahedron.
        
        Features include:
        - Centroid position (x, y, z)
        - Volume
        - Altitude
        - Distance from mesh boundary
        - Number of neighbors
        - Average edge length
        """
        vertices = self.mesh.points[self.tetrahedra[tet_idx]]
        centroid = np.mean(vertices, axis=0)
        
        # Calculate volume using determinant formula
        v0, v1, v2, v3 = vertices
        volume = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
        
        # Altitude (z-coordinate)
        altitude = centroid[2]
        
        # Number of neighbors
        n_neighbors = self.graph.degree(tet_idx)
        
        # Average edge length
        edge_lengths = []
        for neighbor in self.graph.neighbors(tet_idx):
            edge_lengths.append(self.graph[tet_idx][neighbor]['weight'])
        avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0
        
        # Distance to nearest obstacle (if any)
        obstacle_dist = self.get_min_obstacle_distance(centroid)
        
        features = np.array([
            centroid[0], centroid[1], centroid[2],  # Position
            volume,                                   # Size
            altitude,                                 # Height
            n_neighbors,                             # Connectivity
            avg_edge_length,                         # Local mesh density
            obstacle_dist,                           # Safety metric
            self.get_time_features()                 # Time of day (affects traffic)
        ])
        
        return features
    
    def get_time_features(self) -> float:
        """Get time-based features (normalized hour of day)."""
        current_hour = datetime.now().hour
        return current_hour / 24.0
    
    def get_min_obstacle_distance(self, point: np.ndarray) -> float:
        """Calculate minimum distance to known obstacles."""
        if not self.obstacle_memory:
            return 100.0  # Large default distance
        
        min_dist = float('inf')
        for obstacle in self.obstacle_memory:
            dist = np.linalg.norm(point - np.array(obstacle))
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def predict_traversal_cost(self, tet_idx: int, 
                              next_tet_idx: int) -> float:
        """
        Use ML to predict the actual cost of traversing between tetrahedra.
        
        This can account for factors like:
        - Wind conditions
        - Drone traffic
        - Safety considerations
        """
        if not self.enable_ml or self.cost_predictor is None:
            # Fallback to Euclidean distance
            return self.graph[tet_idx][next_tet_idx]['weight']
        
        # Extract features for both tetrahedra
        features1 = self.extract_tetrahedron_features(tet_idx)
        features2 = self.extract_tetrahedron_features(next_tet_idx)
        
        # Combine features
        combined_features = np.concatenate([features1, features2])
        
        try:
            # Predict cost
            features_scaled = self.cost_scaler.transform([combined_features])
            predicted_cost = self.cost_predictor.predict(features_scaled)[0]
            
            # Ensure non-negative cost
            base_cost = self.graph[tet_idx][next_tet_idx]['weight']
            return max(base_cost, predicted_cost)
        except:
            # If prediction fails, use base cost
            return self.graph[tet_idx][next_tet_idx]['weight']
    
    def predict_safety_score(self, tet_idx: int) -> float:
        """
        Predict safety score for a tetrahedron (0-1, higher is safer).
        """
        if not self.enable_ml or self.safety_predictor is None:
            return 1.0  # Default to safe
        
        features = self.extract_tetrahedron_features(tet_idx)
        
        try:
            features_scaled = self.safety_scaler.transform([features])
            safety = self.safety_predictor.predict(features_scaled)[0]
            return np.clip(safety, 0, 1)
        except:
            return 1.0
    
    def find_path_ml(self, start: np.ndarray, goal: np.ndarray,
                     safety_weight: float = 0.3) -> Optional[List[np.ndarray]]:
        """
        Find a path using ML-enhanced A* algorithm.
        
        Args:
            start: 3D start coordinates
            goal: 3D goal coordinates
            safety_weight: Weight for safety vs speed (0-1)
            
        Returns:
            List of waypoints or None
        """
        start_tet = self.find_containing_tetrahedron(start)
        goal_tet = self.find_containing_tetrahedron(goal)
        
        if start_tet is None or goal_tet is None:
            return None
        
        # A* with ML-predicted costs
        open_set = [(0, start_tet)]
        g_score = {start_tet: 0}
        f_score = {start_tet: self.heuristic(start_tet, goal_tet)}
        came_from = {}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_tet:
                # Reconstruct path
                path = self.reconstruct_path(came_from, current)
                waypoints = self.path_to_waypoints(path, start, goal)
                
                # Learn from this path
                if self.enable_ml:
                    self.update_ml_models(path, waypoints)
                
                return waypoints
            
            for neighbor in self.graph.neighbors(current):
                # ML-enhanced cost calculation
                if self.enable_ml:
                    move_cost = self.predict_traversal_cost(current, neighbor)
                    safety = self.predict_safety_score(neighbor)
                    
                    # Combine cost with safety
                    total_cost = move_cost * (1 + safety_weight * (1 - safety))
                else:
                    total_cost = self.graph[current][neighbor]['weight']
                
                tentative_g = g_score[current] + total_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal_tet)
                    f_score[neighbor] = f
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f, neighbor))
        
        return None
    
    def heuristic(self, tet1: int, tet2: int) -> float:
        """Heuristic function for A* (Euclidean distance)."""
        return np.linalg.norm(self.centroids[tet1] - self.centroids[tet2])
    
    def reconstruct_path(self, came_from: Dict[int, int], 
                        current: int) -> List[int]:
        """Reconstruct path from A* search."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))
    
    def path_to_waypoints(self, tet_path: List[int], 
                         start: np.ndarray, 
                         goal: np.ndarray) -> List[np.ndarray]:
        """Convert tetrahedron path to waypoints."""
        waypoints = [start]
        for tet_idx in tet_path[1:-1]:
            waypoints.append(self.centroids[tet_idx])
        waypoints.append(goal)
        return waypoints
    
    def update_ml_models(self, path: List[int], 
                        waypoints: List[np.ndarray]):
        """
        Update ML models with new path data.
        """
        # Store path in history
        self.path_history.append({
            'path': path,
            'waypoints': waypoints,
            'timestamp': datetime.now(),
            'success': True  # Could be updated based on actual flight
        })
        
        # Retrain models periodically
        if len(self.path_history) % 10 == 0:
            self.retrain_models()
    
    def add_obstacle(self, position: np.ndarray, radius: float = 10.0):
        """
        Add a discovered obstacle to the system.
        
        Args:
            position: 3D position of obstacle
            radius: Obstacle radius
        """
        self.obstacle_memory.add(tuple(position))
        
        # Update affected tetrahedra
        affected_tets = []
        for i, centroid in enumerate(self.centroids):
            if np.linalg.norm(centroid - position) < radius * 2:
                affected_tets.append(i)
        
        # Trigger model update for affected area
        if self.enable_ml and affected_tets:
            print(f"Added obstacle at {position}, affecting {len(affected_tets)} tetrahedra")
    
    def generate_synthetic_training_data(self):
        """Generate synthetic training data for initial ML models."""
        n_samples = 1000
        X_cost = []
        y_cost = []
        X_safety = []
        y_safety = []
        
        for _ in range(n_samples):
            # Random tetrahedron pairs
            tet1 = np.random.randint(0, len(self.tetrahedra))
            neighbors = list(self.graph.neighbors(tet1))
            if not neighbors:
                continue
            tet2 = np.random.choice(neighbors)
            
            # Features
            features1 = self.extract_tetrahedron_features(tet1)
            features2 = self.extract_tetrahedron_features(tet2)
            combined = np.concatenate([features1, features2])
            
            # Synthetic cost (base + noise + altitude penalty)
            base_cost = self.graph[tet1][tet2]['weight']
            altitude_penalty = 0.1 * abs(features1[2] - features2[2])
            cost = base_cost + altitude_penalty + np.random.normal(0, 0.1)
            
            X_cost.append(combined)
            y_cost.append(cost)
            
            # Synthetic safety (higher altitude = safer, with noise)
            safety = np.clip(0.5 + 0.3 * (features1[2] / 100) + 
                           np.random.normal(0, 0.1), 0, 1)
            X_safety.append(features1)
            y_safety.append(safety)
        
        # Train models
        if X_cost:
            X_cost = np.array(X_cost)
            X_safety = np.array(X_safety)
            
            # Fit and transform cost features
            self.cost_scaler.fit(X_cost)
            X_cost_scaled = self.cost_scaler.transform(X_cost)
            self.cost_predictor.fit(X_cost_scaled, y_cost)
            
            # Fit and transform safety features separately
            self.safety_scaler.fit(X_safety)
            X_safety_scaled = self.safety_scaler.transform(X_safety)
            self.safety_predictor.fit(X_safety_scaled, y_safety)
            
            print("ML models initialized with synthetic data")
    
    def save_ml_models(self, filepath: str):
        """Save trained ML models to disk."""
        if not self.enable_ml:
            return
        
        models = {
            'cost_predictor': self.cost_predictor,
            'safety_predictor': self.safety_predictor,
            'wind_predictor': self.wind_predictor,
            'cost_scaler': self.cost_scaler,
            'safety_scaler': self.safety_scaler,
            'path_history': self.path_history,
            'obstacle_memory': list(self.obstacle_memory)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        print(f"ML models saved to {filepath}")
    
    def load_ml_models(self, filepath: str):
        """Load trained ML models from disk."""
        try:
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
            
            self.cost_predictor = models['cost_predictor']
            self.safety_predictor = models['safety_predictor']
            self.wind_predictor = models['wind_predictor']
            self.cost_scaler = models['cost_scaler']
            self.safety_scaler = models['safety_scaler']
            self.path_history = models['path_history']
            self.obstacle_memory = set(models['obstacle_memory'])
            
            print(f"ML models loaded from {filepath}")
            self.enable_ml = True
        except Exception as e:
            print(f"Error loading ML models: {e}")
    
    def visualize_ml_insights(self, path: Optional[List[np.ndarray]] = None):
        """
        Visualize the mesh with ML insights (safety scores, predicted costs).
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 3D path visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot mesh with safety coloring
        if self.enable_ml:
            safety_scores = []
            for i in range(min(len(self.tetrahedra), 500)):
                safety = self.predict_safety_score(i)
                safety_scores.append(safety)
            
            # Color tetrahedra by safety
            for i, (tet, safety) in enumerate(zip(self.tetrahedra[:500], 
                                                  safety_scores)):
                vertices = self.mesh.points[tet]
                centroid = self.centroids[i]
                
                # Color based on safety (red=dangerous, green=safe)
                color = plt.cm.RdYlGn(safety)
                ax1.scatter(*centroid, c=[color], s=20, alpha=0.6)
        
        # Plot path
        if path:
            path_array = np.array(path)
            ax1.plot(path_array[:,0], path_array[:,1], path_array[:,2],
                    'b-', linewidth=3, marker='o', markersize=6)
            ax1.scatter(*path[0], color='green', s=200, marker='o', 
                       label='Start')
            ax1.scatter(*path[-1], color='red', s=200, marker='*', 
                       label='Goal')
        
        # Plot obstacles
        for obstacle in self.obstacle_memory:
            ax1.scatter(*obstacle, color='black', s=300, marker='X', 
                       label='Obstacle')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('ML-Enhanced Path with Safety Scores')
        ax1.legend()
        
        # 2D cost heatmap (top view)
        ax2 = fig.add_subplot(122)
        
        if self.enable_ml and len(self.centroids) > 0:
            # Create cost grid
            x_min, x_max = self.centroids[:,0].min(), self.centroids[:,0].max()
            y_min, y_max = self.centroids[:,1].min(), self.centroids[:,1].max()
            
            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, 50),
                np.linspace(y_min, y_max, 50)
            )
            
            # Sample some costs
            costs = np.zeros_like(grid_x)
            for i in range(grid_x.shape[0]):
                for j in range(grid_x.shape[1]):
                    point = np.array([grid_x[i,j], grid_y[i,j], 
                                    np.mean(self.centroids[:,2])])
                    nearest = self.find_nearest_tetrahedron(point)
                    
                    # Average cost to neighbors
                    avg_cost = 0
                    neighbors = list(self.graph.neighbors(nearest))
                    if neighbors:
                        for neighbor in neighbors[:3]:
                            avg_cost += self.predict_traversal_cost(nearest, 
                                                                   neighbor)
                        costs[i,j] = avg_cost / min(3, len(neighbors))
            
            im = ax2.contourf(grid_x, grid_y, costs, levels=20, cmap='viridis')
            plt.colorbar(im, ax=ax2, label='Predicted Traversal Cost')
            
            # Plot path on heatmap
            if path:
                path_array = np.array(path)
                ax2.plot(path_array[:,0], path_array[:,1], 'r-', 
                        linewidth=2, label='Path')
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Predicted Traversal Costs (Top View)')
        
        plt.tight_layout()
        plt.show()
    
    # Include all the base methods from the original class
    def compute_tetrahedron_centroid(self, tet_idx: int) -> np.ndarray:
        vertices = self.mesh.points[self.tetrahedra[tet_idx]]
        return np.mean(vertices, axis=0)
    
    def are_tetrahedra_adjacent(self, tet1_idx: int, tet2_idx: int) -> bool:
        tet1_verts = set(self.tetrahedra[tet1_idx])
        tet2_verts = set(self.tetrahedra[tet2_idx])
        shared_vertices = len(tet1_verts.intersection(tet2_verts))
        return shared_vertices == 3
    

    def build_navigation_graph_new(self):
        """Build a navigation graph where nodes are tetrahedra centroids."""
        print("Building navigation graph...")
        
        # Compute centroids for all tetrahedra
        self.centroids = []
        for i in range(len(self.tetrahedra)):
            centroid = self.compute_tetrahedron_centroid(i)
            self.centroids.append(centroid)
            self.graph.add_node(i, pos=centroid)
        
        self.centroids = np.array(self.centroids)
        
        # Build KD-tree for efficient nearest neighbor queries
        self.kdtree = KDTree(self.centroids)
        
        print(f"Processing {len(self.tetrahedra)} tetrahedra...")
        
        # For very large meshes, use a simplified approach without multiprocessing
        if len(self.tetrahedra) > 100000:
            print("Large mesh detected, using optimized sequential processing...")
            
            edge_count = 0
            
            # Process each tetrahedron
            for i in range(len(self.tetrahedra)):
                if i % 10000 == 0:
                    print(f"  Processed {i}/{len(self.tetrahedra)} tetrahedra...")
                
                # Get vertices of tetrahedron i
                tet_i_verts = set(self.tetrahedra[i])
                
                # Estimate search radius
                vertices_i = self.mesh.points[self.tetrahedra[i]]
                max_edge_length = np.max([np.linalg.norm(vertices_i[j] - vertices_i[k]) 
                                        for j in range(4) for k in range(j+1, 4)])
                search_radius = max_edge_length * 2.0
                
                # Find nearby tetrahedra
                candidates = self.kdtree.query_ball_point(self.centroids[i], search_radius)
                
                # Check adjacency
                for j in candidates:
                    if j > i:  # Only check each pair once
                        tet_j_verts = set(self.tetrahedra[j])
                        shared_vertices = len(tet_i_verts.intersection(tet_j_verts))
                        
                        if shared_vertices == 3:  # Adjacent
                            dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                            self.graph.add_edge(i, j, weight=dist)
                            edge_count += 1
        
        else:
            # For smaller meshes, use parallel processing with a different approach
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import functools
            
            # Create a static function that can be pickled
            process_func = functools.partial(
                _process_tetrahedron_batch,
                tetrahedra=self.tetrahedra,
                centroids=self.centroids,
                points=self.mesh.points,
                kdtree_data=self.centroids  # We'll rebuild KDTree in each process
            )
            
            # Create batches
            batch_size = max(100, len(self.tetrahedra) // 20)
            batches = []
            for i in range(0, len(self.tetrahedra), batch_size):
                batches.append((i, min(i + batch_size, len(self.tetrahedra))))
            
            edge_count = 0
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_func, batch): batch 
                        for batch in batches}
                
                for future in as_completed(futures):
                    edges = future.result()
                    for i, j, dist in edges:
                        self.graph.add_edge(i, j, weight=dist)
                        edge_count += 1
                    
                    batch = futures[future]
                    print(f"  Completed batch {batch[0]}-{batch[1]}")
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and "
            f"{edge_count} edges")


##
    def build_navigation_graph_parallel_fast(self):
        """Ultra-fast parallel graph building using vertex-based adjacency."""
        print("Building navigation graph (parallel fast mode)...")
        
        # Compute centroids
        self.centroids = np.array([self.compute_tetrahedron_centroid(i) 
                                for i in range(len(self.tetrahedra))])
        
        self.kdtree = KDTree(self.centroids)
        
        # Add nodes
        for i in range(len(self.tetrahedra)):
            self.graph.add_node(i, pos=self.centroids[i])
        
        # Build vertex-to-tetrahedra mapping
        print("Building vertex mapping...")
        vertex_to_tets = {}
        for i, tet in enumerate(self.tetrahedra):
            for vertex in tet:
                vertex = int(vertex)  # Ensure it's an int
                if vertex not in vertex_to_tets:
                    vertex_to_tets[vertex] = []
                vertex_to_tets[vertex].append(i)
        
        # Parallel processing of vertices
        from multiprocessing import Pool, cpu_count
        import functools
        
        n_cores = min(cpu_count(), 8)
        vertex_items = list(vertex_to_tets.items())
        chunk_size = max(100, len(vertex_items) // (n_cores * 10))
        
        chunks = []
        for i in range(0, len(vertex_items), chunk_size):
            chunks.append(vertex_items[i:i + chunk_size])
        
        print(f"Processing {len(vertex_items)} vertices in {len(chunks)} chunks using {n_cores} cores...")
        
        # Process chunks in parallel
        process_func = functools.partial(_process_vertex_chunk, 
                                    tetrahedra=self.tetrahedra)
        
        edge_candidates = set()
        with Pool(processes=n_cores) as pool:
            for candidates in pool.imap(process_func, chunks):
                edge_candidates.update(candidates)
        
        print(f"Checking {len(edge_candidates)} candidate edges for adjacency...")
        
        # Check which pairs are actually adjacent (parallel)
        edge_list = list(edge_candidates)
        edge_chunks = []
        chunk_size = max(1000, len(edge_list) // (n_cores * 10))
        
        for i in range(0, len(edge_list), chunk_size):
            edge_chunks.append(edge_list[i:i + chunk_size])
        
        check_func = functools.partial(_check_adjacency_chunk,
                                    tetrahedra=self.tetrahedra,
                                    centroids=self.centroids)
        
        all_edges = []
        with Pool(processes=n_cores) as pool:
            for edges in pool.imap(check_func, edge_chunks):
                all_edges.extend(edges)
        
        # Add edges to graph
        edge_count = 0
        for i, j, dist in all_edges:
            self.graph.add_edge(i, j, weight=dist)
            edge_count += 1
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and "
            f"{edge_count} edges")


    # Module-level functions for parallel processing
    def _process_vertex_chunk(vertex_chunk, tetrahedra):
        """Process a chunk of vertices to find potentially adjacent tetrahedra."""
        candidates = set()
        
        for vertex, tet_list in vertex_chunk:
            # All pairs of tetrahedra sharing this vertex
            for i in range(len(tet_list)):
                for j in range(i + 1, len(tet_list)):
                    tet_i, tet_j = tet_list[i], tet_list[j]
                    if tet_i < tet_j:
                        candidates.add((tet_i, tet_j))
                    else:
                        candidates.add((tet_j, tet_i))
        
        return candidates


    def _check_adjacency_chunk(edge_chunk, tetrahedra, centroids):
        """Check which candidate edges are actual adjacencies."""
        import numpy as np
        
        edges = []
        for i, j in edge_chunk:
            shared = len(set(tetrahedra[i]).intersection(set(tetrahedra[j])))
            if shared == 3:  # Face-adjacent
                dist = np.linalg.norm(centroids[i] - centroids[j])
                edges.append((i, j, dist))
        
        return edges
    ##
# -
    def build_navigation_graph_parallel_optimal(self):
        """Build a navigation graph where nodes are tetrahedra centroids."""
        print("Building navigation graph with optimal parallel...")
        
        # Compute centroids for all tetrahedra
        self.centroids = []
        for i in range(len(self.tetrahedra)):
            centroid = self.compute_tetrahedron_centroid(i)
            self.centroids.append(centroid)
            self.graph.add_node(i, pos=centroid)
        
        self.centroids = np.array(self.centroids)
    
        # Build KD-tree for efficient nearest neighbor queries
        self.kdtree = KDTree(self.centroids)
        
        print(f"Processing {len(self.tetrahedra)} tetrahedra...")
        
        # Use spatial indexing to only check nearby tetrahedra
        from multiprocessing import Pool, cpu_count
        
        def process_tetrahedron(i):
            """Find adjacent tetrahedra for a single tetrahedron."""
            # Get the vertices of tetrahedron i
            tet_i_verts = set(self.tetrahedra[i])
            
            # Find nearby tetrahedra using KD-tree
            # Search radius based on typical tetrahedron size
            centroid_i = self.centroids[i]
            
            # Estimate search radius (adjust based on your mesh)
            vertices_i = self.mesh.points[self.tetrahedra[i]]
            max_edge_length = np.max([np.linalg.norm(vertices_i[j] - vertices_i[k]) 
                                    for j in range(4) for k in range(j+1, 4)])
            search_radius = max_edge_length * 2.0
            
            # Find candidates within radius
            candidates = self.kdtree.query_ball_point(centroid_i, search_radius)
            
            adjacent_edges = []
            for j in candidates:
                if j > i:  # Only check each pair once
                    tet_j_verts = set(self.tetrahedra[j])
                    shared_vertices = len(tet_i_verts.intersection(tet_j_verts))
                    
                    if shared_vertices == 3:  # Adjacent (sharing a face)
                        dist = np.linalg.norm(centroid_i - self.centroids[j])
                        adjacent_edges.append((i, j, dist))
            
            return adjacent_edges
        
        # Process in parallel with progress tracking
        n_cores = cpu_count()
        chunk_size = max(100, len(self.tetrahedra) // (n_cores * 10))
        
        edge_count = 0
        
        # Process in batches to avoid memory issues
        for batch_start in range(0, len(self.tetrahedra), chunk_size * n_cores):
            batch_end = min(batch_start + chunk_size * n_cores, len(self.tetrahedra))
            batch_indices = list(range(batch_start, batch_end))
            
            if batch_start % (chunk_size * n_cores * 10) == 0:
                print(f"  Processed {batch_start}/{len(self.tetrahedra)} tetrahedra...")
            
            with Pool(processes=n_cores) as pool:
                batch_results = pool.map(process_tetrahedron, batch_indices)
            
            # Add edges from this batch
            for edges in batch_results:
                for i, j, dist in edges:
                    self.graph.add_edge(i, j, weight=dist)
                    edge_count += 1
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and "
            f"{edge_count} edges")
# -

    #----
    def build_navigation_graph_parallel(self):
    #Build a navigation graph where nodes are tetrahedra centroids.
        print("Building navigation graph in parallel...")
    
    # Compute centroids for all tetrahedra
        self.centroids = []
        for i in range(len(self.tetrahedra)):
            print(f"Processing tetrahedron {i+1}/{len(self.tetrahedra)}")
            centroid = self.compute_tetrahedron_centroid(i)
            self.centroids.append(centroid)
            self.graph.add_node(i, pos=centroid)
    
        self.centroids = np.array(self.centroids)
    
        # Build KD-tree for efficient nearest neighbor queries
        self.kdtree = KDTree(self.centroids)
    
        # Parallel adjacency checking
        from multiprocessing import Pool, cpu_count
        import itertools
    
        # Generate all pairs to check
        n_tets = len(self.tetrahedra)
        pairs = [(i, j) for i in range(n_tets) for j in range(i + 1, n_tets)]
    
        #Function to check adjacency for a batch of pairs
        def check_adjacency_batch(pair_batch):
            adjacent_pairs = []
            for i, j in pair_batch:
                tet1_verts = set(self.tetrahedra[i])
                tet2_verts = set(self.tetrahedra[j])
                shared_vertices = len(tet1_verts.intersection(tet2_verts))
                if shared_vertices == 3:  # Adjacent if sharing a face (3 vertices)
                    dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    adjacent_pairs.append((i, j, dist))
            return adjacent_pairs
    
        # Split pairs into chunks for parallel processing
        n_cores = cpu_count()
        chunk_size = max(1, len(pairs) // (n_cores * 10))  # 10 chunks per core
        pair_chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    
        print(f"Checking {len(pairs)} potential edges using {n_cores} cores...")
    
        # Process in parallel
        with Pool(processes=n_cores) as pool:
            results = pool.map(check_adjacency_batch, pair_chunks)
    
        # Add edges to graph
        edge_count = 0
        for batch_result in results:
            for i, j, dist in batch_result:
                self.graph.add_edge(i, j, weight=dist)
                edge_count += 1
    
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and "
            f"{edge_count} edges")
    #----

    
    def find_nearest_tetrahedron(self, point: np.ndarray) -> int:
        _, idx = self.kdtree.query(point)
        return idx
    
    def point_in_tetrahedron(self, point: np.ndarray, tet_idx: int) -> bool:
        vertices = self.mesh.points[self.tetrahedra[tet_idx]]
        
        v0 = vertices[0]
        v1 = vertices[1] - v0
        v2 = vertices[2] - v0
        v3 = vertices[3] - v0
        p = point - v0
        
        mat = np.column_stack([v1, v2, v3])
        try:
            bary = np.linalg.solve(mat, p)
            return (bary >= 0).all() and (bary.sum() <= 1)
        except:
            return False
    
    def find_containing_tetrahedron(self, point: np.ndarray) -> Optional[int]:
        nearest_idx = self.find_nearest_tetrahedron(point)
        
        if self.point_in_tetrahedron(point, nearest_idx):
            return nearest_idx
        
        neighbors = list(self.graph.neighbors(nearest_idx))
        for neighbor in neighbors:
            if self.point_in_tetrahedron(point, neighbor):
                return neighbor
        
        return nearest_idx
    
    def retrain_models(self):
        """Retrain ML models with accumulated path history."""
        print("Retraining ML models...")
        # Implementation would process path_history to extract training data
        # and retrain the models
        pass


# Example usage
if __name__ == "__main__":
    # Create sample mesh
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        [0.5, 0.5, 0.5]
    ])
    
    cells = [
        ("tetra", np.array([
            [0, 1, 2, 8], [1, 3, 2, 8], [0, 2, 4, 8], [2, 6, 4, 8],
            [1, 5, 3, 8], [3, 5, 7, 8], [2, 3, 6, 8], [3, 7, 6, 8],
            [0, 1, 4, 8], [1, 4, 5, 8], [4, 5, 6, 8], [5, 6, 7, 8]
        ]))
    ]
    
    mesh = meshio.Mesh(points, cells)
    meshio.write("sample_city.vtu", mesh)
    
    # Initialize ML-enhanced pathfinder
    pathfinder = MLEnhancedPathfinder("sample_city.vtu", enable_ml=True)
    
    # Add some obstacles
    pathfinder.add_obstacle(np.array([0.3, 0.3, 0.5]))
    pathfinder.add_obstacle(np.array([0.7, 0.7, 0.5]))
    
    # Find ML-optimized path
    start = np.array([0.1, 0.1, 0.1])
    goal = np.array([0.9, 0.9, 0.9])
    
    
    # Compare standard and ML paths
    print("\nFinding ML-enhanced path...")
    ml_path = pathfinder.find_path_ml(start, goal, safety_weight=0.5)
    
    if ml_path:
        print(f"ML path found with {len(ml_path)} waypoints")
        
        # Visualize with ML insights
        pathfinder.visualize_ml_insights(ml_path)
        
        # Save trained models
        pathfinder.save_ml_models("drone_pathfinder_models.pkl")
    else:
        print("No path found!")