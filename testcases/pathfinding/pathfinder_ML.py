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
        self.build_navigation_graph()
        
        if self.enable_ml:
            self.initialize_ml_models()
    
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
    
    def build_navigation_graph(self):
        print("Building navigation graph...")
        
        self.centroids = []
        for i in range(len(self.tetrahedra)):
            centroid = self.compute_tetrahedron_centroid(i)
            self.centroids.append(centroid)
            self.graph.add_node(i, pos=centroid)
        
        self.centroids = np.array(self.centroids)
        self.kdtree = KDTree(self.centroids)
        
        for i in range(len(self.tetrahedra)):
            for j in range(i + 1, len(self.tetrahedra)):
                if self.are_tetrahedra_adjacent(i, j):
                    dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    self.graph.add_edge(i, j, weight=dist)
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and "
              f"{self.graph.number_of_edges()} edges")
    
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