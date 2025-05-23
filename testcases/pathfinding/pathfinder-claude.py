import numpy as np
import meshio
import networkx as nx
from scipy.spatial import KDTree
import heapq
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TetrahedralPathfinder:
    """
    A pathfinding system for drones navigating through a city represented
    as a tetrahedral mesh in VTU format.
    """
    
    def __init__(self, vtu_file: str):
        """
        Initialize the pathfinder with a VTU mesh file.
        
        Args:
            vtu_file: Path to the VTU file containing the tetrahedral mesh
        """
        self.mesh = None
        self.graph = nx.Graph()
        self.centroids = []
        self.kdtree = None
        self.tetrahedra = []
        self.load_mesh(vtu_file)
        self.build_navigation_graph()
    
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
    
    def compute_tetrahedron_centroid(self, tet_idx: int) -> np.ndarray:
        """
        Compute the centroid of a tetrahedron.
        
        Args:
            tet_idx: Index of the tetrahedron
            
        Returns:
            3D coordinates of the centroid
        """
        vertices = self.mesh.points[self.tetrahedra[tet_idx]]
        return np.mean(vertices, axis=0)
    
    def are_tetrahedra_adjacent(self, tet1_idx: int, tet2_idx: int) -> bool:
        """
        Check if two tetrahedra share a face (3 vertices).
        
        Args:
            tet1_idx: Index of first tetrahedron
            tet2_idx: Index of second tetrahedron
            
        Returns:
            True if tetrahedra are adjacent
        """
        tet1_verts = set(self.tetrahedra[tet1_idx])
        tet2_verts = set(self.tetrahedra[tet2_idx])
        shared_vertices = len(tet1_verts.intersection(tet2_verts))
        return shared_vertices == 3
    
    def build_navigation_graph(self):
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
        
        # Connect adjacent tetrahedra
        for i in range(len(self.tetrahedra)):
            for j in range(i + 1, len(self.tetrahedra)):
                if self.are_tetrahedra_adjacent(i, j):
                    # Weight edges by Euclidean distance between centroids
                    dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    self.graph.add_edge(i, j, weight=dist)
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and "
              f"{self.graph.number_of_edges()} edges")
    
    def find_nearest_tetrahedron(self, point: np.ndarray) -> int:
        """
        Find the nearest tetrahedron to a given point.
        
        Args:
            point: 3D coordinates
            
        Returns:
            Index of nearest tetrahedron
        """
        _, idx = self.kdtree.query(point)
        return idx
    
    def point_in_tetrahedron(self, point: np.ndarray, tet_idx: int) -> bool:
        """
        Check if a point is inside a tetrahedron using barycentric coordinates.
        
        Args:
            point: 3D coordinates
            tet_idx: Index of tetrahedron
            
        Returns:
            True if point is inside tetrahedron
        """
        vertices = self.mesh.points[self.tetrahedra[tet_idx]]
        
        # Compute barycentric coordinates
        v0 = vertices[0]
        v1 = vertices[1] - v0
        v2 = vertices[2] - v0
        v3 = vertices[3] - v0
        p = point - v0
        
        # Solve the linear system
        mat = np.column_stack([v1, v2, v3])
        try:
            bary = np.linalg.solve(mat, p)
            
            # Check if all barycentric coordinates are non-negative
            # and their sum is <= 1
            return (bary >= 0).all() and (bary.sum() <= 1)
        except:
            return False
    
    def find_containing_tetrahedron(self, point: np.ndarray) -> Optional[int]:
        """
        Find the tetrahedron containing a given point.
        
        Args:
            point: 3D coordinates
            
        Returns:
            Index of containing tetrahedron or None
        """
        # Start with nearest tetrahedron
        nearest_idx = self.find_nearest_tetrahedron(point)
        
        if self.point_in_tetrahedron(point, nearest_idx):
            return nearest_idx
        
        # Check neighboring tetrahedra
        neighbors = list(self.graph.neighbors(nearest_idx))
        for neighbor in neighbors:
            if self.point_in_tetrahedron(point, neighbor):
                return neighbor
        
        # Fallback: return nearest
        return nearest_idx
    
    def find_path(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Find a path from start to goal position.
        
        Args:
            start: 3D start coordinates
            goal: 3D goal coordinates
            
        Returns:
            List of waypoints (3D coordinates) or None if no path exists
        """
        # Find start and goal tetrahedra
        start_tet = self.find_containing_tetrahedron(start)
        goal_tet = self.find_containing_tetrahedron(goal)
        
        if start_tet is None or goal_tet is None:
            print("Start or goal position outside mesh")
            return None
        
        try:
            # Find shortest path through tetrahedra
            tet_path = nx.shortest_path(self.graph, start_tet, goal_tet, 
                                        weight='weight')
            
            # Convert to waypoints
            waypoints = [start]
            for tet_idx in tet_path[1:-1]:
                waypoints.append(self.centroids[tet_idx])
            waypoints.append(goal)
            
            return waypoints
            
        except nx.NetworkXNoPath:
            print("No path found between start and goal")
            return None
    
    def smooth_path(self, waypoints: List[np.ndarray], 
                    iterations: int = 5) -> List[np.ndarray]:
        """
        Smooth a path using simple averaging.
        
        Args:
            waypoints: List of 3D waypoints
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed waypoints
        """
        if len(waypoints) <= 2:
            return waypoints
        
        smoothed = waypoints.copy()
        
        for _ in range(iterations):
            new_waypoints = [smoothed[0]]  # Keep start
            
            for i in range(1, len(smoothed) - 1):
                # Average with neighbors
                avg = (smoothed[i-1] + smoothed[i] + smoothed[i+1]) / 3
                new_waypoints.append(avg)
            
            new_waypoints.append(smoothed[-1])  # Keep goal
            smoothed = new_waypoints
        
        return smoothed
    
    def visualize_path(self, waypoints: Optional[List[np.ndarray]] = None):
        """
        Visualize the mesh and path.
        
        Args:
            waypoints: Optional path to visualize
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot tetrahedra edges
        plotted_edges = set()
        for tet in self.tetrahedra[:1000]:  # Limit for performance
            vertices = self.mesh.points[tet]
            
            # Plot edges of tetrahedron
            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            for i, j in edges:
                edge = tuple(sorted([tet[i], tet[j]]))
                if edge not in plotted_edges:
                    plotted_edges.add(edge)
                    ax.plot([vertices[i,0], vertices[j,0]],
                           [vertices[i,1], vertices[j,1]],
                           [vertices[i,2], vertices[j,2]],
                           'b-', alpha=0.1, linewidth=0.5)
        
        # Plot path
        if waypoints:
            waypoints_array = np.array(waypoints)
            ax.plot(waypoints_array[:,0], 
                   waypoints_array[:,1], 
                   waypoints_array[:,2],
                   'r-', linewidth=3, marker='o', markersize=8)
            
            # Mark start and goal
            ax.scatter(*waypoints[0], color='green', s=100, marker='o', 
                      label='Start')
            ax.scatter(*waypoints[-1], color='red', s=100, marker='*', 
                      label='Goal')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Path Through Tetrahedral Mesh')
        ax.legend()
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example: Create a simple test mesh (you would load your actual VTU file)
    # For demonstration, let's create a simple cubic mesh
    
    # Create sample points for a small mesh
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        [0.5, 0.5, 0.5]  # Center point
    ])
    
    # Define tetrahedra (indices into points array)
    cells = [
        ("tetra", np.array([
            [0, 1, 2, 8],
            [1, 3, 2, 8],
            [0, 2, 4, 8],
            [2, 6, 4, 8],
            [1, 5, 3, 8],
            [3, 5, 7, 8],
            [2, 3, 6, 8],
            [3, 7, 6, 8],
            [0, 1, 4, 8],
            [1, 4, 5, 8],
            [4, 5, 6, 8],
            [5, 6, 7, 8]
        ]))
    ]
    
    # Save as VTU file
    mesh = meshio.Mesh(points, cells)
    meshio.write("sample_city.vtu", mesh)
    
    # Initialize pathfinder
    pathfinder = TetrahedralPathfinder("sample_city.vtu")
    
    # Find path
    start = np.array([0.1, 0.1, 0.1])
    goal = np.array([0.9, 0.9, 0.9])
    
    path = pathfinder.find_path(start, goal)
    
    if path:
        print(f"\nPath found with {len(path)} waypoints:")
        for i, wp in enumerate(path):
            print(f"  Waypoint {i}: {wp}")
        
        # Smooth the path
        smoothed_path = pathfinder.smooth_path(path)
        print(f"\nSmoothed path has {len(smoothed_path)} waypoints")
        
        # Visualize
        pathfinder.visualize_path(smoothed_path)
    else:
        print("No path found!")