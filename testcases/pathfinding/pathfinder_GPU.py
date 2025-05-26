import numpy as np
import cupy as cp  # GPU arrays
import cupyx.scipy.spatial as spatial_gpu
from numba import cuda
import torch
import torch.nn as nn

class GPUEnhancedPathfinder:
    """GPU-accelerated pathfinding for large tetrahedral meshes."""
    
    def __init__(self, vtu_file: str, enable_ml: bool = True, 
                 subdomain: Optional[Dict[str, Tuple[float, float]]] = None,
                 use_gpu: bool = True):
        """
        Initialize pathfinder with optional GPU acceleration.
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires CUDA)
        """
        self.use_gpu = use_gpu and cuda.is_available()
        
        if self.use_gpu:
            print(f"GPU acceleration enabled on {cuda.get_current_device().name}")
            self.init_gpu_structures()
        else:
            print("GPU not available, using CPU")
        
        # Rest of initialization...
        self.mesh = None
        self.graph = nx.Graph()
        self.load_mesh(vtu_file)
        
        if subdomain:
            self.filter_mesh_to_subdomain()
        
        if self.use_gpu:
            self.build_navigation_graph_gpu()
        else:
            self.build_navigation_graph()
    
    def init_gpu_structures(self):
        """Initialize GPU memory pools and structures."""
        # Set memory pool for better allocation
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # Pre-allocate for better performance
        mempool.set_limit(size=4 * 1024**3)  # 4GB limit
    
    def build_navigation_graph_gpu(self):
        """Build navigation graph using GPU acceleration."""
        print("Building navigation graph on GPU...")
        
        # Compute centroids on GPU
        centroids_cpu = np.array([self.compute_tetrahedron_centroid(i) 
                                 for i in range(len(self.tetrahedra))])
        self.centroids = centroids_cpu
        
        # Transfer to GPU
        centroids_gpu = cp.asarray(centroids_cpu)
        tetrahedra_gpu = cp.asarray(self.tetrahedra)
        points_gpu = cp.asarray(self.mesh.points)
        
        # Build KD-tree on GPU (if available in your CUDA toolkit)
        # Note: CuPy doesn't have KDTree yet, so we'll use a custom implementation
        self.gpu_spatial_index = GPUSpatialIndex(centroids_gpu)
        
        # Keep CPU KDTree for compatibility
        from scipy.spatial import KDTree
        self.kdtree = KDTree(centroids_cpu)
        
        # Add nodes
        for i in range(len(self.tetrahedra)):
            self.graph.add_node(i, pos=self.centroids[i])
        
        # GPU kernel for adjacency checking
        edges_gpu = self._find_adjacent_tetrahedra_gpu(
            tetrahedra_gpu, centroids_gpu, points_gpu
        )
        
        # Transfer back and add edges
        edges_cpu = cp.asnumpy(edges_gpu)
        edge_count = 0
        
        for i in range(len(edges_cpu)):
            if edges_cpu[i, 2] > 0:  # Valid edge
                self.graph.add_edge(
                    int(edges_cpu[i, 0]), 
                    int(edges_cpu[i, 1]), 
                    weight=float(edges_cpu[i, 2])
                )
                edge_count += 1
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and "
              f"{edge_count} edges")
    
    @cuda.jit
    def _adjacency_kernel(tetrahedra, centroids, adjacency_matrix, n_tets):
        """CUDA kernel for checking tetrahedra adjacency."""
        i = cuda.grid(1)
        
        if i >= n_tets:
            return
        
        # Get vertices of tetrahedron i
        v0, v1, v2, v3 = tetrahedra[i]
        tet_i_verts = set([v0, v1, v2, v3])
        
        # Check against all other tetrahedra
        for j in range(i + 1, n_tets):
            # Get vertices of tetrahedron j
            w0, w1, w2, w3 = tetrahedra[j]
            
            # Count shared vertices
            shared = 0
            for v in [v0, v1, v2, v3]:
                if v == w0 or v == w1 or v == w2 or v == w3:
                    shared += 1
            
            # If sharing 3 vertices (a face), they're adjacent
            if shared == 3:
                # Calculate distance
                dx = centroids[i, 0] - centroids[j, 0]
                dy = centroids[i, 1] - centroids[j, 1]
                dz = centroids[i, 2] - centroids[j, 2]
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                
                adjacency_matrix[i, j] = dist
                adjacency_matrix[j, i] = dist
    
    def _find_adjacent_tetrahedra_gpu(self, tetrahedra_gpu, centroids_gpu, points_gpu):
        """Find adjacent tetrahedra using GPU acceleration."""
        n_tets = len(tetrahedra_gpu)
        
        # Method 1: Using sparse matrix on GPU
        if n_tets < 50000:  # For smaller meshes
            # Allocate adjacency matrix on GPU
            adjacency_gpu = cp.zeros((n_tets, n_tets), dtype=cp.float32)
            
            # Launch kernel
            threads_per_block = 256
            blocks_per_grid = (n_tets + threads_per_block - 1) // threads_per_block
            
            self._adjacency_kernel[blocks_per_grid, threads_per_block](
                tetrahedra_gpu, centroids_gpu, adjacency_gpu, n_tets
            )
            
            # Extract edges
            indices = cp.where(adjacency_gpu > 0)
            edges = cp.stack([
                indices[0], 
                indices[1], 
                adjacency_gpu[indices]
            ], axis=1)
            
            return edges
        
        else:  # For larger meshes, use spatial partitioning
            return self._find_adjacent_spatial_gpu(tetrahedra_gpu, centroids_gpu)
    
    def _find_adjacent_spatial_gpu(self, tetrahedra_gpu, centroids_gpu):
        """Use spatial partitioning for large meshes on GPU."""
        # Implement grid-based spatial partitioning
        print("Using GPU spatial partitioning for large mesh...")
        
        # Create spatial grid
        min_coords = cp.min(centroids_gpu, axis=0)
        max_coords = cp.max(centroids_gpu, axis=0)
        
        # Grid resolution
        grid_size = 50
        cell_size = (max_coords - min_coords) / grid_size
        
        # Assign tetrahedra to grid cells
        grid_indices = ((centroids_gpu - min_coords) / cell_size).astype(cp.int32)
        grid_indices = cp.clip(grid_indices, 0, grid_size - 1)
        
        # Build cell lists
        cell_lists = {}
        for i in range(len(tetrahedra_gpu)):
            cell = tuple(cp.asnumpy(grid_indices[i]))
            if cell not in cell_lists:
                cell_lists[cell] = []
            cell_lists[cell].append(i)
        
        # Check adjacency only within neighboring cells
        edges = []
        
        for cell, tet_list in cell_lists.items():
            # Check within cell and neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_cell = (
                            cell[0] + dx, 
                            cell[1] + dy, 
                            cell[2] + dz
                        )
                        
                        if neighbor_cell in cell_lists:
                            # Check pairs
                            for i in tet_list:
                                for j in cell_lists[neighbor_cell]:
                                    if i < j:
                                        # Check adjacency
                                        shared = len(
                                            set(cp.asnumpy(tetrahedra_gpu[i])) & 
                                            set(cp.asnumpy(tetrahedra_gpu[j]))
                                        )
                                        
                                        if shared == 3:
                                            dist = cp.linalg.norm(
                                                centroids_gpu[i] - centroids_gpu[j]
                                            )
                                            edges.append([i, j, float(dist)])
        
        return cp.array(edges)
    
    def find_path_gpu(self, start: np.ndarray, goal: np.ndarray, 
                      safety_weight: float = 0.3) -> Optional[List[np.ndarray]]:
        """GPU-accelerated pathfinding using parallel Dijkstra."""
        if not self.use_gpu:
            return self.find_path_ml(start, goal, safety_weight)
        
        start_tet = self.find_containing_tetrahedron(start)
        goal_tet = self.find_containing_tetrahedron(goal)
        
        if start_tet is None or goal_tet is None:
            return None
        
        # Convert graph to GPU format
        n_nodes = self.graph.number_of_nodes()
        edges_list = []
        
        for u, v, data in self.graph.edges(data=True):
            edges_list.append([u, v, data['weight']])
        
        edges_gpu = cp.asarray(edges_list)
        
        # Run GPU Dijkstra
        distances, predecessors = self._dijkstra_gpu(
            edges_gpu, n_nodes, start_tet, goal_tet
        )
        
        # Reconstruct path
        if distances[goal_tet] == cp.inf:
            return None
        
        path_indices = []
        current = goal_tet
        
        while current != start_tet:
            path_indices.append(current)
            current = int(predecessors[current])
            if current == -1:
                return None
        
        path_indices.append(start_tet)
        path_indices.reverse()
        
        # Convert to waypoints
        waypoints = [start]
        for idx in path_indices[1:-1]:
            waypoints.append(self.centroids[idx])
        waypoints.append(goal)
        
        return waypoints
    
    @cuda.jit
    def _dijkstra_kernel(edges, distances, predecessors, updated, n_edges):
        """CUDA kernel for parallel Dijkstra relaxation."""
        idx = cuda.grid(1)
        
        if idx >= n_edges:
            return
        
        u, v, weight = edges[idx]
        u, v = int(u), int(v)
        
        # Relax edge u->v
        new_dist = distances[u] + weight
        if new_dist < distances[v]:
            distances[v] = new_dist
            predecessors[v] = u
            updated[0] = 1
        
        # Relax edge v->u (undirected)
        new_dist = distances[v] + weight
        if new_dist < distances[u]:
            distances[u] = new_dist
            predecessors[u] = v
            updated[0] = 1
    
    def _dijkstra_gpu(self, edges_gpu, n_nodes, start, goal):
        """Run Dijkstra's algorithm on GPU."""
        # Initialize distances and predecessors
        distances = cp.full(n_nodes, cp.inf, dtype=cp.float32)
        distances[start] = 0
        predecessors = cp.full(n_nodes, -1, dtype=cp.int32)
        
        # Bellman-Ford style relaxation on GPU
        n_edges = len(edges_gpu)
        threads_per_block = 256
        blocks_per_grid = (n_edges + threads_per_block - 1) // threads_per_block
        
        for iteration in range(n_nodes):
            updated = cp.zeros(1, dtype=cp.int32)
            
            self._dijkstra_kernel[blocks_per_grid, threads_per_block](
                edges_gpu, distances, predecessors, updated, n_edges
            )
            
            if updated[0] == 0:
                break
        
        return distances, predecessors


class GPUSpatialIndex:
    """Custom GPU-based spatial index for fast nearest neighbor queries."""
    
    def __init__(self, points_gpu):
        self.points = points_gpu
        self.n_points = len(points_gpu)
        
        # Build grid index
        self.build_grid_index()
    
    def build_grid_index(self):
        """Build spatial grid for fast lookups."""
        self.min_coords = cp.min(self.points, axis=0)
        self.max_coords = cp.max(self.points, axis=0)
        
        # Determine grid size
        self.grid_size = int(np.cbrt(self.n_points))
        self.cell_size = (self.max_coords - self.min_coords) / self.grid_size
        
        # Assign points to grid cells
        self.grid_indices = ((self.points - self.min_coords) / self.cell_size).astype(cp.int32)
        self.grid_indices = cp.clip(self.grid_indices, 0, self.grid_size - 1)
    
    @cuda.jit
    def _nearest_neighbor_kernel(points, query, result_idx, result_dist, n_points):
        """CUDA kernel for nearest neighbor search."""
        idx = cuda.grid(1)
        
        if idx >= n_points:
            return
        
        # Calculate distance
        dx = points[idx, 0] - query[0]
        dy = points[idx, 1] - query[1]
        dz = points[idx, 2] - query[2]
        dist = dx*dx + dy*dy + dz*dz
        
        # Atomic min operation
        cuda.atomic.min(result_dist, 0, dist)
        if dist == result_dist[0]:
            result_idx[0] = idx
    
    def query(self, point):
        """Find nearest neighbor on GPU."""
        query_gpu = cp.asarray(point)
        result_idx = cp.array([-1], dtype=cp.int32)
        result_dist = cp.array([cp.inf], dtype=cp.float32)
        
        threads_per_block = 256
        blocks_per_grid = (self.n_points + threads_per_block - 1) // threads_per_block
        
        self._nearest_neighbor_kernel[blocks_per_grid, threads_per_block](
            self.points, query_gpu, result_idx, result_dist, self.n_points
        )
        
        return float(cp.sqrt(result_dist[0])), int(result_idx[0])


# PyTorch-based ML models for GPU
class GPUMLModels:
    """GPU-accelerated ML models using PyTorch."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Cost prediction model
        self.cost_model = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(self.device)
        
        # Safety prediction model
        self.safety_model = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def predict_batch(self, features, model_type='cost'):
        """Batch prediction on GPU."""
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            if model_type == 'cost':
                predictions = self.cost_model(features_tensor)
            else:
                predictions = self.safety_model(features_tensor)
        
        return predictions.cpu().numpy()