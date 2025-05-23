import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio
from matplotlib.colors import Normalize
import matplotlib.cm as cm



# First, let's create a sample city mesh for demonstration
def create_sample_city_mesh():
    """Create a sample city mesh with obstacles (buildings)."""
    # Create a grid of points
    x = np.linspace(0, 100, 6)
    y = np.linspace(0, 100, 6)
    z = np.linspace(0, 50, 4)
    
    points = []
    for xi in x:
        for yi in y:
            for zi in z:
                # Skip some points to create "buildings" (obstacles)
                if (30 < xi < 70 and 30 < yi < 70 and zi < 20):
                    continue  # This creates a building in the center
                points.append([xi, yi, zi])
    
    points = np.array(points)
    
    # Create tetrahedra using Delaunay triangulation (simplified)
    # In practice, you'd use a proper meshing tool
    from scipy.spatial import Delaunay
    
    # Project to 2D for triangulation, then extend to 3D
    tri = Delaunay(points[:, :2])
    
    # Convert triangles to tetrahedra (simplified approach)
    tetrahedra = []
    for simplex in tri.simplices[:100]:  # Limit for demo
        if len(simplex) == 3:
            # Find a 4th point at different z
            p1, p2, p3 = simplex
            z_avg = (points[p1, 2] + points[p2, 2] + points[p3, 2]) / 3
            
            # Find a point at different z level
            for i, p in enumerate(points):
                if i not in simplex and abs(p[2] - z_avg) > 5:
                    tetrahedra.append([p1, p2, p3, i])
                    break
    
    cells = [("tetra", np.array(tetrahedra[:50]))]  # Limit for visibility
    mesh = meshio.Mesh(points, cells)
    meshio.write("city_with_buildings.vtu", mesh)
    
    return mesh, np.array(tetrahedra[:50])

# Complete visualization function
def visualize_mesh_and_path(pathfinder, path, save_fig=False):
    """
    Create a comprehensive visualization of the mesh and path.
    
    Args:
        pathfinder: The pathfinder object with mesh data
        path: List of waypoints
        save_fig: Whether to save the figure
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Main 3D view with full mesh and path
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_full_mesh_and_path(ax1, pathfinder, path, "Full Mesh with Path")
    
    # 2. Simplified mesh view (only tetrahedra near path)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_path_corridor(ax2, pathfinder, path, "Path Corridor View")
    
    # 3. Wireframe view
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_wireframe_view(ax3, pathfinder, path, "Wireframe View")
    
    # 4. Top-down view
    ax4 = fig.add_subplot(2, 3, 4)
    plot_top_view(ax4, pathfinder, path, "Top View (2D)")
    
    # 5. Side view
    ax5 = fig.add_subplot(2, 3, 5)
    plot_side_view(ax5, pathfinder, path, "Side View (2D)")
    
    # 6. Interactive view with safety coloring
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_safety_view(ax6, pathfinder, path, "Safety Analysis View")
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('mesh_path_visualization.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_full_mesh_and_path(ax, pathfinder, path, title):
    """Plot the complete mesh with the path."""
    
    # Plot all tetrahedra edges (with transparency)
    plotted_edges = set()
    
    for i, tet in enumerate(pathfinder.tetrahedra):
        vertices = pathfinder.mesh.points[tet]
        
        # Define edges of tetrahedron
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        
        for e1, e2 in edges:
            # Create unique edge identifier
            edge_key = tuple(sorted([tet[e1], tet[e2]]))
            
            if edge_key not in plotted_edges:
                plotted_edges.add(edge_key)
                
                # Plot edge
                ax.plot([vertices[e1, 0], vertices[e2, 0]],
                       [vertices[e1, 1], vertices[e2, 1]],
                       [vertices[e1, 2], vertices[e2, 2]],
                       'gray', alpha=0.2, linewidth=0.5)
    
    # Plot tetrahedra centroids
    if len(pathfinder.centroids) < 1000:  # Only if not too many
        ax.scatter(pathfinder.centroids[:, 0], 
                  pathfinder.centroids[:, 1], 
                  pathfinder.centroids[:, 2],
                  c='lightblue', s=10, alpha=0.5)
    
    # Plot the path
    if path:
        path_array = np.array(path)
        
        # Main path line
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                'b-', linewidth=4, label='Flight Path', zorder=100)
        
        # Waypoints
        ax.scatter(path_array[1:-1, 0], 
                  path_array[1:-1, 1], 
                  path_array[1:-1, 2],
                  c='yellow', s=100, edgecolors='black', 
                  linewidth=1, zorder=101)
        
        # Start and Goal
        ax.scatter(*path[0], color='green', s=300, marker='o', 
                  edgecolors='black', linewidth=2, label='Start', zorder=102)
        ax.scatter(*path[-1], color='red', s=300, marker='*', 
                  edgecolors='black', linewidth=2, label='Goal', zorder=102)
    
    # Add obstacles if any
    for obstacle in pathfinder.obstacle_memory:
        ax.scatter(*obstacle, color='black', s=500, marker='X', 
                  alpha=0.8, label='Obstacle')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

def plot_path_corridor(ax, pathfinder, path, title):
    """Plot only tetrahedra near the path for clarity."""
    
    if not path:
        return
    
    # Find tetrahedra along the path
    path_tets = set()
    for waypoint in path:
        tet_idx = pathfinder.find_containing_tetrahedron(waypoint)
        if tet_idx is not None:
            path_tets.add(tet_idx)
            # Add neighbors too
            for neighbor in pathfinder.graph.neighbors(tet_idx):
                path_tets.add(neighbor)
    
    # Plot only these tetrahedra
    for tet_idx in path_tets:
        tet = pathfinder.tetrahedra[tet_idx]
        vertices = pathfinder.mesh.points[tet]
        
        # Plot faces with transparency
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        faces = [
            [vertices[0], vertices[1], vertices[2]],
            [vertices[0], vertices[1], vertices[3]],
            [vertices[0], vertices[2], vertices[3]],
            [vertices[1], vertices[2], vertices[3]]
        ]
        
        poly = Poly3DCollection(faces, alpha=0.2, facecolor='cyan', 
                               edgecolor='darkblue', linewidth=0.5)
        ax.add_collection3d(poly)
    
    # Plot the path
    path_array = np.array(path)
    ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
            'r-', linewidth=4, label='Flight Path', zorder=100)
    
    ax.scatter(*path[0], color='green', s=300, marker='o', label='Start')
    ax.scatter(*path[-1], color='red', s=300, marker='*', label='Goal')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

def plot_wireframe_view(ax, pathfinder, path, title):
    """Create a wireframe view of the mesh."""
    
    # Sample points for wireframe
    n_samples = min(len(pathfinder.centroids), 200)
    sample_indices = np.random.choice(len(pathfinder.centroids), 
                                    n_samples, replace=False)
    
    # Plot connections between sampled points
    for i in sample_indices:
        for neighbor in list(pathfinder.graph.neighbors(i))[:3]:
            ax.plot([pathfinder.centroids[i, 0], pathfinder.centroids[neighbor, 0]],
                   [pathfinder.centroids[i, 1], pathfinder.centroids[neighbor, 1]],
                   [pathfinder.centroids[i, 2], pathfinder.centroids[neighbor, 2]],
                   'b-', alpha=0.3, linewidth=0.5)
    
    # Plot path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                'r-', linewidth=3, label='Path')
        ax.scatter(*path[0], color='green', s=200, marker='o')
        ax.scatter(*path[-1], color='red', s=200, marker='*')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

def plot_top_view(ax, pathfinder, path, title):
    """2D top-down view."""
    
    # Plot mesh projection
    for i, tet in enumerate(pathfinder.tetrahedra[:200]):  # Limit for clarity
        vertices = pathfinder.mesh.points[tet]
        
        # Plot edges in 2D
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for e1, e2 in edges:
            ax.plot([vertices[e1, 0], vertices[e2, 0]],
                   [vertices[e1, 1], vertices[e2, 1]],
                   'gray', alpha=0.2, linewidth=0.5)
    
    # Plot path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 
                'b-', linewidth=3, label='Path')
        ax.scatter(path_array[:, 0], path_array[:, 1], 
                  c=path_array[:, 2], cmap='viridis', s=50, 
                  edgecolors='black', linewidth=0.5)
        
        # Start and goal
        ax.scatter(path[0][0], path[0][1], color='green', 
                  s=200, marker='o', zorder=5)
        ax.scatter(path[-1][0], path[-1][1], color='red', 
                  s=200, marker='*', zorder=5)
        
        # Add colorbar for altitude
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=Normalize(vmin=path_array[:, 2].min(),
                                                vmax=path_array[:, 2].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Altitude (m)')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def plot_side_view(ax, pathfinder, path, title):
    """2D side view (X-Z plane)."""
    
    # Plot mesh projection
    for i, tet in enumerate(pathfinder.tetrahedra[:200]):
        vertices = pathfinder.mesh.points[tet]
        
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for e1, e2 in edges:
            ax.plot([vertices[e1, 0], vertices[e2, 0]],
                   [vertices[e1, 2], vertices[e2, 2]],
                   'gray', alpha=0.2, linewidth=0.5)
    
    # Plot path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 2], 
                'b-', linewidth=3, label='Path')
        ax.scatter(path_array[:, 0], path_array[:, 2], 
                  c='blue', s=50)
        
        # Fill area under path
        ax.fill_between(path_array[:, 0], 0, path_array[:, 2], 
                       alpha=0.2, color='blue')
        
        ax.scatter(path[0][0], path[0][2], color='green', 
                  s=200, marker='o', zorder=5)
        ax.scatter(path[-1][0], path[-1][2], color='red', 
                  s=200, marker='*', zorder=5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m) - Altitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def plot_safety_view(ax, pathfinder, path, title):
    """3D view with safety coloring if ML is enabled."""
    
    if pathfinder.enable_ml:
        # Color tetrahedra by safety score
        safety_scores = []
        positions = []
        
        for i in range(min(200, len(pathfinder.tetrahedra))):
            safety = pathfinder.predict_safety_score(i)
            safety_scores.append(safety)
            positions.append(pathfinder.centroids[i])
        
        positions = np.array(positions)
        
        # Scatter plot with safety coloring
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=safety_scores, cmap='RdYlGn', s=50, alpha=0.6,
                           vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Safety Score')
    else:
        # Just plot centroids
        ax.scatter(pathfinder.centroids[:200, 0],
                  pathfinder.centroids[:200, 1],
                  pathfinder.centroids[:200, 2],
                  c='lightblue', s=20, alpha=0.5)
    
    # Plot path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                'b-', linewidth=4, label='Path')
        ax.scatter(*path[0], color='green', s=300, marker='o', label='Start')
        ax.scatter(*path[-1], color='red', s=300, marker='*', label='Goal')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

# Interactive visualization function
def create_interactive_visualization(pathfinder, path):
    """
    Create an interactive visualization that can be rotated.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh with different levels of detail
    # Level 1: Main structure (fewer elements, solid)
    plot_mesh_level(ax, pathfinder, level='structure')
    
    # Level 2: Path corridor (medium detail)
    plot_mesh_level(ax, pathfinder, level='corridor', path=path)
    
    # Level 3: The path itself
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                'r-', linewidth=5, label='Drone Path', zorder=1000)
        
        # Animated marker for current position
        ax.scatter(*path[0], color='green', s=400, marker='o', 
                  edgecolors='black', linewidth=3, label='Start', zorder=1001)
        ax.scatter(*path[-1], color='red', s=400, marker='*', 
                  edgecolors='black', linewidth=3, label='Goal', zorder=1001)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Interactive 3D Mesh and Path Visualization')
    ax.legend()
    
    # Add text box with instructions
    textstr = "Use mouse to rotate view\nScroll to zoom"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
    
    plt.show()

def plot_mesh_level(ax, pathfinder, level='structure', path=None):
    """Plot different levels of mesh detail."""
    
    if level == 'structure':
        # Plot sparse wireframe for overall structure
        n_samples = min(100, len(pathfinder.centroids))
        sample_indices = np.linspace(0, len(pathfinder.centroids)-1, 
                                   n_samples, dtype=int)
        
        for i in sample_indices:
            for neighbor in list(pathfinder.graph.neighbors(i))[:2]:
                ax.plot([pathfinder.centroids[i, 0], 
                        pathfinder.centroids[neighbor, 0]],
                       [pathfinder.centroids[i, 1], 
                        pathfinder.centroids[neighbor, 1]],
                       [pathfinder.centroids[i, 2], 
                        pathfinder.centroids[neighbor, 2]],
                       'lightgray', alpha=0.3, linewidth=0.5)
    
    elif level == 'corridor' and path:
        # Plot detailed mesh around the path
        path_tets = set()
        for waypoint in path:
            tet_idx = pathfinder.find_containing_tetrahedron(waypoint)
            if tet_idx is not None:
                path_tets.add(tet_idx)
                for neighbor in pathfinder.graph.neighbors(tet_idx):
                    path_tets.add(neighbor)
        
        for tet_idx in path_tets:
            tet = pathfinder.tetrahedra[tet_idx]
            vertices = pathfinder.mesh.points[tet]
            
            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            for e1, e2 in edges:
                ax.plot([vertices[e1, 0], vertices[e2, 0]],
                       [vertices[e1, 1], vertices[e2, 1]],
                       [vertices[e1, 2], vertices[e2, 2]],
                       'blue', alpha=0.4, linewidth=1)

# Main execution example
if __name__ == "__main__":
    # Create sample mesh
    print("Creating sample city mesh...")
    mesh, tetrahedra = create_sample_city_mesh()
    
    # Initialize pathfinder
    from pathfinder_ML import MLEnhancedPathfinder  # Your pathfinder class
    
    pathfinder = MLEnhancedPathfinder("city_with_buildings.vtu", enable_ml=True)
    
    # Find a path
    start = np.array([10, 10, 25])
    goal = np.array([90, 90, 35])
    
    print("Finding path...")
    path = pathfinder.find_path_ml(start, goal, safety_weight=0.5)
    
    if path:
        print(f"Path found with {len(path)} waypoints!")
        
        # Create comprehensive visualization
        print("Creating visualizations...")
        visualize_mesh_and_path(pathfinder, path, save_fig=True)
        
        # Create interactive visualization
        print("Creating interactive view...")
        create_interactive_visualization(pathfinder, path)
    else:
        print("No path found!")