import pyvista as pv
from collections import deque

def build_adjacency(mesh):
    """
    Builds a naive adjacency list where each cell (tetrahedron) is linked
    to neighbors that share a face (3 common vertices).
    This is O(n^2), but straightforward for smaller meshes.
    """
    n_cells = mesh.n_cells
    adjacency = [[] for _ in range(n_cells)]
    
    for i in range(n_cells):
        cell_i_points = set(mesh.get_cell_points(i))
        for j in range(i + 1, n_cells):
            cell_j_points = set(mesh.get_cell_points(j))
            # Tets sharing a face have exactly 3 vertices in common
            if len(cell_i_points.intersection(cell_j_points)) == 3:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    return adjacency

def find_path_bfs(mesh, adjacency, start_pt, goal_pt):
    """
    Simple BFS path search between two points in the mesh volume.
    1. Find which tetrahedron contains (or is closest to) start_pt and goal_pt.
    2. Explore neighbors layer by layer until we reach the goal cell.
    3. Return the sequence of cell IDs representing the path, or None if not found.
    """
    start_cell = mesh.find_closest_cell(start_pt)
    goal_cell  = mesh.find_closest_cell(goal_pt)

    if start_cell == -1 or goal_cell == -1:
        print("Error: Could not find valid tetrahedron for start or goal.")
        return None

    queue = deque([start_cell])
    visited = {start_cell}
    parent = {}

    while queue:
        current = queue.popleft()
        if current == goal_cell:
            # Reconstruct path from goal -> start
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1]  # reverse to get start->goal

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    return None

if __name__ == "__main__":
    # 1. Load your tetrahedral mesh from a VTU file
    mesh = pv.read("city_mesh.vtu")

    # 2. Build adjacency (which tets share a face?)
    adjacency = build_adjacency(mesh)

    # 3. Define start and goal points in 3D
    start_point = [10.0, 5.0, 2.0]
    goal_point  = [90.0, 45.0, 20.0]

    # 4. Attempt BFS pathfinding
    path = find_path_bfs(mesh, adjacency, start_point, goal_point)

    if path is not None:
        print("Path found! (Sequence of tetra cell IDs):")
        print(path)
    else:
        print("No path found.")
