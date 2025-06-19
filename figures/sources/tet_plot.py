# Plot partitioning of prism into tets
# Anders Logg 2023-12-05

from pylab import *
import seaborn as sns

colors = sns.color_palette("hls", 6)

font_props = {
    'family': 'sans-serif',
    'weight': 'bold',
    'size':   15
}



def draw_boundary_edges(ax, tets):
    """
    Given:
      - ax   : a mpl_toolkits.mplot3d Axes3D object
      - tets : a list of N tetrahedra, each of shape (4,3) (numpy arrays)
    
    This function finds every edge that appears in exactly one tetrahedron
    (i.e. a “boundary” edge) and plots it in solid black on ax.
    """
    # Step 1: Build a dictionary counting how often each edge shows up.
    # We store edges as pairs of 3-tuples ((x1,y1,z1), (x2,y2,z2)) sorted so that
    # (p,q) and (q,p) collapse to the same key.
    edge_count = {}
    for tet in tets:
        # Convert each vertex into a Python tuple for hashing
        pts = [tuple(pt) for pt in tet]  # length-4 list of (x,y,z) tuples
        # Enumerate all 6 edges of this tetrahedron
        for i in range(4):
            for j in range(i+1, 4):
                e = tuple(sorted((pts[i], pts[j])))
                edge_count[e] = edge_count.get(e, 0) + 1
    
    # Step 2: Draw any edge that appears exactly once
    for edge, count in edge_count.items():
        if count ==1 :
            (x1, y1, z1), (x2, y2, z2) = edge
            ax.plot([x1, x2], [y1, y2], [z1, z2],
                    color='k', linewidth=1.5)
        elif count >1 and edge[0][2] == edge[1][2]:
            # If the edge is horizontal, draw it in red
            (x1, y1, z1), (x2, y2, z2) = edge
            ax.plot([x1, x2], [y1, y2], [z1, z2],
                    color='k', linewidth=1.5)
        else:
            # If the edge is horizontal, draw it in red
            (x1, y1, z1), (x2, y2, z2) = edge
            ax.plot([x1, x2], [y1, y2], [z1, z2],
                    color='k',linestyle= 'dotted', alpha =0.8, linewidth=1.5)
    

def plot_tet(ax, v, color):
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=faces, alpha=0.05, color=color)
    for i in range(4):
        ax.plot([v[i, 0]], [v[i, 1]], [v[i, 2]], "ko", markersize=5)


def plot_tets(tets):
    ax = axes(projection="3d")
    for i, tet in enumerate(tets):
        plot_tet(ax, tet, colors[i])
    draw_boundary_edges(ax, tets)
    vertices = np.unique(concatenate(tets, axis=0), axis=0)
    for i, (x, y, z) in enumerate(vertices):
        lbl = f"v{i}"
        ax.plot([x], [y], [z], "ko", markersize=8,)
        ax.text(x, y, z + 0.001, rf"   $v_{i}$", color='k', family = 'monospace', zorder=10)
    ax.set_proj_type("persp", focal_length=0.3)
    ax.view_init(elev=15, azim=-80)
    ax.set_axis_off()
    plt.tight_layout()


def check_tets(tets):
    for tet in tets:
        print("det = %g" % abs(det(tet[1:] - tet[0])))


def partition_0(u, v, w):
    "Partition prism when 0 edges are cut"
    K0 = (u[0], u[1], u[2], v[2])
    K1 = (u[0], v[1], u[1], v[2])
    K2 = (u[0], v[0], v[1], v[2])
    return [array(K) for K in (K0, K1, K2)]


def partition_1(u, v, w):
    "Partition prism when 1 edge is cut"
    K0 = (u[0], u[1], u[2], w[0])
    K1 = (u[1], v[2], u[2], w[0])
    K2 = (u[1], v[2], w[0], v[1])
    K3 = (w[0], v[0], v[1], v[2])
    return [array(K) for K in (K0, K1, K2, K3)]


def partition_2(u, v, w):
    "Partition prism when 2 edges are cut"
    K0 = (u[0], u[1], u[2], w[1])
    K1 = (u[0], u[2], w[0], w[1])
    K2 = (u[2], v[2], w[0], w[1])
    K3 = (v[0], v[2], v[1], w[0])
    K4 = (v[1], v[2], w[1], w[0])
    return [array(K) for K in (K0, K1, K2, K3, K4)]


def partition_3(u, v, w):
    "Partition prism when 3 edges are cut"
    K0 = (u[0], u[1], u[2], w[2])
    K1 = (u[0], w[1], u[1], w[2])
    K2 = (u[0], w[0], w[1], w[2])
    K3 = (w[0], w[1], w[2], v[2])
    K4 = (w[0], v[1], w[1], v[2])
    K5 = (w[0], v[0], v[1], v[2])
    return [array(K) for K in (K0, K1, K2, K3, K4, K5)]


# Define vertices
h = sqrt(2 / 3)
z = array((0, 0, h))
u = array(((0, 0, 0), (1, 0, 0), (0.5, sqrt(3) / 2, 0)))
v = u + h * z
w = u + 0.5 * h * z
# tet = array([(0, 0, 0), (1, 0, 0), (0.5, sqrt(3) / 2, 0), (0.5, 1 / (2 * sqrt(3)), h)])

# Create partitions
tets_0 = partition_0(u, v, w)
tets_1 = partition_1(u, v, w)
tets_2 = partition_2(u, v, w)
tets_3 = partition_3(u, v, w)

# Check tets
check_tets(tets_0)
check_tets(tets_1)
check_tets(tets_2)
check_tets(tets_3)

# Plot partitions
figure()
plot_tets(tets_0)
# title(f"0 edges cut: {len(tets_0)} tets")

figure()
plot_tets(tets_1)

# title(f"1 edge cut: {len(tets_1)} tets")

figure()
plot_tets(tets_2)
# title(f"2 edges cut: {len(tets_2)} tets")

figure()
plot_tets(tets_3)
# title(f"3 edges cut: {len(tets_3)} tets")

show()
