import math
import random
import numpy as np
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def detect_and_plot(g: ig.Graph,
                            resolution: float = 1.0,
                            weight_attr: str | None = None,
                            output_path: str | None = None,
                            layout_name: str = "fr",
                            vertex_size: int = 80,
                            arrangement: str = "circle",   # "circle" or "grid"
                            group_spacing: float = 4.0,    # distance between group centers
                            draw_group_circles: bool = True):
    """
    Detect communities with Leiden, compute sublayouts and place each community in its own region,
    then plot with matplotlib.

    Parameters
    ----------
    g : igraph.Graph
    resolution : float
    weight_attr : str | None
    output_path : str | None : If not None, saves to file.
    layout_name : str : used for per-community layout ("fr" or "kk" or others igraph supports)
    vertex_size : int : matplotlib marker size
    arrangement : "circle" or "grid"
    group_spacing : float : spacing multiplier between community centers
    draw_group_circles : bool : draw circle around each community
    """
    # 1) Run Leiden
    if weight_attr is not None and weight_attr in g.edge_attributes():
        weights = g.es[weight_attr]
    else:
        weights = None

    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                         weights=weights, resolution_parameter=resolution)
    membership = partition.membership
    ncomms = len(set(membership))
    print(f"Found {ncomms} communities.")

    # 2) Build list of node indices per community
    comm_nodes = {}
    for v_idx, c in enumerate(membership):
        comm_nodes.setdefault(c, []).append(v_idx)
    comm_nodes = {c: comm_nodes[c] for c in sorted(comm_nodes.keys())}

    # 3) For each community compute an internal layout (coords relative to (0,0))
    comm_coords = {}
    comm_radii = {}
    for c, nodes in comm_nodes.items():
        if len(nodes) == 1:
            # single node: at origin
            coords = np.array([[0.0, 0.0]])
        else:
            subg = g.subgraph(nodes)
            # choose layout
            if layout_name == "fr":
                layout = subg.layout_fruchterman_reingold()
            elif layout_name == "kk":
                layout = subg.layout_kamada_kawai()
            elif layout_name == "lgl":
                layout = subg.layout_lgl()
            else:
                try:
                    layout = subg.layout(layout_name)
                except Exception:
                    layout = subg.layout_fruchterman_reingold()
            coords = np.array(layout.coords)
            # normalize scale to unit std dev to make sizes comparable across communities
            if coords.shape[0] > 1:
                coords = coords - coords.mean(axis=0)
                std = coords.std()
                if std > 0:
                    coords = coords / std
        # compute radius (max distance from origin)
        radii = np.linalg.norm(coords, axis=1)
        radius = float(radii.max() if len(radii)>0 else 0.5)
        if radius < 0.5:
            radius = 0.5
        comm_coords[c] = coords
        comm_radii[c] = radius

    # 4) Compute positions for community centers (circle or grid)
    centers = {}
    if arrangement == "circle":
        R = group_spacing * (max(comm_radii.values()) + 1.5) * math.sqrt(ncomms)  # heuristic radius
        for i, c in enumerate(comm_coords.keys()):
            angle = 2 * math.pi * i / max(1, ncomms)
            centers[c] = np.array([R * math.cos(angle), R * math.sin(angle)])
    elif arrangement == "grid":
        # grid dims
        cols = int(math.ceil(math.sqrt(ncomms)))
        rows = int(math.ceil(ncomms / cols))
        dx = group_spacing * (max(comm_radii.values()) + 1.2)
        dy = dx
        i = 0
        for r in range(rows):
            for col in range(cols):
                if i >= ncomms: break
                c = list(comm_coords.keys())[i]
                centers[c] = np.array([col * dx, -r * dy])
                i += 1
    else:
        raise ValueError("arrangement must be 'circle' or 'grid'")

    # 5) Translate each community's local coords to global coords (apply center + small jitter)
    global_coords = np.zeros((g.vcount(), 2), dtype=float)
    for c, nodes in comm_nodes.items():
        coords = comm_coords[c]
        center = centers[c]
        # scale each community by its radius so communities have similar visual size
        scale = comm_radii[c] * 1.2
        # if coords is single point, keep it small
        if coords.shape[0] == 1:
            coords_t = coords * 0.3 + center
        else:
            coords_t = coords * (1.0 / max(1.0, comm_radii[c])) * scale + center
        # slight random jitter to avoid perfect overlaps
        jitter = np.random.normal(scale=0.02, size=coords_t.shape)
        coords_t = coords_t + jitter
        for local_idx, v_idx in enumerate(nodes):
            global_coords[v_idx, :] = coords_t[local_idx]

    # 6) Prepare colors
    palette = ig.drawing.colors.ClusterColoringPalette(ncomms)
    colors = [palette.get(m) for m in membership]

    # 7) Plot with matplotlib: edges then nodes then group circles
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect('equal')
    ax.axis('off')

    # edges: draw as lines between global coords
    # To make inter-community edges a little faded, we'll plot them with lower alpha
    for e in g.es:
        s = e.tuple[0]
        t = e.tuple[1]
        x1, y1 = global_coords[s]
        x2, y2 = global_coords[t]
        # if same community -> solid stronger line
        same = (membership[s] == membership[t])
        ax.plot([x1, x2], [y1, y2],
                linewidth=0.6 if same else 0.4,
                alpha=0.8 if same else 0.25,
                zorder=1,
                color='gray')

    # nodes
    xs = global_coords[:,0]
    ys = global_coords[:,1]
    ax.scatter(xs, ys, s=vertex_size,
               c=colors, edgecolors='k', linewidths=0.4, zorder=2)

    # labels (optional: you can comment this out if graph big)
    # labels = None
    # if "label" in g.vertex_attributes() and any(g.vs["label"]):
    #     labels = g.vs["label"]
    # elif "name" in g.vertex_attributes() and any(g.vs["name"]):
    #     labels = g.vs["name"]
    # if labels is not None:
    #     for i, lab in enumerate(labels):
    #         ax.text(xs[i], ys[i], str(lab),
    #                 fontsize=7, ha='center', va='center', zorder=3)

    # draw group circles around each community to emphasize separation
    if draw_group_circles:
        for c in comm_nodes.keys():
            center = centers[c]
            # choose radius large enough to include nodes plus margin
            node_indices = comm_nodes[c]
            if len(node_indices) == 0: continue
            dists = np.linalg.norm(global_coords[node_indices] - center, axis=1)
            rad = float(dists.max() + 0.6)
            circ = Circle(center, radius=rad, facecolor='none', edgecolor='black', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)
            ax.add_patch(circ)
            # label community id near its center
            ax.text(center[0], center[1] + rad + 0.2, f"Comm {c} (n={len(node_indices)})",
                    fontsize=9, ha='center', va='bottom', zorder=4, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # autoscale margins
    margin = 1.0
    xmin, xmax = xs.min() - margin, xs.max() + margin
    ymin, ymax = ys.min() - margin, ys.max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("Saved to:", output_path)
    plt.show()

    return partition, global_coords

# Example usage:
# g = ig.Graph.Famous("Zachary")   # or load/construct your graph
# partition, coords = detect_and_plot_grouped(g, arrangement='circle', group_spacing=3.0)
