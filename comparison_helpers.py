# community_compare.py
# Requer: python-igraph, leidenalg, matplotlib (pandas opcional)
import math
import itertools
import numpy as np
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt

# -------------------------
# Detectores
# -------------------------
def detect_leiden(g: ig.Graph, resolution: float = 1.0, weight_attr: str = None):
    weights = g.es[weight_attr] if (weight_attr and weight_attr in g.edge_attributes()) else None
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    weights=weights, resolution_parameter=resolution)
    return list(part.membership), part

def detect_louvain(g: ig.Graph, weight_attr: str = None):
    weights = g.es[weight_attr] if (weight_attr and weight_attr in g.edge_attributes()) else None
    # igraph's multilevel is Louvain-like
    vc = g.community_multilevel(weights=weights)
    return list(vc.membership), vc

def detect_infomap(g: ig.Graph, weight_attr: str = None, trials: int = 10):
    weights = g.es[weight_attr] if (weight_attr and weight_attr in g.edge_attributes()) else None
    vc = g.community_infomap(edge_weights=weights, trials=trials)
    return list(vc.membership), vc

# Optional: label propagation as extra baseline
def detect_label_propagation(g: ig.Graph):
    vc = g.community_label_propagation()
    return list(vc.membership), vc

# -------------------------
# Métricas de qualidade (sem ground truth)
# -------------------------
def modularity(g: ig.Graph, membership, weight_attr: str = None):
    weights = g.es[weight_attr] if (weight_attr and weight_attr in g.edge_attributes()) else None
    return g.modularity(membership, weights=weights)

def internal_density(g: ig.Graph, membership):
    # média (ou soma) da densidade interna de cada comunidade
    comms = {}
    for v, c in enumerate(membership):
        comms.setdefault(c, []).append(v)
    densities = []
    for nodes in comms.values():
        n = len(nodes)
        if n <= 1:
            densities.append(0.0)
            continue
        sub = g.subgraph(nodes)
        m = sub.ecount()
        possible = n * (n - 1) / 2
        densities.append(m / possible)
    return float(np.mean(densities)), densities

def avg_conductance(g: ig.Graph, membership, weight_attr: str = None):
    # conductance(S) = cut(S, V\S) / min(vol(S), vol(V\S)), vol = sum degrees (or sum weights)
    w = g.es[weight_attr] if (weight_attr and weight_attr in g.edge_attributes()) else None
    n = g.vcount()
    degs = np.array(g.strength(weights=w))  # strength works for weights or deg if None
    total_vol = degs.sum()
    comms = {}
    for v, c in enumerate(membership):
        comms.setdefault(c, []).append(v)
    conds = []
    for nodes in comms.values():
        node_set = set(nodes)
        # cut: sum weights of edges with one end in nodes and other outside
        cut = 0.0
        for e in g.es:
            a, b = e.tuple
            ew = (e["weight"] if ("weight" in e.attribute_names() and e["weight"] is not None) 
                  else (w[e.index] if (w is not None) else 1.0))
            in_a = a in node_set
            in_b = b in node_set
            if in_a ^ in_b:
                cut += (ew if ew is not None else 1.0)
        volS = degs[nodes].sum()
        volRest = total_vol - volS
        denom = min(volS, volRest)
        if denom <= 0:
            conds.append(0.0)
        else:
            conds.append(cut / denom)
    return float(np.mean(conds)), conds

def community_size_stats(membership):
    from collections import Counter
    c = Counter(membership)
    sizes = np.array(sorted(c.values(), reverse=True))
    return {
        "n_communities": len(sizes),
        "mean_size": float(sizes.mean()) if len(sizes)>0 else 0.0,
        "median_size": float(np.median(sizes)) if len(sizes)>0 else 0.0,
        "sizes": sizes.tolist()
    }

# -------------------------
# Métricas de comparação entre partições (pairwise): ARI e NMI
# Implementações independentes (não requer sklearn)
# -------------------------
def contingency_matrix(labels_true, labels_pred):
    # returns contingency dict and arrays
    from collections import defaultdict
    label_to_index_true = {}
    label_to_index_pred = {}
    for l in labels_true:
        if l not in label_to_index_true:
            label_to_index_true[l] = len(label_to_index_true)
    for l in labels_pred:
        if l not in label_to_index_pred:
            label_to_index_pred[l] = len(label_to_index_pred)
    R = len(label_to_index_true)
    C = len(label_to_index_pred)
    mat = np.zeros((R, C), dtype=int)
    for t, p in zip(labels_true, labels_pred):
        i = label_to_index_true[t]
        j = label_to_index_pred[p]
        mat[i, j] += 1
    return mat

def adjusted_rand_index(labels_true, labels_pred):
    # ARI formula using contingency table combinatorics
    from math import comb
    mat = contingency_matrix(labels_true, labels_pred)
    n = mat.sum()
    if n == 0:
        return 0.0
    sum_comb_c = sum(comb(int(x), 2) for x in mat.sum(axis=1))
    sum_comb_k = sum(comb(int(x), 2) for x in mat.sum(axis=0))
    sum_comb = sum(comb(int(x), 2) for x in mat.flatten())
    total = comb(n, 2)
    expected_index = (sum_comb_c * sum_comb_k) / total if total > 0 else 0.0
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    if max_index - expected_index == 0:
        return 0.0
    ari = (sum_comb - expected_index) / (max_index - expected_index)
    return float(ari)

def normalized_mutual_information(labels_true, labels_pred):
    # NMI = I(U;V) / sqrt(H(U) H(V))
    from math import log
    mat = contingency_matrix(labels_true, labels_pred).astype(float)
    n = mat.sum()
    if n == 0:
        return 0.0
    pi = mat.sum(axis=1)  # row sums
    pj = mat.sum(axis=0)  # col sums
    # entropies
    def H_from_probs(p):
        p_nonzero = p[p > 0] / p.sum()
        return -np.sum(p_nonzero * np.log(p_nonzero))
    HU = H_from_probs(pi)
    HV = H_from_probs(pj)
    # mutual information
    MI = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                pij = mat[i, j] / n
                pi_i = pi[i] / n
                pj_j = pj[j] / n
                MI += pij * math.log(pij / (pi_i * pj_j))
    # normalize
    denom = math.sqrt(HU * HV) if (HU > 0 and HV > 0) else (HU + HV) / 2.0 if (HU + HV) > 0 else 1.0
    nmi = MI / denom if denom > 0 else 0.0
    return float(nmi)

# -------------------------
# Pipeline de comparação: roda todos os métodos, calcula métricas e comparações
# -------------------------
def run_all_and_compare(g: ig.Graph, weight_attr: str = None, leiden_res=1.0, infomap_trials=10, include=['leiden','louvain','infomap']):
    partitions = {}
    backrefs = {}
    if 'leiden' in include:
        mem, part = detect_leiden(g, resolution=leiden_res, weight_attr=weight_attr)
        partitions['leiden'] = mem; backrefs['leiden'] = part
    if 'louvain' in include:
        mem, part = detect_louvain(g, weight_attr=weight_attr)
        partitions['louvain'] = mem; backrefs['louvain'] = part
    if 'infomap' in include:
        mem, part = detect_infomap(g, weight_attr=weight_attr, trials=infomap_trials)
        partitions['infomap'] = mem; backrefs['infomap'] = part
    if 'labelprop' in include:
        mem, part = detect_label_propagation(g)
        partitions['labelprop'] = mem; backrefs['labelprop'] = part

    # compute per-method quality metrics
    report = {}
    for name, mem in partitions.items():
        rep = {}
        rep['modularity'] = modularity(g, mem, weight_attr)
        avg_den, dens_list = internal_density(g, mem)
        rep['avg_internal_density'] = avg_den
        rep['internal_density_list'] = dens_list
        avg_cond, cond_list = avg_conductance(g, mem, weight_attr)
        rep['avg_conductance'] = avg_cond
        rep['conductance_list'] = cond_list
        rep.update(community_size_stats(mem))
        report[name] = rep

    # pairwise comparisons: ARI and NMI
    comparisons = {}
    names = list(partitions.keys())
    for a, b in itertools.combinations(names, 2):
        labels_a = partitions[a]
        labels_b = partitions[b]
        ari = adjusted_rand_index(labels_a, labels_b)
        nmi = normalized_mutual_information(labels_a, labels_b)
        comparisons[f"{a}__{b}"] = {"ARI": ari, "NMI": nmi}

    return partitions, backrefs, report, comparisons

# -------------------------
# Utility: pretty print report (and optional pandas DataFrame)
# -------------------------
def report_to_dataframe(report: dict, comparisons: dict = None):
    try:
        import pandas as pd
    except Exception:
        print("pandas não disponível; retornando dicionários brutos.")
        return None
    rows = []
    for name, rep in report.items():
        rows.append({
            "method": name,
            "modularity": rep['modularity'],
            "avg_internal_density": rep['avg_internal_density'],
            "avg_conductance": rep['avg_conductance'],
            "n_communities": rep['n_communities'],
            "mean_size": rep['mean_size'],
            "median_size": rep['median_size']
        })
    df = pd.DataFrame(rows).set_index("method").sort_values("modularity", ascending=False)
    # if comparisons provided, make a separate df
    comp_df = None
    if comparisons is not None:
        comp_rows = []
        for k, v in comparisons.items():
            a, b = k.split("__")
            comp_rows.append({"pair": k, "method_a": a, "method_b": b, "ARI": v["ARI"], "NMI": v["NMI"]})
        comp_df = pd.DataFrame(comp_rows).set_index("pair")
    return df, comp_df

# -------------------------
# Plotting helper: cria plots lado-a-lado usando 'grouped' layout (reutiliza ideia anterior)
# -------------------------
def plot_partitions_side_by_side(g: ig.Graph, partitions: dict, figsize=(15, 5), arrangement='circle', output_path=None):
    # partitions: dict[name] -> membership list
    n = len(partitions)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (name, mem) in zip(axes, partitions.items()):
        ax.set_title(name)
        # compute global grouped coords using the grouped routine but return coords without plotting
        part, coords = detect_and_plot_grouped_positions(g, mem, arrangement=arrangement)
        # draw edges, nodes, etc.
        ax.set_aspect('equal'); ax.axis('off')
        xs = coords[:,0]; ys = coords[:,1]
        # edges
        for e in g.es:
            s, t = e.tuple
            ax.plot([xs[s], xs[t]], [ys[s], ys[t]], linewidth=0.5, color='gray', alpha=0.6, zorder=1)
        # colors
        ncomms = len(set(mem))
        palette = ig.drawing.colors.ClusterColoringPalette(ncomms)
        colors = [palette.get(m) for m in mem]
        ax.scatter(xs, ys, s=30, c=colors, edgecolors='k', linewidths=0.3, zorder=2)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

# helper to compute grouped positions (without plotting); uses internal logic adapted from previous answer
def detect_and_plot_grouped_positions(g: ig.Graph, membership, arrangement='circle', layout_name='fr'):
    import numpy as np
    from math import sqrt
    # build comm nodes list
    comm_nodes = {}
    for v_idx, c in enumerate(membership):
        comm_nodes.setdefault(c, []).append(v_idx)
    comm_nodes = {c: comm_nodes[c] for c in sorted(comm_nodes.keys())}
    # per-community local layouts
    comm_coords = {}
    comm_radii = {}
    for c, nodes in comm_nodes.items():
        if len(nodes) == 1:
            coords = np.array([[0.0, 0.0]])
        else:
            subg = g.subgraph(nodes)
            if layout_name == "fr":
                layout = subg.layout_fruchterman_reingold()
            elif layout_name == "kk":
                layout = subg.layout_kamada_kawai()
            else:
                try:
                    layout = subg.layout(layout_name)
                except Exception:
                    layout = subg.layout_fruchterman_reingold()
            coords = np.array(layout.coords)
            if coords.shape[0] > 1:
                coords = coords - coords.mean(axis=0)
                std = coords.std()
                if std > 0:
                    coords = coords / std
        radii = np.linalg.norm(coords, axis=1)
        radius = float(radii.max() if len(radii)>0 else 0.5)
        if radius < 0.5:
            radius = 0.5
        comm_coords[c] = coords
        comm_radii[c] = radius
    # centers
    centers = {}
    ncomms = len(comm_coords)
    if arrangement == 'circle':
        R = 4.0 * (max(comm_radii.values()) + 1.5) * math.sqrt(ncomms)
        for i, c in enumerate(comm_coords.keys()):
            angle = 2 * math.pi * i / max(1, ncomms)
            centers[c] = np.array([R * math.cos(angle), R * math.sin(angle)])
    else:
        cols = int(math.ceil(math.sqrt(ncomms)))
        dx = 4.0 * (max(comm_radii.values()) + 1.2)
        dy = dx
        i = 0
        for r in range(int(math.ceil(ncomms / cols))):
            for col in range(cols):
                if i >= ncomms: break
                c = list(comm_coords.keys())[i]
                centers[c] = np.array([col * dx, -r * dy])
                i += 1
    # produce global coords
    global_coords = np.zeros((g.vcount(), 2), dtype=float)
    for c, nodes in comm_nodes.items():
        coords = comm_coords[c]
        center = centers[c]
        if coords.shape[0] == 1:
            coords_t = coords * 0.3 + center
        else:
            coords_t = coords * (1.0 / max(1.0, comm_radii[c])) * (comm_radii[c] * 1.2) + center
        jitter = np.random.normal(scale=0.02, size=coords_t.shape)
        coords_t = coords_t + jitter
        for local_idx, v_idx in enumerate(nodes):
            global_coords[v_idx, :] = coords_t[local_idx]
    return membership, global_coords

# -------------------------
# Example de uso (Zachary)
# -------------------------
if __name__ == "__main__":
    # carregue seu grafo aqui (ex: ig.Graph.Read_GraphML / Read_Ncol / etc)
    try:
        g = ig.Graph.Famous("Zachary")
        print("Zachary loaded")
    except Exception:
        print("Substitua pelo seu grafo real; este é só exemplo.")
        g = ig.Graph.Famous("Zachary")
    parts, refs, report, comps = run_all_and_compare(g, weight_attr=None, leiden_res=1.0, infomap_trials=10)
    print("Report (summary):")
    for k, v in report.items():
        print(k, "-> modularity:", v['modularity'], "avg_cond:", v['avg_conductance'], "ncomms:", v['n_communities'])
    print("\nPairwise comparisons (ARI / NMI):")
    for k, v in comps.items():
        print(k, "-> ARI:", v['ARI'], "NMI:", v['NMI'])
    # if pandas available, display DataFrame
    df_pair = report_to_dataframe(report, comps)
    if df_pair is not None:
        df, comp_df = df_pair
        print("\nDataFrame report:\n", df)
        print("\nComparisons:\n", comp_df)
    # plot side-by-side
    plot_partitions_side_by_side(g, {k:v for k,v in parts.items()}, figsize=(12,4))
