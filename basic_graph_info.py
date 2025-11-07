import math
import numpy as np
import igraph as ig
from collections import Counter
import random
from typing import Optional, Tuple, List, Dict

def _safe_call(func, *args, fallback=None, **kwargs):
    """Tenta chamar func(*args, **kwargs). Se falhar, retorna fallback (ou lança se fallback None)."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return fallback

def _degree_stats(g: ig.Graph) -> Dict[str, float]:
    deg = g.degree()  # lista de inteiros
    arr = np.array(deg, dtype=float)
    stats = {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "cv": float(arr.std(ddof=0) / arr.mean()) if arr.mean() != 0 else float('nan'),
        "variance": float(arr.var(ddof=0)),
        "degree_histogram": None,  # preenchido adiante
        "degree_counts": dict(Counter(deg))
    }
    # histograma (valor de grau -> frequência)
    hist = Counter(deg)
    stats["degree_histogram"] = dict(sorted(hist.items()))
    return stats

def _basic_connectivity_info(g: ig.Graph) -> Dict:
    info = {}
    info['directed'] = g.is_directed()
    info['n_vertices'] = g.vcount()
    info['n_edges'] = g.ecount()
    info['density'] = _safe_call(g.density, fallback=float('nan'))
    # componentes
    comps = _safe_call(g.components, fallback=None)
    if comps is not None:
        info['n_components'] = len(comps)
        info['component_sizes'] = list(comps.sizes())
        # giant component
        giant = comps.giant()
        info['giant_size'] = giant.vcount()
        info['giant_fraction'] = giant.vcount() / g.vcount() if g.vcount() > 0 else 0.0
    else:
        info['n_components'] = None
        info['component_sizes'] = None
        info['giant_size'] = None
        info['giant_fraction'] = None
    return info

def _path_length_and_diameter(g: ig.Graph) -> Dict:
    # calcula em componente gigante para evitar problemas com grafo desconexo
    comps = _safe_call(g.components, fallback=None)
    if comps is None:
        # fallback: tenta funções diretas (pode lançar)
        try:
            diam = g.diameter()
            apl = g.average_path_length()
            return {"diameter": float(diam), "average_path_length": float(apl)}
        except Exception:
            return {"diameter": None, "average_path_length": None}
    else:
        giant = comps.giant()
        try:
            diam = _safe_call(giant.diameter, fallback=None)
        except Exception:
            diam = None
        try:
            apl = _safe_call(giant.average_path_length, fallback=None)
        except Exception:
            apl = None
        return {"diameter": float(diam) if diam is not None else None,
                "average_path_length": float(apl) if apl is not None else None,
                "giant_vcount": giant.vcount()}

def _clustering_info(g: ig.Graph) -> Dict:
    # tenta usar funções nativas do igraph, com fallback
    info = {}
    # transitivity global (um equivalente ao clustering global / coeficiente de agrupamento)
    tg = _safe_call(g.transitivity_undirected if hasattr(g, "transitivity_undirected") else g.transitivity, fallback=None)
    if tg is None:
        # fallback: usar igraph.transitivity_undirected (módulo) ou None
        tg = _safe_call(ig.Graph.transitivity_undirected, g, fallback=None)
    info['transitivity_global'] = float(tg) if tg is not None else None

    # média local do coeficiente de clustering
    tavl = None
    if hasattr(g, "transitivity_avglocal_undirected"):
        tavl = _safe_call(g.transitivity_avglocal_undirected, fallback=None)
    else:
        # fallback: calcular local clustering coef por vértice e tirar média
        try:
            local = _safe_call(g.transitivity_local_undirected, range(g.vcount()), fallback=None)
            if local is None:
                local = _safe_call(lambda: [float(x) for x in g.transitivity_local_undirected(range(g.vcount()))], fallback=None)
            if local is not None:
                tavl = float(np.nanmean([x for x in local if x is not None]))
        except Exception:
            tavl = None
    info['transitivity_avg_local'] = float(tavl) if tavl is not None else None

    # triângulos por vértice (quantos triângulos cada nó participa)
    tri_counts = _safe_call(g.count_triangles, fallback=None)
    if tri_counts is None:
        # grafos antigos: igraph.Graph.count_triangles existe como método; caso não, omitimos
        tri_counts = None
    info['triangles_per_vertex_sample'] = None
    if tri_counts is not None:
        # verdadeiro retorno pode ser um array com counts por vértice
        try:
            arr = np.array(tri_counts)
            info['triangles_per_vertex_sample'] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "max": float(arr.max())
            }
        except Exception:
            info['triangles_per_vertex_sample'] = None
    return info

def _degree_correlations(g: ig.Graph) -> Dict:
    # assortatividade de grau (coeficiente de Pearson entre graus nas extremidades das arestas)
    corr = None
    try:
        if hasattr(g, "assortativity_degree"):
            corr = g.assortativity_degree(directed=False)
        else:
            # fallback: calcular pearson entre deg(u) e deg(v) para cada aresta
            degs = g.degree()
            edges = g.get_edgelist()
            a_degs = np.array([degs[e[0]] for e in edges], dtype=float)
            b_degs = np.array([degs[e[1]] for e in edges], dtype=float)
            if len(a_degs) > 1:
                corr = float(np.corrcoef(a_degs, b_degs)[0,1])
            else:
                corr = None
    except Exception:
        corr = None

    # knn: average neighbor degree per degree value (se disponível)
    knn_info = None
    try:
        if hasattr(g, "knn"):
            knn_info = g.knn()[1]  # g.knn() retorna (degrees, knn)
    except Exception:
        knn_info = None

    return {"assortativity_degree": float(corr) if corr is not None else None,
            "knn_sample": knn_info}

def _dispersion_metrics(g: ig.Graph) -> Dict:
    # "dispersão da rede" interpretada como heterogeneidade do grau (coef. variação, Gini de grau)
    deg = np.array(g.degree(), dtype=float)
    mean = deg.mean()
    std = deg.std(ddof=0)
    cv = std / mean if mean != 0 else float('nan')

    # Gini coefficient for degree distribution
    def gini(x: np.ndarray) -> float:
        if x.size == 0:
            return float('nan')
        # Gini via sum|xi - xj| / (2 n^2 mean)
        x = x.flatten()
        if x.sum() == 0:
            return 0.0
        n = x.size
        # efficient computation
        sorted_x = np.sort(x)
        index = np.arange(1, n+1)
        return (2.0 * np.sum(index * sorted_x) - (n + 1) * np.sum(sorted_x)) / (n * np.sum(sorted_x))

    gini_deg = gini(deg)

    return {"degree_mean": float(mean), "degree_std": float(std), "degree_cv": float(cv), "degree_gini": float(gini_deg)}

# --- Robustez: área sob curva de componente gigante conforme removemos nós
def _robustness_auc(g: ig.Graph, removal_order: List[int]) -> float:
    """
    Dado um grafo e uma ordem de remoção de vértices (lista de índices de vértices),
    remove progressivamente os nós nessa ordem e computa a curva: tamanho do maior componente
    normalizado (frac. de nós restantes) vs frac. de nós removidos. Retorna AUC (0..1),
    onde valores maiores => grafo mais robusto.
    """
    n = g.vcount()
    if n == 0:
        return float('nan')
    remaining = set(range(n))
    sizes = []
    # cópia leve do grafo para remoções sucessivas (usamos subgraph por desempenho razoável)
    current = g
    removed = 0
    # início: frac removido = 0
    comps = current.components()
    sizes.append(comps.giant().vcount() / n)
    # iterar remoções (podemos amostrar se ordem muito grande)
    for idx in removal_order:
        if idx not in remaining:
            continue
        remaining.remove(idx)
        # reconstruir subgrafo com os nós restantes
        if len(remaining) == 0:
            sizes.append(0.0)
            break
        sub = g.subgraph(list(remaining))
        comps = sub.components()
        sizes.append(comps.giant().vcount() / n)
    # sizes length = removed+1, x axis fractions
    x = np.linspace(0.0, 1.0, len(sizes))
    y = np.array(sizes, dtype=float)
    # AUC via trapézio (normalizado por 1.0 porque x spans 0..1)
    auc = float(np.trapz(y, x))
    return auc

def _compute_robustness(g: ig.Graph, trials: int = 5, targeted: bool = True, random_trials: int = 10) -> Dict:
    """
    Retorna métricas de robustez:
      - auc_targeted: AUC removendo nós em ordem decrescente de grau (simula ataque dirigido)
      - auc_random_mean / std: média e desvio do AUC para remoções aleatórias (simula falhas)
    OBS: para grafos muito grandes isso pode ser custoso.
    """
    n = g.vcount()
    if n == 0:
        return {"auc_targeted": None, "auc_random_mean": None, "auc_random_std": None}

    # ordem targeted: graus decrescentes (se empates, ordem arbitrária)
    degs = g.degree()
    order_targeted = sorted(range(n), key=lambda v: degs[v], reverse=True)
    auc_targeted = _robustness_auc(g, order_targeted)

    # ordens random (média)
    aucs = []
    for t in range(random_trials):
        order_rand = list(range(n))
        random.shuffle(order_rand)
        aucs.append(_robustness_auc(g, order_rand))
    auc_rand_mean = float(np.mean(aucs)) if aucs else None
    auc_rand_std = float(np.std(aucs, ddof=0)) if aucs else None

    return {"auc_targeted": auc_targeted, "auc_random_mean": auc_rand_mean, "auc_random_std": auc_rand_std}

# -------------------------
# Função pública principal
# -------------------------
# def imprimir_informacoes_gerais(g: ig.Graph,
#                                 weight_attr: Optional[str] = None,
#                                 compute_robustness: bool = True,
#                                 random_robustness_trials: int = 5,
#                                 show_degree_histogram: bool = True) -> Dict:
#     """
#     Imprime informações gerais e estatísticas de uma rede (grafo igraph).
#     Parâmetros:
#       - g: igraph.Graph (pode ser direcionado; várias métricas usam versão gigante se desconexo)
#       - weight_attr: nome do atributo nas arestas para pesos (opcional)
#       - compute_robustness: se True, roda a simulação de robustez (pode ser custoso em grafos grandes)
#       - random_robustness_trials: quantas simulações aleatórias para estimar AUC média
#       - show_degree_histogram: se True, imprime histograma de graus resumo
#     A função imprime textos descritivos e retorna um dicionário com todas as métricas calculadas.
#     """
#     print("=== INFORMAÇÕES GERAIS DO GRAFO ===")
#     basic = _basic_connectivity_info(g)
#     print(f"Vértices (n): {basic['n_vertices']}")
#     print(f"Arestas (m): {basic['n_edges']}")
#     print(f"Direcionado: {basic['directed']}")
#     print(f"Densidade do grafo (m / possível): {basic['density']:.6f}" if basic['density'] is not None else "Densidade: N/A")
#     if basic['n_components'] is not None:
#         print(f"Número de componentes conexas: {basic['n_components']}")
#         print(f"Tamanhos das componentes (ex.: 5 maiores): {sorted(basic['component_sizes'], reverse=True)[:5]}")
#         print(f"Tamanho da componente gigante: {basic['giant_size']} ({basic['giant_fraction']*100:.2f}% dos nós)")
#     print()

#     # graus
#     print("=== DISTRIBUIÇÃO DE GRAUS ===")
#     deg_stats = _degree_stats(g)
#     print(f"Grau mínimo: {deg_stats['min']}, máximo: {deg_stats['max']}")
#     print(f"Média: {deg_stats['mean']:.4f}, mediana: {deg_stats['median']:.4f}, desvio padrão: {deg_stats['std']:.4f}")
#     print(f"Coeficiente de variação (CV = std/mean): {deg_stats['cv']:.4f}")
#     if show_degree_histogram:
#         print("Histograma (grau: frequência) — primeiros pares:")
#         # mostrar apenas os 20 primeiros graus por legibilidade
#         items = list(deg_stats["degree_histogram"].items())
#         print(dict(items[:20]))
#     print()

#     # caminhos e diâmetro (na giant component)
#     print("=== CAMINHOS E DIÂMETRO (componente gigante) ===")
#     pl = _path_length_and_diameter(g)
#     if pl['average_path_length'] is not None:
#         print(f"Comprimento médio do caminho (média de distâncias, componente gigante): {pl['average_path_length']:.4f}")
#     else:
#         print("Comprimento médio do caminho: N/A (não foi possível calcular na sua versão/estado do grafo)")
#     if pl['diameter'] is not None:
#         print(f"Diâmetro (maior distância) da componente gigante: {pl['diameter']}")
#     else:
#         print("Diâmetro: N/A")
#     print()

#     # coeficiente de agrupamento / transitivity
#     print("=== COEFICIENTES DE AGRUPAMENTO ===")
#     cl = _clustering_info(g)
#     print(f"Transitivity (global / coeficiente de agrupamento global): {cl['transitivity_global']}")
#     print(f"Transitivity (média local): {cl['transitivity_avg_local']}")
#     if cl.get('triangles_per_vertex_sample') is not None:
#         print(f"Triângulos por vértice (média/mediana/max): {cl['triangles_per_vertex_sample']}")
#     print()

#     # dispersão / heterogeneidade do grau
#     print("=== DISPERSÃO / HETEROGENEIDADE DO GRAU ===")
#     disp = _dispersion_metrics(g)
#     print(f"Média do grau: {disp['degree_mean']:.4f}, desvio: {disp['degree_std']:.4f}, CV: {disp['degree_cv']:.4f}")
#     print(f"Gini (dispersão do grau; 0=igualdade, 1=máxima desigualdade): {disp['degree_gini']:.4f}")
#     print()

#     # correlações de grau
#     print("=== CORRELAÇÕES DE GRAU / ASSORTATIVIDADE ===")
#     corr = _degree_correlations(g)
#     print(f"Assortatividade (coef. de correlação de grau entre extremos de arestas): {corr['assortativity_degree']}")
#     if corr['knn_sample'] is not None:
#         # knn retorna lista onde índice = grau e valor = avg neighbor degree
#         print("Exemplo knn (avg neighbor degree por grau) — primeiro pares grau:knn:")
#         knn = corr['knn_sample']
#         # knn é um array indexado por grau (pode faltar graus altos)
#         for k in range(min(10, len(knn))):
#             print(f"  grau={k} -> knn={knn[k]}")
#     print()

#     # robustez (simulação) — opcional
#     robustness_res = None
#     if compute_robustness:
#         print("=== ROBUSTEZ DO GRAFO (simulação AUC) ===")
#         print("Avaliando robustez por remoção de nós (ataque dirigido por grau e falhas aleatórias).")
#         robustness_res = _compute_robustness(g, random_trials=random_robustness_trials)
#         print(f"AUC (ataque dirigido por grau): {robustness_res['auc_targeted']:.4f}")
#         print(f"AUC média (remoção aleatória) ± std: {robustness_res['auc_random_mean']:.4f} ± {robustness_res['auc_random_std']:.4f}")
#         print("Interpretação: AUC mais alta => rede mantém componente gigante por mais remoções => mais robusta.")
#         print()

#     # montar dicionário final com tudo e retornar
#     resultado = {
#         "basic": basic,
#         "degree_stats": deg_stats,
#         "path_and_diameter": pl,
#         "clustering": cl,
#         "dispersion": disp,
#         "degree_correlations": corr,
#         "robustness": robustness_res
#     }

#     return resultado

def imprimir_informacoes_gerais(g: ig.Graph):
    """
    Imprime estatísticas básicas e estruturais de um grafo IGraph.

    Parâmetros
    ----------
    g : igraph.Graph
        O grafo a ser analisado.
    """
    print("="*60)
    print("📊 INFORMAÇÕES GERAIS DO GRAFO")
    print("="*60)
    
    # 1. Informações básicas
    print(f"• Número de nós (vértices): {g.vcount()}")
    print(f"• Número de arestas (links): {g.ecount()}")
    print(f"• Tipo de grafo: {'Direcionado' if g.is_directed() else 'Não direcionado'}")
    print()
    
    # 2. Distribuição de graus
    graus = g.degree()
    print("• Grau médio dos nós:", np.mean(graus))
    print("• Grau máximo:", np.max(graus))
    print("• Grau mínimo:", np.min(graus))
    print("• Desvio padrão dos graus:", np.std(graus))
    print()
    
    # 3. Coeficiente de agrupamento (clustering coefficient)
    # Transitivity global: mede probabilidade de formar triângulos (0–1)
    trans_global = g.transitivity_undirected()
    # Transitivity local: média do coeficiente de clustering local
    trans_local = np.mean([v for v in g.transitivity_local_undirected(mode="zero")])
    print("• Coeficiente de agrupamento (global):", round(trans_global, 4))
    print("• Coeficiente de agrupamento (médio por nó):", round(trans_local, 4))
    print()
    
    # 4. Comprimento médio do caminho (average path length)
    if g.is_connected():
        path_len = g.average_path_length(directed=False)
        diametro = g.diameter(directed=False)
        print("• Comprimento médio do caminho:", round(path_len, 4))
        print("• Diâmetro (maior distância entre dois nós):", diametro)
    else:
        comp = g.components().giant()
        path_len = comp.average_path_length(directed=False)
        diametro = comp.diameter(directed=False)
        print("• (Grafo desconectado → análise feita na componente gigante)")
        print("  Comprimento médio do caminho:", round(path_len, 4))
        print("  Diâmetro (maior distância):", diametro)
    print()
    
    # 5. Robustez da rede (proxy: tamanho da componente gigante)
    comps = g.components()
    giant = max(comps, key=len)
    frac_gigante = len(giant) / g.vcount()
    print("• Tamanho da maior componente conectada:", len(giant))
    print(f"• Proporção da maior componente: {frac_gigante:.2%}")
    print()
    
    # 6. Dispersão da rede (variabilidade dos graus)
    dispersao = np.std(graus) / np.mean(graus) if np.mean(graus) > 0 else 0
    print("• Dispersão (coeficiente de variação dos graus):", round(dispersao, 4))
    print()
    
    # 7. Correlação de grau (assortatividade)
    # Valor entre -1 e 1 → negativo: nós de alto grau se ligam a nós de baixo grau.
    assort = g.assortativity_degree(directed=False)
    print("• Correlação de grau (assortatividade):", round(assort, 4))
    print()
    
    # 8. Triângulos (se aplicável)
    try:
        tri_por_no = g.transitivity_local_undirected(mode="zero")
        total_triangulos = int(sum(tri_por_no) / 3)
        print(f"• Estimativa do número total de triângulos: {total_triangulos}")
    except Exception:
        print("• Número de triângulos: (não disponível nesta versão do igraph)")
    
    print("="*60)
    print()