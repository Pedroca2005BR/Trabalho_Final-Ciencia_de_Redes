import json
import pandas as pd
import igraph as ig
# from detect_and_plot_leiden import detect_and_plot

def prepare_data(country_code: str) -> ig.Graph:
    """Possible country codes: RO, HR, HU"""
    directory = "deezer_clean_data/"
    edges_file = f"{country_code}_edges.csv"
    genres_file = f"{country_code}_genres.json"

    try:
        # --- 1. Ler o CSV de arestas ---
        # O arquivo contém pares de IDs (u, v) representando amizades mútuas
        edges_df = pd.read_csv(directory + edges_file, header=None, skiprows=1)
    except FileNotFoundError:
        print(f"Arquivo {edges_file} não encontrado no diretório {directory}.")
        return None
    
    # Converter DataFrame para lista de tuplas (u, v)
    edges = list(zip(edges_df[0], edges_df[1]))

    # --- 2. Criar grafo não-direcionado ---
    G = ig.Graph(edges=edges, directed=False)

    print(f"Grafo criado com {G.vcount()} nós e {G.ecount()} arestas")

    # --- 3. Ler o JSON com preferências musicais ---
    with open(directory + genres_file, "r") as f:
        genres_data = json.load(f)

    # --- 4. Adicionar atributo 'genres' a cada nó ---
    # Os nós foram indexados de 0 em diante, então o índice do JSON = ID do nó
    genres_attr = [None] * G.vcount()  # inicializa lista de atributos

    for node_id, liked_genres in genres_data.items():
        node_id = int(node_id)
        if node_id < G.vcount():  # evita erro se houver IDs fora do range
            genres_attr[node_id] = liked_genres

    # adiciona o atributo ao grafo
    G.vs["genres"] = genres_attr

    return G

def print_graph_info(g: ig.Graph):
    print(f"Grafo tem {g.vcount()} nós e {g.ecount()} arestas.")
    if "genres" in g.vs.attributes():
        sample_node = g.vs[0]
        print(f"Nó de exemplo (ID {sample_node.index}) tem gêneros: {sample_node['genres']}")
    else:
        print("Atributo 'genres' não encontrado nos nós do grafo.")

def get_genre_distribution(g: ig.Graph, percentage: bool = False) -> dict:
    """Retorna a distribuição de gêneros no grafo.
    
    Parameters
    ----------
    g : ig.Graph
        O grafo com atributo 'genres' nos nós.
    percentage : bool, optional
        Se True, retorna a porcentagem de ouvintes por gênero.
        Se False (padrão), retorna a contagem absoluta.
    
    Returns
    -------
    dict
        Dicionário com gêneros como chaves e contagem (ou porcentagem) como valores,
        ordenado em ordem decrescente.
    """
    genre_count = {}
    for genres in g.vs["genres"]:
        if genres:
            for genre in genres:
                genre_count[genre] = genre_count.get(genre, 0) + 1
    
    # Calcula porcentagem se solicitado
    if percentage:
        total = g.vcount()
        if total > 0:
            genre_count = {g: round((count / total) * 100, 2) for g, count in genre_count.items()}
    
    # Ordena por quantidade de ouvintes em ordem decrescente
    return dict(sorted(genre_count.items(), key=lambda x: x[1], reverse=True))

def get_top_songs_by_community(partitions, graph: ig.Graph = None, song_attr: str = 'tracks', top_n: int = 5) -> dict:
    """Agrupa nós por comunidade (a partir de `partitions`) e retorna os top N itens

    Parameters
    ----------
    partitions : VertexClustering ou lista
        Objeto retornado por `leidenalg.find_partition` (ou lista de rótulos de comunidade).
    graph : ig.Graph, optional
        O grafo onde os atributos dos nós residem. Se None, tenta obter de
        `partitions.graph` (quando aplicável). Se não disponível, lança erro.
    song_attr : str
        Nome do atributo de vértice que contém as músicas (pode ser lista, tupla, set
        ou dict mapping song->count). Valor padrão: `'tracks'`.
    top_n : int
        Quantos itens retornar por comunidade.

    Returns
    -------
    dict
        Dicionário {community_id: [(item, count), ...]} ordenado por frequência decrescente.

    Observações
    ----------
    - Se o atributo `song_attr` não existir, a função tenta usar `'genres'` como
      fallback e retorna os gêneros mais comuns por comunidade.
    """
    from collections import Counter

    # extrai membership
    if hasattr(partitions, 'membership'):
        membership = list(partitions.membership)
        if graph is None and hasattr(partitions, 'graph'):
            try:
                graph = partitions.graph
            except Exception:
                graph = None
    elif isinstance(partitions, (list, tuple)):
        membership = list(partitions)
    else:
        raise TypeError("'partitions' deve ser um objeto VertexClustering ou uma lista de rótulos de comunidade")

    if graph is None:
        raise ValueError("Parâmetro 'graph' não fornecido e não encontrado em 'partitions'. Passe o grafo como argumento.")

    attr_names = graph.vs.attribute_names()
    # fallback para 'genres' se não existir o atributo de músicas
    if song_attr not in attr_names:
        if 'genres' in attr_names:
            # computa top gêneros por comunidade
            comms = {}
            for v_idx, c in enumerate(membership):
                comms.setdefault(c, []).append(v_idx)
            result = {}
            for c, nodes in comms.items():
                ctr = Counter()
                for v in nodes:
                    genres = graph.vs[v]['genres']
                    if not genres:
                        continue
                    for g in genres:
                        ctr[g] += 1
                result[c] = ctr.most_common(top_n)
            return result
        raise ValueError(f"Atributo de nó '{song_attr}' não encontrado. Atributos disponíveis: {attr_names}")

    # agrupa nós por comunidade
    comms = {}
    for v_idx, c in enumerate(membership):
        comms.setdefault(c, []).append(v_idx)

    result = {}
    for c, nodes in comms.items():
        ctr = Counter()
        for v in nodes:
            val = graph.vs[v].get(song_attr)
            if val is None:
                continue
            # se for dict: key -> count
            if isinstance(val, dict):
                for k, cnt in val.items():
                    try:
                        ctr[k] += int(cnt)
                    except Exception:
                        ctr[k] += 1
            # se for lista/tupla/set: cada entrada conta 1
            elif isinstance(val, (list, tuple, set)):
                for item in val:
                    ctr[item] += 1
            else:
                # valor escalar (ex: string)
                ctr[val] += 1
        result[c] = ctr.most_common(top_n)

    return result


def is_scale_free(g: ig.Graph, method: str = 'regression', kmin: int | None = None,
                  gamma_range: tuple = (2.0, 3.0), r2_threshold: float = 0.8,
                  min_points: int = 3, return_details: bool = False) -> bool:
    """Testa se um grafo é compatível com uma distribuição livre de escala.

    Implementação leve baseada em regressão linear no espaço log-log da
    distribuição de grau (P(k) vs k). Não requer dependências externas
    (como `powerlaw`) e fornece uma heurística útil para análise exploratória.

    Parameters
    ----------
    g : ig.Graph
        Grafo a ser testado.
    method : str
        Atualmente apenas 'regression' é suportado (regressão log-log).
    kmin : int | None
        Grau mínimo a considerar na cauda (se None, usa 1).
    gamma_range : tuple
        Intervalo aceitável para o expoente $
        \gamma$ (por exemplo (2.0, 3.0)).
    r2_threshold : float
        Valor mínimo de R^2 da regressão para aceitar a hipótese de lei de potência.
    min_points : int
        Número mínimo de pontos únicos (k) na cauda para executar a regressão.
    return_details : bool
        Se True, retorna um dicionário com detalhes em vez de apenas bool.

    Returns
    -------
    bool ou dict
        Se `return_details` for False, retorna True/False indicando se o grafo passa
        no teste heurístico. Se True, retorna dicionário com 'is_scale_free',
        'gamma', 'r2', 'n_points', 'slope', 'intercept' e arrays usados.
    """
    if method != 'regression':
        raise ValueError("Apenas método 'regression' está implementado nesta função.")

    import numpy as np
    from collections import Counter

    degs = np.array(g.degree())
    if kmin is None:
        kmin = 1

    # conta frequência de graus >= kmin
    degs_tail = degs[degs >= kmin]
    if degs_tail.size == 0:
        result = {'is_scale_free': False, 'reason': 'no_degrees_ge_kmin'}
        return result if return_details else False

    cnt = Counter(degs_tail)
    ks = np.array(sorted([k for k in cnt.keys() if k > 0]))
    freqs = np.array([cnt[k] for k in ks], dtype=float)
    # probabilidades P(k)
    pk = freqs / freqs.sum()

    n_points = ks.size
    if n_points < min_points:
        result = {'is_scale_free': False, 'reason': 'too_few_points', 'n_points': n_points}
        return result if return_details else False

    # regressão linear em log-log
    logk = np.log(ks)
    logpk = np.log(pk)

    slope, intercept = np.polyfit(logk, logpk, 1)
    pred = slope * logk + intercept
    ss_res = np.sum((logpk - pred) ** 2)
    ss_tot = np.sum((logpk - logpk.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    gamma = -slope  # P(k) ~ k^{-gamma} => slope ~= -gamma

    is_sf = (gamma_range[0] <= gamma <= gamma_range[1]) and (r2 >= r2_threshold)

    details = {
        'is_scale_free': bool(is_sf),
        'gamma': float(gamma),
        'r2': float(r2),
        'n_points': int(n_points),
        'slope': float(slope),
        'intercept': float(intercept),
        'ks': ks,
        'pk': pk
    }

    return details if return_details else bool(is_sf)


def is_small_world(g: ig.Graph) -> float:
    """Verifica se o grafo é um "small world" baseado na razão entre
    o comprimento médio do caminho e o logaritmo do número de nós.

    Parameters
    ----------
    g : ig.Graph
        O grafo a ser testado.
    threshold : float
        Valor máximo aceitável para a razão (média do caminho / log(N)).

    Returns
    -------
    bool
        True se o grafo for considerado "small world", False caso contrário.
    """
    import numpy as np

    if g.vcount() < 2:
        return False  # grafos muito pequenos não são considerados

    try:
        avg_path_length = g.average_path_length()
    except Exception:
        return False  # se não for conexo ou erro, retorna False

    n = g.vcount()
    ratio = avg_path_length / np.log(n)

    return ratio


def musical_influence_analysis(g: ig.Graph, centrality_measure: str = 'degree',
                               percentile_low: int = 25, percentile_high: int = 75,
                               return_details: bool = False) -> dict:
    """Analisa se nós mais centrais tendem a ouvir gêneros diferentes de nós menos centrais.

    A hipótese de "influência musical" é verificada através da comparação entre
    a diversidade/popularidade dos gêneros ouvidos pelos nós mais centrais
    versus os menos centrais.

    Parameters
    ----------
    g : ig.Graph
        Grafo com atributo 'genres' nos nós.
    centrality_measure : str
        Medida de centralidade: 'degree', 'betweenness', 'closeness', 'eigenvector'.
    percentile_low : int
        Percentil inferior para definir nós "menos centrais" (ex: 25).
    percentile_high : int
        Percentil superior para definir nós "mais centrais" (ex: 75).
    return_details : bool
        Se True, retorna dicionário com análises detalhadas; se False, retorna
        apenas um float com a razão de diferença de diversidade.

    Returns
    -------
    dict ou float
        Se `return_details` for False, retorna um float na faixa [0, 1] indicando
        a diferença normalizada entre diversidade de gêneros (nós centrais vs nós periféricos).
        Se True, retorna dicionário com 'diversity_central', 'diversity_peripheral',
        'ratio', 'n_central', 'n_peripheral', 'centrality_values'.
    """
    import numpy as np
    from collections import Counter

    # calcula centralidade
    if centrality_measure == 'degree':
        centrality = np.array(g.degree())
    elif centrality_measure == 'betweenness':
        centrality = np.array(g.betweenness())
    elif centrality_measure == 'closeness':
        centrality = np.array(g.closeness())
    elif centrality_measure == 'eigenvector':
        try:
            centrality = np.array(g.eigenvector_centrality())
        except Exception:
            # se não convergir, usa grau como fallback
            centrality = np.array(g.degree())
    else:
        raise ValueError(f"Medida de centralidade '{centrality_measure}' não suportada.")

    # define limites de percentis
    low_threshold = np.percentile(centrality, percentile_low)
    high_threshold = np.percentile(centrality, percentile_high)

    # indices de nós em cada grupo
    idx_peripheral = np.where(centrality <= low_threshold)[0]
    idx_central = np.where(centrality >= high_threshold)[0]

    # calcula diversidade: usa Shanon entropy ou número único de gêneros por grupo
    def calc_diversity(indices):
        genres_list = []
        for i in indices:
            genres = g.vs[i]['genres']
            if genres:
                genres_list.extend(genres)
        if not genres_list:
            return 0.0, 0
        cnt = Counter(genres_list)
        n_unique = len(cnt)
        # Shannon entropy: H = -sum(p_i * log(p_i))
        total = sum(cnt.values())
        h = 0.0
        for count in cnt.values():
            p = count / total
            if p > 0:
                h -= p * np.log(p)
        return h, n_unique

    entropy_central, n_unique_central = calc_diversity(idx_central)
    entropy_periph, n_unique_periph = calc_diversity(idx_peripheral)

    # razão de diversidade
    if entropy_periph > 0:
        ratio = entropy_central / entropy_periph
    else:
        ratio = 1.0 if entropy_central == 0 else float('inf')

    details = {
        'entropy_central': float(entropy_central),
        'entropy_peripheral': float(entropy_periph),
        'n_unique_genres_central': int(n_unique_central),
        'n_unique_genres_peripheral': int(n_unique_periph),
        'ratio_entropy': float(ratio),
        'n_central_nodes': int(len(idx_central)),
        'n_peripheral_nodes': int(len(idx_peripheral)),
        'centrality_measure': centrality_measure,
        'centrality_values': centrality
    }

    return details if return_details else float(ratio)


def check_neighbor_similarity(g: ig.Graph, percentile_low: int = 25,
                              similarity_metric: str = 'jaccard',
                              return_details: bool = False) -> dict:
    """Verifica se nós menos centrais (periféricos) ouvem gêneros similares aos
    seus vizinhos mais centrais.

    A hipótese é: nós periféricos tendem a adotar gostos musicais similares
    aos seus vizinhos mais centrais (influência de amigos com maior centralidade)?

    Parameters
    ----------
    g : ig.Graph
        Grafo com atributo 'genres' nos nós.
    percentile_low : int
        Percentil para definir nós "menos centrais" (ex: 25).
    similarity_metric : str
        'jaccard' — similaridade de Jaccard entre conjuntos de gêneros.
        'cosine' — similaridade de cosseno (baseada em frequência).
    return_details : bool
        Se True, retorna dict completo; se False, retorna um float [0, 1].

    Returns
    -------
    dict ou float
        Se False, retorna float: média de similaridade entre nós periféricos e
        seus vizinhos mais centrais.
        Se True, retorna dicionário com 'avg_similarity', 'std_similarity',
        'n_peripheral_nodes', 'n_edges_periph_to_central', 'similarities'.
    """
    import numpy as np

    # calcula grau (medida simples de centralidade)
    degrees = np.array(g.degree())
    low_threshold = np.percentile(degrees, percentile_low)

    idx_peripheral = np.where(degrees <= low_threshold)[0]
    idx_central_set = set(np.where(degrees > low_threshold)[0])

    # função de similaridade
    def jaccard_similarity(set_a, set_b):
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def cosine_similarity(list_a, list_b):
        from collections import Counter
        if not list_a or not list_b:
            return 0.0
        cnt_a = Counter(list_a)
        cnt_b = Counter(list_b)
        all_genres = set(cnt_a.keys()) | set(cnt_b.keys())
        dot_product = sum(cnt_a.get(g, 0) * cnt_b.get(g, 0) for g in all_genres)
        norm_a = sum(cnt_a[g] ** 2 for g in all_genres) ** 0.5
        norm_b = sum(cnt_b[g] ** 2 for g in all_genres) ** 0.5
        denom = norm_a * norm_b
        return dot_product / denom if denom > 0 else 0.0

    similarities = []
    edges_periph_to_central = 0

    for v_periph in idx_peripheral:
        genres_v = set(g.vs[v_periph]['genres'] or [])
        if not genres_v:
            continue

        # encontra vizinhos que são mais centrais
        neighbors = set(g.neighbors(v_periph))
        neighbors_central = neighbors & idx_central_set

        if not neighbors_central:
            continue

        edges_periph_to_central += len(neighbors_central)

        # calcula similaridade com cada vizinho central
        for v_central in neighbors_central:
            genres_central = g.vs[v_central]['genres'] or []

            if similarity_metric == 'jaccard':
                sim = jaccard_similarity(genres_v, set(genres_central))
            elif similarity_metric == 'cosine':
                sim = cosine_similarity(list(genres_v), genres_central)
            else:
                raise ValueError(f"Métrica '{similarity_metric}' não suportada.")

            similarities.append(sim)

    if not similarities:
        avg_sim = 0.0
        std_sim = 0.0
    else:
        avg_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))

    details = {
        'avg_similarity': avg_sim,
        'std_similarity': std_sim,
        'n_peripheral_nodes': int(len(idx_peripheral)),
        'n_edges_periph_to_central': int(edges_periph_to_central),
        'similarity_metric': similarity_metric,
        'percentile_threshold': percentile_low,
        'similarities': similarities
    }

    return details if return_details else avg_sim


def plot_degree_distribution(g: ig.Graph, title: str = "Distribuição de Graus",
                             figsize: tuple = (12, 5), show_fit: bool = True,
                             output_path: str | None = None):
    """Plota a distribuição de graus em escala log-log para visualizar lei de potência.

    A visualização em log-log ajuda a identificar visualmente se o grafo segue
    uma distribuição de lei de potência (power-law). Uma reta em log-log indica
    conformidade com lei de potência.

    Parameters
    ----------
    g : ig.Graph
        Grafo a ser analisado.
    title : str
        Título do gráfico.
    figsize : tuple
        Tamanho da figura (width, height).
    show_fit : bool
        Se True, ajusta e plota uma reta em log-log (regressão linear).
    output_path : str | None
        Se fornecido, salva a figura neste caminho.

    Returns
    -------
    None
        Exibe o gráfico usando matplotlib.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    degs = np.array(g.degree())
    deg_count = Counter(degs)

    # ordena por grau
    ks = np.array(sorted(deg_count.keys()))
    freqs = np.array([deg_count[k] for k in ks], dtype=float)
    # probabilidade P(k)
    pk = freqs / freqs.sum()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Escala Linear ---
    ax_lin = axes[0]
    ax_lin.bar(ks, pk, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_lin.set_xlabel('Grau (k)', fontsize=11)
    ax_lin.set_ylabel('P(k)', fontsize=11)
    ax_lin.set_title('Escala Linear', fontsize=12, fontweight='bold')
    ax_lin.grid(True, alpha=0.3, linestyle='--')

    # --- Plot 2: Escala Log-Log ---
    ax_log = axes[1]

    # remove zeros para log
    mask = pk > 0
    ks_nonzero = ks[mask]
    pk_nonzero = pk[mask]

    ax_log.scatter(ks_nonzero, pk_nonzero, s=50, alpha=0.6, color='darkblue', 
                   edgecolor='black', linewidth=0.5, label='Dados observados')

    # ajusta reta em log-log se solicitado
    if show_fit and len(ks_nonzero) > 2:
        logk = np.log(ks_nonzero)
        logpk = np.log(pk_nonzero)
        slope, intercept = np.polyfit(logk, logpk, 1)
        
        # calcula R²
        pred = slope * logk + intercept
        ss_res = np.sum((logpk - pred) ** 2)
        ss_tot = np.sum((logpk - logpk.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # plota reta ajustada
        k_fit = np.linspace(ks_nonzero.min(), ks_nonzero.max(), 100)
        pk_fit = np.exp(slope * np.log(k_fit) + intercept)
        ax_log.plot(k_fit, pk_fit, 'r-', linewidth=2.5, label=f'Ajuste: γ={-slope:.2f}, R²={r2:.3f}')

    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_xlabel('Grau (k)', fontsize=11)
    ax_log.set_ylabel('P(k)', fontsize=11)
    ax_log.set_title('Escala Log-Log (Power-Law Check)', fontsize=12, fontweight='bold')
    ax_log.grid(True, alpha=0.3, linestyle='--', which='both')
    ax_log.legend(loc='best', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")

    plt.show()


def evaluate_community_quality(g: ig.Graph, partitions, return_details: bool = True) -> dict | float:
    """Avalia a qualidade e conectividade das comunidades detectadas.

    Calcula múltiplas métricas de qualidade: modularidade, densidade interna,
    condutância (isolamento), tamanho dos clusters, etc.

    Parameters
    ----------
    g : ig.Graph
        Grafo onde as comunidades foram detectadas.
    partitions : VertexClustering, lista ou dict
        Partição de comunidades. Pode ser:
        - Objeto VertexClustering (leidenalg, igraph)
        - Lista de rótulos de comunidade [0, 1, 1, 2, ...]
        - Dict {node_id: community_id}
    return_details : bool
        Se True, retorna dict completo; se False, retorna apenas a modularidade.

    Returns
    -------
    dict ou float
        Se `return_details` for False, retorna float (modularidade).
        Se True, retorna dict com métricas detalhadas:
        - 'modularity': modularidade global
        - 'n_communities': número de comunidades
        - 'n_nodes': número total de nós
        - 'internal_density': densidade média dentro de comunidades
        - 'conductance': condutância média (fluxo para fora)
        - 'community_sizes': distribuição de tamanhos
        - 'quality_score': score [0, 1] combinando métricas
    """
    import numpy as np

    # extrai membership
    if hasattr(partitions, 'membership'):
        membership = list(partitions.membership)
    elif isinstance(partitions, (list, tuple)):
        membership = list(partitions)
    elif isinstance(partitions, dict):
        membership = [partitions.get(v, 0) for v in range(g.vcount())]
    else:
        raise TypeError("'partitions' deve ser VertexClustering, lista ou dict")

    # valida membership
    if len(membership) != g.vcount():
        raise ValueError(f"Tamanho do membership ({len(membership)}) != nós do grafo ({g.vcount()})")

    # constrói grupos de nós por comunidade
    comms = {}
    for v_idx, c in enumerate(membership):
        comms.setdefault(c, []).append(v_idx)

    n_comms = len(comms)
    n_nodes = g.vcount()

    # --- 1. Modularidade ---
    modularity = g.modularity(membership)

    # --- 2. Densidade Interna ---
    densities = []
    for nodes in comms.values():
        n = len(nodes)
        if n <= 1:
            densities.append(0.0)
            continue
        subg = g.subgraph(nodes)
        m_internal = subg.ecount()
        possible_edges = n * (n - 1) / 2
        densities.append(m_internal / possible_edges if possible_edges > 0 else 0.0)

    avg_internal_density = float(np.mean(densities)) if densities else 0.0

    # --- 3. Condutância (fluxo de fora da comunidade) ---
    conductances = []
    total_degree = np.array(g.degree())

    for nodes in comms.values():
        node_set = set(nodes)
        # arestas que cruzam a fronteira
        cut_edges = 0
        for v in nodes:
            for neighbor in g.neighbors(v):
                if neighbor not in node_set:
                    cut_edges += 1

        # volume (soma de graus)
        vol = sum(total_degree[v] for v in nodes)

        if vol > 0:
            conductance = cut_edges / vol
        else:
            conductance = 0.0

        conductances.append(conductance)

    avg_conductance = float(np.mean(conductances)) if conductances else 0.0

    # --- 4. Distribuição de tamanhos ---
    sizes = np.array([len(nodes) for nodes in comms.values()])
    size_stats = {
        'min': int(sizes.min()),
        'max': int(sizes.max()),
        'mean': float(sizes.mean()),
        'median': float(np.median(sizes)),
        'std': float(sizes.std())
    }

    # --- 5. Score de qualidade combinado [0, 1] ---
    # quanto maior a modularidade, melhor
    # quanto menor a condutância, melhor (comunidades bem isoladas)
    # quanto maior a densidade interna, melhor
    q_modularity = max(0.0, modularity)  # normaliza para [0, ...]
    q_density = avg_internal_density  # já em [0, 1]
    q_isolation = 1.0 - avg_conductance  # quanto menor cond, maior score

    # combina com pesos
    quality_score = (0.5 * q_modularity + 0.3 * q_density + 0.2 * q_isolation)

    details = {
        'modularity': float(modularity),
        'n_communities': int(n_comms),
        'n_nodes': int(n_nodes),
        'internal_density_avg': avg_internal_density,
        'internal_density_per_community': [float(d) for d in densities],
        'conductance_avg': avg_conductance,
        'conductance_per_community': [float(c) for c in conductances],
        'community_sizes': size_stats,
        'quality_score': float(quality_score)
    }

    return details if return_details else float(modularity)


def detect_and_plot_with_quality(g: ig.Graph, resolution: float = 1.0,
                                 plot: bool = True, return_quality: bool = True):
    """Detecta comunidades com Leiden e retorna tanto a partição quanto sua qualidade.

    Wrapper que chama `detect_and_plot` e em seguida `evaluate_community_quality`.

    Parameters
    ----------
    g : ig.Graph
        Grafo para detectar comunidades.
    resolution : float
        Parâmetro de resolução do algoritmo Leiden.
    plot : bool
        Se True, plota o resultado usando `detect_and_plot`.
    return_quality : bool
        Se True, retorna (partitions, quality_dict); se False, retorna apenas partitions.

    Returns
    -------
    tuple ou VertexClustering
        Se `return_quality` for True: (partitions, quality_dict)
        Se False: partitions
    """
    from detect_and_plot_leiden import detect_and_plot

    # detecta comunidades
    result = detect_and_plot(g, resolution=resolution, plot=plot)

    if plot:
        # detect_and_plot retorna (partition, coords) quando plot=True
        partitions = result[0] if isinstance(result, tuple) else result
    else:
        partitions = result

    if return_quality:
        quality = evaluate_community_quality(g, partitions, return_details=True)
        return partitions, quality
    else:
        return partitions


def analyze_inter_community_communication(g: ig.Graph, partitions,
                                          return_details: bool = True) -> dict:
    """Analisa a comunicação (arestas) entre comunidades.

    Computa quantas arestas conectam cada par de comunidades,
    e calcula métricas de volume de tráfego inter-comunitário.

    Parameters
    ----------
    g : ig.Graph
        Grafo com as comunidades.
    partitions : VertexClustering, lista ou dict
        Partição de comunidades.
    return_details : bool
        Se True, retorna dict completo; se False, retorna apenas matriz de comunicação.

    Returns
    -------
    dict ou array
        Se False, retorna matriz de arestas inter-comunitárias.
        Se True, retorna dict com:
        - 'inter_community_edges': matriz de comunicação
        - 'total_inter_edges': total de arestas entre comunidades
        - 'total_intra_edges': total de arestas dentro de comunidades
        - 'edge_ratio': proporção inter/total
        - 'communication_strength': força normalizada de cada par
        - 'community_connectivity': quantas outras comunidades cada uma toca
    """
    import numpy as np

    # extrai membership
    if hasattr(partitions, 'membership'):
        membership = list(partitions.membership)
    elif isinstance(partitions, (list, tuple)):
        membership = list(partitions)
    elif isinstance(partitions, dict):
        membership = [partitions.get(v, 0) for v in range(g.vcount())]
    else:
        raise TypeError("'partitions' deve ser VertexClustering, lista ou dict")

    n_comms = max(membership) + 1
    # matriz de arestas entre comunidades
    inter_edges = np.zeros((n_comms, n_comms), dtype=int)
    intra_edges = 0

    for edge in g.es:
        u, v = edge.tuple
        comm_u = membership[u]
        comm_v = membership[v]

        if comm_u == comm_v:
            intra_edges += 1
        else:
            # garante simetria
            if comm_u < comm_v:
                inter_edges[comm_u, comm_v] += 1
            else:
                inter_edges[comm_v, comm_u] += 1

    # torna simétrica
    inter_edges = inter_edges + inter_edges.T

    total_inter = np.sum(inter_edges) // 2  # divide por 2 pois contamos duas vezes
    total_edges = g.ecount()
    edge_ratio = total_inter / total_edges if total_edges > 0 else 0.0

    # força normalizada (max = 1)
    max_inter = inter_edges.max() if inter_edges.max() > 0 else 1
    communication_strength = inter_edges / max_inter

    # conectividade: quantas comunidades cada uma se conecta
    connectivity = np.zeros(n_comms, dtype=int)
    for i in range(n_comms):
        connectivity[i] = np.count_nonzero(inter_edges[i])

    details = {
        'inter_community_edges': inter_edges,
        'total_inter_edges': int(total_inter),
        'total_intra_edges': int(intra_edges),
        'edge_ratio': float(edge_ratio),
        'communication_strength': communication_strength,
        'community_connectivity': connectivity.tolist(),
        'n_communities': int(n_comms)
    }

    return details if return_details else inter_edges


def analyze_genre_influence_on_communication(g: ig.Graph, partitions,
                                              return_details: bool = True) -> dict:
    """Analisa se a similaridade de gêneros musicais influencia a comunicação entre comunidades.

    Computa:
    1. Similaridade de gêneros entre pares de comunidades
    2. Volume de arestas entre os mesmos pares
    3. Correlação entre similaridade de gêneros e comunicação

    Parameters
    ----------
    g : ig.Graph
        Grafo com atributo 'genres' nos nós.
    partitions : VertexClustering, lista ou dict
        Partição de comunidades.
    return_details : bool
        Se True, retorna dict completo.

    Returns
    -------
    dict
        - 'genre_similarity': matriz de similaridade de gêneros
        - 'edge_volume': matriz de volume de arestas
        - 'correlation': correlação entre gêneros e comunicação
        - 'genre_diversity_per_community': diversidade de gêneros por comunidade
        - 'analysis': parecer qualitativo
    """
    import numpy as np
    from collections import Counter

    # extrai membership
    if hasattr(partitions, 'membership'):
        membership = list(partitions.membership)
    elif isinstance(partitions, (list, tuple)):
        membership = list(partitions)
    elif isinstance(partitions, dict):
        membership = [partitions.get(v, 0) for v in range(g.vcount())]
    else:
        raise TypeError("'partitions' deve ser VertexClustering, lista ou dict")

    n_comms = max(membership) + 1

    # agrupa gêneros por comunidade
    comms = {}
    for v_idx, c in enumerate(membership):
        comms.setdefault(c, []).append(v_idx)

    comm_genres = {}
    for c, nodes in comms.items():
        all_genres = []
        for v in nodes:
            genres = g.vs[v]['genres']
            if genres:
                all_genres.extend(genres)
        comm_genres[c] = Counter(all_genres)

    # matriz de similaridade Jaccard de gêneros
    genre_similarity = np.zeros((n_comms, n_comms), dtype=float)

    for i in range(n_comms):
        for j in range(i + 1, n_comms):
            genres_i = set(comm_genres[i].keys())
            genres_j = set(comm_genres[j].keys())

            if not genres_i and not genres_j:
                sim = 1.0
            elif not genres_i or not genres_j:
                sim = 0.0
            else:
                intersection = len(genres_i & genres_j)
                union = len(genres_i | genres_j)
                sim = intersection / union if union > 0 else 0.0

            genre_similarity[i, j] = sim
            genre_similarity[j, i] = sim

    # obter volume de arestas inter-comunitárias
    inter_comm = analyze_inter_community_communication(g, membership, return_details=False)

    # correlação entre similaridade de gêneros e volume de arestas
    # remove diagonal e triângulo inferior
    genre_vec = []
    edge_vec = []

    for i in range(n_comms):
        for j in range(i + 1, n_comms):
            genre_vec.append(genre_similarity[i, j])
            edge_vec.append(inter_comm[i, j])

    if len(genre_vec) > 1:
        correlation = float(np.corrcoef(genre_vec, edge_vec)[0, 1])
    else:
        correlation = 0.0

    # diversidade de gêneros por comunidade (usando Shannon entropy)
    genre_diversity = []
    for c in range(n_comms):
        if not comm_genres[c]:
            genre_diversity.append(0.0)
            continue
        total = sum(comm_genres[c].values())
        h = 0.0
        for count in comm_genres[c].values():
            p = count / total
            if p > 0:
                h -= p * np.log(p)
        genre_diversity.append(h)

    # parecer qualitativo
    if np.isnan(correlation):
        analysis = "Correlação indefinida (poucos pares de comunidades)"
    elif abs(correlation) < 0.3:
        analysis = "Influência fraca: gêneros musicais não explicam bem a comunicação"
    elif correlation > 0.5:
        analysis = "Influência forte positiva: comunidades com gêneros similares se comunicam mais"
    elif correlation < -0.5:
        analysis = "Influência forte negativa: comunidades com gêneros diferentes se comunicam mais"
    else:
        analysis = "Influência moderada entre gêneros musicais e comunicação"

    details = {
        'genre_similarity': genre_similarity,
        'edge_volume': inter_comm,
        'correlation': float(correlation) if not np.isnan(correlation) else None,
        'genre_diversity_per_community': genre_diversity,
        'analysis': analysis,
        'n_communities': int(n_comms),
        'genre_correlation_pairs': list(zip(genre_vec, edge_vec))
    }

    return details


def plot_inter_community_communication(g: ig.Graph, partitions,
                                       figsize: tuple = (14, 5),
                                       output_path: str | None = None):
    """Plota a comunicação entre comunidades de forma visual.

    Cria dois plots:
    1. Matriz de calor (heatmap) do volume de arestas
    2. Gráfico de dispersão: similaridade de gêneros vs volume de arestas

    Parameters
    ----------
    g : ig.Graph
        Grafo.
    partitions : VertexClustering, lista ou dict
        Partição de comunidades.
    figsize : tuple
        Tamanho da figura.
    output_path : str | None
        Caminho para salvar a figura.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # analisa comunicação
    comm_analysis = analyze_inter_community_communication(g, partitions, return_details=True)
    inter_edges = comm_analysis['inter_community_edges']

    # analisa influência de gêneros
    genre_analysis = analyze_genre_influence_on_communication(g, partitions, return_details=True)
    genre_sim = genre_analysis['genre_similarity']
    correlation = genre_analysis['correlation']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Heatmap de comunicação ---
    ax1 = axes[0]
    im1 = ax1.imshow(inter_edges, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('Comunidade', fontsize=11)
    ax1.set_ylabel('Comunidade', fontsize=11)
    ax1.set_title('Arestas inter-comunitárias', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='# Arestas')

    # adiciona valores nas células
    for i in range(inter_edges.shape[0]):
        for j in range(inter_edges.shape[1]):
            ax1.text(j, i, f'{int(inter_edges[i, j])}',
                    ha='center', va='center', color='black', fontsize=9)

    # --- Plot 2: Dispersão: Gêneros vs Comunicação ---
    ax2 = axes[1]

    # extrai pares únicos
    pairs_sim = []
    pairs_edges = []
    for i in range(genre_sim.shape[0]):
        for j in range(i + 1, genre_sim.shape[1]):
            pairs_sim.append(genre_sim[i, j])
            pairs_edges.append(inter_edges[i, j])

    if pairs_sim:
        ax2.scatter(pairs_sim, pairs_edges, s=100, alpha=0.6, color='steelblue',
                   edgecolor='black', linewidth=0.8)
        ax2.set_xlabel('Similaridade de Gêneros (Jaccard)', fontsize=11)
        ax2.set_ylabel('Volume de Arestas', fontsize=11)
        ax2.set_title(f'Gêneros vs Comunicação (r={correlation:.3f})', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f"Comunicação Inter-Comunitária - {genre_analysis['analysis']}", 
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")

    plt.show()