import networkx as nx
import leidenalg
import igraph as ig
from collections import defaultdict
from itertools import combinations
import numpy as np

# ============================== Função para Construir Grafo ==============================
def construir_grafo(conexoes):
    """
    Cria um grafo NetworkX a partir de um dicionário de conexões entre áreas.

    Parâmetros:
        conexoes (dict): Chaves são tuplas de áreas (nós), e valores são os pesos (quantidade de conexões).

    Retorno:
        G (Graph): Grafo NetworkX com os nós e arestas ponderadas.
    """
    G = nx.Graph()
    areas = set([a for pair in conexoes for a in pair])  # Lista única de áreas envolvidas nas conexões
    G.add_nodes_from(areas)  # Adiciona os nós

    # Adiciona as arestas com pesos
    for (a1, a2), peso in conexoes.items():
        G.add_edge(a1, a2, weight=peso)

    return G

# ============================== Grafos por Ano ==============================
def construir_grafos_por_ano(conexoes_por_ano):
    """
    Cria um grafo separado para cada ano.

    Parâmetros:
        conexoes_por_ano (dict): Dicionário com ano como chave e conexões como valor.

    Retorno:
        grafos (dict): Dicionário com ano como chave e grafo NetworkX como valor.
    """
    grafos = {}
    for ano, conexoes in conexoes_por_ano.items():
        grafos[ano] = construir_grafo(conexoes)
    return grafos

def gerar_pares_por_ano(df):
    """
    Gera os pares de áreas conectadas por crime para cada ano.

    Parâmetros:
        df (DataFrame): Dados processados de ocorrências.

    Retorno:
        grafos_por_ano (dict): Um grafo NetworkX para cada ano.
        contagem_por_ano (dict): Contagem de registros por área, por ano e total.
    """
    df["ANO"] = df["DATE OCC"].dt.year  # Cria uma nova coluna com o ano da ocorrência
    conexoes_por_ano = {}
    contagem_por_ano = {}
    contagem_por_ano["total"] = defaultdict(int)  # Contador total acumulado

    # Agrupa o DataFrame por ano
    for ano, df_ano in df.groupby("ANO"):
        grupos = df_ano.groupby(["Mocodes", "Crm Cd Desc"])  # Agrupa por tipo de crime
        conexoes = defaultdict(int)
        contagem_por_ano[ano] = defaultdict(int)

        # Para cada grupo de crime
        for _, grupo in grupos:
            areas = grupo["AREA NAME"]
            
            # Conta quantas vezes cada área aparece no grupo
            for area in areas:
                contagem_por_ano[ano][area] += 1
                contagem_por_ano["total"][area] += 1

            # Gera todas as combinações possíveis de áreas para formar as conexões
            for a1, a2 in combinations(sorted(areas), 2):
                conexoes[(a1, a2)] += 1

        conexoes_por_ano[ano] = conexoes

    return construir_grafos_por_ano(conexoes_por_ano), contagem_por_ano

# ============================== Grafo Total ==============================
def gerar_pares_total(df):
    """
    Cria o grafo total considerando todos os anos juntos.

    Parâmetros:
        df (DataFrame): Dados processados.

    Retorno:
        grafo_total (Graph): Grafo NetworkX único para o dataset completo.
    """
    grupos = df.groupby(["Mocodes", "Crm Cd Desc", df["DATE OCC"].dt.date])  # Agrupa por dia e tipo de crime
    conexoes_total = defaultdict(int)

    # Cria conexões entre áreas no mesmo grupo
    for _, grupo in grupos:
        areas = grupo["AREA NAME"]
        for a1, a2 in combinations(sorted(areas), 2):
            conexoes_total[(a1, a2)] += 1

    return construir_grafo(conexoes_total)

# ============================== Aplicar Leiden ==============================
def aplicar_leiden_e_analisar(grafo_nx, contagem_por_area):
    """
    Converte o grafo NetworkX para iGraph e aplica o algoritmo Leiden de detecção de comunidades.

    Parâmetros:
        grafo_nx (Graph): Grafo original NetworkX.
        contagem_por_area (dict): Número de ocorrências por área.

    Retorno:
        particao: Objeto de partição retornado pelo Leiden.
        dados_comunidades (dict): Dados analíticos por comunidade detectada.
    """
    # Converte grafo NetworkX para iGraph
    grafo_ig = ig.Graph()
    grafo_ig.add_vertices(list(grafo_nx.nodes))
    grafo_ig.add_edges(list(grafo_nx.edges))

    # Define os pesos das arestas
    pesos = [grafo_nx[u][v].get('weight', 1.0) for u, v in grafo_nx.edges]
    grafo_ig.es['weight'] = pesos

    # Aplica o algoritmo Leiden
    particao = leidenalg.find_partition(
        grafo_ig,
        leidenalg.RBConfigurationVertexPartition, # Parâmetro que controla o método de separação das comunidades e extração da modularidade
        resolution_parameter=1,  # Quanto maior, mais comunidades
        weights='weight',
        seed=42 # Deixa a partição determinística, ideal para análise
    )

    # Mapeamento de área para comunidade
    comunidades = {grafo_ig.vs[i]['name']: cid for i, cid in enumerate(particao.membership)}

    # Inicialização dos dicionários para análise
    pesos_comunidade = {}
    registros_comunidade = {}
    dados_comunidades = {}

    # Conta o número de registros por comunidade
    for area, cid in comunidades.items():
        registros_comunidade[cid] = registros_comunidade.get(cid, 0) + contagem_por_area.get(area, 0)

    # Calcula o peso interno das comunidades (soma dos pesos das arestas internas)
    for eid in range(len(grafo_ig.es)):
        v1 = grafo_ig.vs[grafo_ig.es[eid].source]['name']
        v2 = grafo_ig.vs[grafo_ig.es[eid].target]['name']
        w = grafo_ig.es[eid]['weight']
        c1 = comunidades[v1]
        c2 = comunidades[v2]
        if c1 == c2:
            pesos_comunidade[c1] = pesos_comunidade.get(c1, 0) + w

    # Prepara os dados finais por comunidade
    for cid in pesos_comunidade:
        peso_total = pesos_comunidade[cid]
        total_registros = registros_comunidade.get(cid, 1)
        proporcao = peso_total / total_registros if total_registros > 0 else 0

        # Detalhes por área dentro da comunidade
        areas_comunidade = sorted(set(area for area, com_id in comunidades.items() if com_id == cid))
        dados_area = {}

        for area in sorted(areas_comunidade):
            grau_ponderado = grafo_nx.degree(area, weight='weight') if grafo_nx.has_node(area) else 0
            registros_area = contagem_por_area.get(area, 0)
            dados_area[area] = {"registros_area": registros_area,
                                "grau_ponderado": grau_ponderado}
        
        dados_comunidades[cid] = {'peso_total': peso_total,
                                  "total_registros": total_registros,
                                  "proporcao": proporcao,
                                  "dados_area": dados_area}

    return particao, dados_comunidades

def layout_por_comunidade(grafo, particao, escala_comunidade=5, escala_local=1):
    """
    Gera um layout para o grafo onde as comunidades ficam agrupadas visualmente.

    Parâmetros:
        grafo (Graph): Grafo NetworkX.
        particao: Resultado do Leiden (partição das comunidades).
        escala_comunidade (float): Controle de espaçamento entre comunidades.
        escala_local (float): Controle de espaçamento dentro de cada comunidade.

    Retorno:
        pos_final (dict): Posições finais dos nós para plotagem.
    """
    comunidades = {}
    for node, cid in zip(grafo.nodes, particao.membership):
        comunidades.setdefault(cid, []).append(node)

    # Cria o meta-grafo (um nó por comunidade)
    meta_grafo = nx.Graph()
    for cid in comunidades:
        meta_grafo.add_node(cid)
    for u, v in grafo.edges:
        cu = particao.membership[list(grafo.nodes).index(u)]
        cv = particao.membership[list(grafo.nodes).index(v)]
        if cu != cv:
            meta_grafo.add_edge(cu, cv)

    # Calcula layout entre comunidades
    pos_comunidades = nx.spring_layout(meta_grafo, scale=escala_comunidade, seed=42)

    # Calcula layout dentro de cada comunidade
    pos_final = {}
    for cid, nodes in comunidades.items():
        subgrafo = grafo.subgraph(nodes)
        centro = pos_comunidades[cid]
        pos_local = nx.spring_layout(subgrafo, scale=escala_local, seed=42)
        for n in subgrafo.nodes:
            offset = np.array(pos_local[n])
            pos_final[n] = centro + offset

    return pos_final