"""
Robustez simulada para grafos (iGraph)

Funções principais:
  - simulated_robustness(g, strategy='targeted'|'random', step=1, random_trials=20, sample_frac=None, seed=None)
      roda uma simulação única ou múltiplas (para random) e retorna:
        - fractions_removed: array com frações de nós removidos (0..1)
        - giant_sizes: array com tamanho da maior componente normalizado (0..1) correspondendo a fractions_removed
        - auc: área sob a curva giant_sizes vs fractions_removed (valor entre 0 e 1)

  - plot_robustness_curves(result_targeted, result_random_mean, out_path=None)
      plota as curvas e salva imagem se out_path fornecido.

  - compute_and_plot_robustness(g, ...)
      wrapper que executa both targeted and random simulations, plota, imprime resumo e retorna dicionário.
"""

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import Optional, Dict, Tuple, List

# ========== utilitário: aceita networkx.Graph convertendo para igraph se necessário ==========
def ensure_igraph_graph(g):
    """Se g for networkx.Graph, converte para igraph.Graph; caso contrário assume igraph.Graph."""
    try:
        import networkx as nx
    except Exception:
        nx = None
    if nx is not None and isinstance(g, nx.Graph):
        # converter via edges + nodelist para manter rótulos
        return ig.Graph.TupleList(g.edges(), directed=False)
    if isinstance(g, ig.Graph):
        return g
    raise TypeError("O grafo deve ser igraph.Graph ou networkx.Graph (se networkx estiver instalado).")

# ========== utilitário: AUC via trapézio ==========
def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Computa a área sob a curva y(x) no intervalo [0,1] usando trapézio.
       x deve estar ordenado e ir de 0..1 (ou será normalizado).
    """
    if len(x) < 2:
        return float('nan')
    # normalizar x para 0..1 se necessário
    x0 = (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else x
    return float(np.trapz(y, x0))

# ========== simulação principal ==========
def simulated_robustness(g,
                         strategy: str = "targeted",
                         step: int = 1,
                         random_trials: int = 20,
                         sample_frac: Optional[float] = None,
                         seed: Optional[int] = None) -> Dict:
    """
    Simula remoção progressiva de nós e mede o tamanho da componente gigante.
    Parâmetros:
      - g: igraph.Graph (ou networkx.Graph)
      - strategy: 'targeted' (remover por grau decrescente) ou 'random' (ordem aleatória)
      - step: quantos nós remover por iteração (1 = remoção node-a-node; >1 para acelerar)
      - random_trials: se strategy == 'random', quantos trials aleatórios rodar (retorna média e std)
      - sample_frac: se fornecido (0<sample_frac<=1), executa simulação removendo apenas até essa fração do grafo (útil para performance)
      - seed: seed para reprodutibilidade (usado em random)
    Retornos (dicionário):
      - 'fractions': array (fração removida) de comprimento T+1 (começa em 0.0)
      - 'giant_sizes_mean': média (sobre trials para random) ou a curva única para targeted
      - 'giant_sizes_std': std (para random) ou zeros
      - 'auc_mean', 'auc_std' (se random) ou auc para targeted
      - 'all_trials' : lista de arrays (cada trial giant sizes) — opcionalmente para análise
    Observações:
      - A simulação recalcula components subgrafo = g.subgraph(remaining_nodes) a cada passo.
      - Para grafos grandes, use step > 1 e/ou sample_frac < 1 para acelerar.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    g = ensure_igraph_graph(g)
    n = g.vcount()
    if n == 0:
        raise ValueError("Grafo vazio.")

    # numero máximo de remoções a simular
    max_remove = int(math.floor(n * sample_frac)) if (sample_frac is not None and 0 < sample_frac <= 1.0) else n

    # gera array de frações correspondentes (inclusive 0.0 inicial)
    steps = list(range(0, max_remove + 1, step))
    if steps[-1] != max_remove:
        steps.append(max_remove)
    fractions = np.array([s / n for s in steps], dtype=float)

    def run_single_order(order_list: List[int]) -> np.ndarray:
        """Dada uma ordem de remoção (lista com ordem de nós a remover), retorna curva giant_sizes normalizados."""
        remaining = set(range(n))
        giant_sizes = []
        # initial giant
        comps0 = g.subgraph(list(remaining)).components()
        giant_sizes.append(comps0.giant().vcount() / n)
        removed_so_far = 0
        idx = 0
        # percorre steps: remove up to steps[-1] nodes
        for target_removals in steps[1:]:
            # remove nodes until removed_so_far == target_removals
            while removed_so_far < target_removals and idx < len(order_list):
                v = order_list[idx]
                if v in remaining:
                    remaining.remove(v)
                    removed_so_far += 1
                idx += 1
            if len(remaining) == 0:
                giant_sizes.append(0.0)
            else:
                sub = g.subgraph(list(remaining))
                comps = sub.components()
                giant_sizes.append(comps.giant().vcount() / n)
        return np.array(giant_sizes, dtype=float)

    # ===== strategy targeted =====
    if strategy == "targeted":
        # ordem por grau decrescente no grafo original (usamos degree; se houver peso poderia usar strength)
        degs = g.degree()
        # ordena nós por grau decrescente; quebra empates por índice
        order = sorted(range(n), key=lambda v: (-degs[v], v))
        giant_curve = run_single_order(order)
        auc = auc_trapezoid(fractions, giant_curve)
        return {
            "fractions": fractions,
            "giant_sizes_mean": giant_curve,
            "giant_sizes_std": np.zeros_like(giant_curve),
            "auc_mean": auc,
            "auc_std": 0.0,
            "all_trials": [giant_curve]
        }

    # ===== strategy random =====
    elif strategy == "random":
        all_curves = []
        for t in range(random_trials):
            order_rand = list(range(n))
            random.shuffle(order_rand)
            curve = run_single_order(order_rand)
            all_curves.append(curve)
        all_curves = np.array(all_curves)  # shape (trials, len(fractions))
        mean_curve = all_curves.mean(axis=0)
        std_curve = all_curves.std(axis=0, ddof=0)
        aucs = [auc_trapezoid(fractions, all_curves[i]) for i in range(all_curves.shape[0])]
        return {
            "fractions": fractions,
            "giant_sizes_mean": mean_curve,
            "giant_sizes_std": std_curve,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs, ddof=0)),
            "all_trials": all_curves
        }
    else:
        raise ValueError("strategy deve ser 'targeted' ou 'random'.")

# ========== função de plotagem ==========
def plot_robustness_curves(res_targeted: Dict,
                           res_random: Dict,
                           figsize: Tuple[int,int]=(8,6),
                           out_path: Optional[str]=None,
                           title: Optional[str]=None,
                           show_legend: bool=True):
    """
    Plota curvas de robustez: targeted vs random(mean ± std).
      - res_targeted: retorno de simulated_robustness(strategy='targeted')
      - res_random: retorno de simulated_robustness(strategy='random')
    """
    fr = res_targeted['fractions']
    fig, ax = plt.subplots(figsize=figsize)
    # targeted
    ax.plot(fr, res_targeted['giant_sizes_mean'], label=f"Targeted (AUC={res_targeted['auc_mean']:.3f})", linewidth=2, zorder=3)
    # random mean
    ax.plot(fr, res_random['giant_sizes_mean'], label=f"Random mean (AUC={res_random['auc_mean']:.3f})", linestyle='--', linewidth=2, zorder=2)
    # random std shaded
    ax.fill_between(fr,
                    res_random['giant_sizes_mean'] - res_random['giant_sizes_std'],
                    res_random['giant_sizes_mean'] + res_random['giant_sizes_std'],
                    alpha=0.25, label="Random ± std", zorder=1)
    ax.set_xlabel("Fração de nós removidos")
    ax.set_ylabel("Tamanho da maior componente (fração dos nós originais)")
    ax.set_title(title or "Robustez: Targeted vs Random")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

# ========== wrapper que executa ambos e imprime resumo ==========
def compute_and_plot_robustness(g,
                                step: int = 1,
                                random_trials: int = 20,
                                sample_frac: Optional[float] = None,
                                seed: Optional[int] = None,
                                out_plot_path: Optional[str] = None):
    """
    Executa simulações targeted + random e plota/retorna resultados.
    Parâmetros:
      - g: igraph.Graph ou networkx.Graph
      - step: quantos nós remover por iteração (para performance, use step > 1)
      - random_trials: N trials para random
      - sample_frac: rodar apenas até essa fração de remoção (ex: 0.5) para economizar tempo
      - seed: seed para aleatoriedade
      - out_plot_path: se informado, salva figura
    Retorna dicionário com res_targeted, res_random e resumo.
    """
    print("Executando simulação de robustez...")
    res_t = simulated_robustness(g, strategy='targeted', step=step, sample_frac=sample_frac, seed=seed)
    print("50%% complete...")
    res_r = simulated_robustness(g, strategy='random', step=step, random_trials=random_trials, sample_frac=sample_frac, seed=seed)

    print("\n--- Resumo ---")
    print(f"AUC (Targeted by degree): {res_t['auc_mean']:.4f}  (quanto maior, mais robusta)")
    print(f"AUC (Random) mean ± std : {res_r['auc_mean']:.4f} ± {res_r['auc_std']:.4f}")
    print("Interpretação: se Targeted << Random, a rede é vulnerável a ataques em hubs (esperado em redes heterogêneas).")

    plot_robustness_curves(res_t, res_r, out_path=out_plot_path)

    return {"targeted": res_t, "random": res_r}

# ========== exemplo de uso ==========
# if __name__ == "__main__":
#     # exemplo rápido (se tiver igraph)
#     try:
#         g = ig.Graph.Famous("Zachary")
#     except Exception:
#         # fallback: gera Erdos-Renyi
#         g = ig.Graph.Erdos_Renyi(n=100, p=0.06)
#     # executa (use step>1 para grafos maiores; sample_frac=0.5 para rodar até metade)
#     results = compute_and_plot_robustness(g, step=1, random_trials=30, sample_frac=None, seed=42, out_plot_path="robustness.png")
#     # results contém curvas e AUCs
#     print("Done. Resultados retornados em 'results'.")

