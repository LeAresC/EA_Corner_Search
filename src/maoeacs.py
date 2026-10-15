import os
import sys

import numpy as np
import moeabench as mb
from moeabench.progress import get_active_pbar

sys.path.append(os.path.abspath("."))
from src.maoeacs_core import executa, frentes_com_restricao


def _normalize_constraints(constraints, n_individuals):
    G = np.asarray(constraints, dtype=float)

    if G.size == 0 and n_individuals > 0:
        raise ValueError("G vazio para um lote com indivíduos")
    if G.ndim == 0:
        if n_individuals != 1:
            raise ValueError("G escalar só é válido para um indivíduo")
        G = G.reshape(1, 1)
    elif G.ndim == 1:
        if n_individuals == 1:
            G = G.reshape(1, -1)
        elif G.shape[0] == n_individuals:
            G = G.reshape(n_individuals, 1)
        else:
            raise ValueError(
                f"shape de G incompatível: {G.shape} para {n_individuals} indivíduos"
            )
    elif G.ndim == 2:
        if G.shape[0] == n_individuals:
            pass
        elif n_individuals > 1 and G.shape == (1, n_individuals):
            G = G.T
        else:
            raise ValueError(
                f"shape de G incompatível: {G.shape} para {n_individuals} indivíduos"
            )
    else:
        raise ValueError(f"G deve ter no máximo 2 dimensões; recebido ndim={G.ndim}")

    if not np.all(np.isfinite(G)):
        raise ValueError("G contém valores de restrição não finitos")
    return G


def _normalize_objectives(objectives, n_individuals, n_obj):
    F = np.asarray(objectives, dtype=float)
    if F.ndim == 1 and n_individuals == 1:
        F = F.reshape(1, -1)
    if F.ndim != 2 or F.shape != (n_individuals, n_obj):
        raise ValueError(
            f"shape de F incompatível: {F.shape}; esperado {(n_individuals, n_obj)}"
        )
    if not np.all(np.isfinite(F)):
        raise ValueError("F contém valores de objetivo não finitos")
    return F


def _number_of_constraints(problem):
    for name in ("get_n_ieq_constr", "n_ieq_constr"):
        value = getattr(problem, name, None)
        if value is not None:
            value = value() if callable(value) else value
            return int(value)
    return 0


def _limites(valor, n_var):
    v = np.asarray(valor, dtype=float)
    return np.full(n_var, float(v)) if v.ndim == 0 else v.reshape(n_var)


class MAOEACS(mb.moeas.BaseMoea):
    """MaOEA-CS reimplementado a partir do artigo (Cai et al., arXiv:1806.02967).
    Nao e o codigo dos autores; as decisoes tomadas onde o artigo e ambiguo estao
    documentadas em src/maoeacs.py.

    Orcamento: population * (generations + 1) avaliacoes, o mesmo do MEAMT_CANTOS.
    No artigo: populacao 25 * m e orcamento max(100000, 10000 * D).

    Observacao: na fase exploitativa cada geracao gera floor(|P|/|P_c|) * |P_c|
    filhos (ate N), entao o numero de geracoes registradas pode passar de
    `generations`; o orcamento de avaliacoes e respeitado.
    """

    delta = 0.9 
    limiar_troca = None   # None = 0,001 * m, como no artigo
    len_troca = 50

    def __init__(self, problem=None, population=None, generations=None, seed=None):
        super().__init__(problem, population, generations, seed)
        self.name = "MAOEACS"

    def evaluation(self):
        mop = self.get_problem()
        n_obj = mop.M
        n_var = mop.N
        n_constraints = _number_of_constraints(mop)
        xl = _limites(mop.xl, n_var)
        xu = _limites(mop.xu, n_var)

        self.F_gens, self.X_gens = [], []
        self.F_nd_gens, self.X_nd_gens = [], []
        self.F_dom_gens, self.X_dom_gens = [], []
        self.fes_gasto = 0

        def avaliar(X01):
            X_eval = xl + np.asarray(X01) * (xu - xl)
            n = len(X_eval)
            # A chamada a evaluation_benchmark e OBRIGATORIA (contagem de FES do framework).
            resultado = self.evaluation_benchmark(X_eval)
            F = _normalize_objectives(resultado["F"], n, n_obj)
            if "G" in resultado:
                G = _normalize_constraints(resultado["G"], n)
                CV = np.maximum(G, 0.0).sum(axis=1)
            elif n_constraints > 0:
                raise ValueError("problema restrito não retornou G em evaluation_benchmark")
            else:
                CV = np.zeros(n, dtype=float)
            self.fes_gasto += n
            return F, CV

        def snapshot_callback(X01, F, CV, rank, n_cantos):
            X = xl + X01 * (xu - xl)
            nd = rank == 0
            self.F_gens.append(np.asarray(F).copy()); self.X_gens.append(X.copy())
            self.F_nd_gens.append(F[nd].copy()); self.X_nd_gens.append(X[nd].copy())
            self.F_dom_gens.append(F[~nd].copy()); self.X_dom_gens.append(X[~nd].copy())
            pbar = get_active_pbar()
            if pbar:
                pbar.update_to(min(len(self.F_gens) - 1, self.generations))

        X01, F, CV, info = executa(
            avaliar,
            n_obj=n_obj,
            n_var=n_var,
            N=self.population,
            max_fe=self.population * (self.generations + 1),
            seed=self.seed,
            delta=self.delta,
            limiar=self.limiar_troca,
            len_=self.len_troca,
            callback=snapshot_callback,
        )
        self.info = info

        rank = frentes_com_restricao(F, CV)
        F_final = F[rank == 0]

        return (
            self.F_gens,
            self.X_gens,
            F_final,
            self.F_nd_gens,
            self.X_nd_gens,
            self.F_dom_gens,
            self.X_dom_gens,
        )