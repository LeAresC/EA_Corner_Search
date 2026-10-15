import os
import sys

import numpy as np
import moeabench as mb
from moeabench.progress import get_active_pbar

sys.path.append(os.path.abspath("."))
from src.rvea_cs import execute


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


AJUSTES_VALIDOS = ("forca_premio", "p", "r", "ancora", "recentrar", "alpha", "pc", "eta")


def _nao_dominados(F):
    """Máscara booleana das soluções que ninguém domina (só para separar o histórico)."""
    domina = (F[:, None, :] <= F[None, :, :]).all(axis=2) & (F[:, None, :] < F[None, :, :]).any(axis=2)
    return ~domina.any(axis=0)


class RVEACS(mb.moeas.BaseMoea):
    """RVEA-CS: seleção por vazios angulares com APD modificado (prêmio pela distância ao
    eixo central). O algoritmo está em src/rvea_cs.py.

    Orçamento: population * (generations + 1) avaliações. Todo o orçamento vai para a
    seleção do rvea_cs (phase1 = 1), e T = generations + 1 na agenda do prêmio.

    Uso:
        alg = RVEACS(problem=..., population=..., generations=..., seed=...)
        alg = RVEACS(..., forca_premio=0.5, ancora="p_gap")   # ajustes pontuais
    Ajustes aceitos: forca_premio, p, r, ancora, recentrar, alpha, pc, eta.
    Não trata problemas com restrições.
    """

    def __init__(self, problem=None, population=None, generations=None, seed=None, **ajustes):
        super().__init__(problem, population, generations, seed)
        desconhecidos = set(ajustes) - set(AJUSTES_VALIDOS)
        if desconhecidos:
            raise ValueError("ajustes desconhecidos: %s. Validos: %s"
                             % (", ".join(sorted(desconhecidos)), ", ".join(AJUSTES_VALIDOS)))
        self.ajustes = ajustes
        self.name = "RVEACS"

    def evaluation(self):
        mop = self.get_problem()
        n_obj = mop.M
        n_var = mop.N
        if _number_of_constraints(mop) > 0:
            raise ValueError("o RVEA-CS não trata restrições; use um problema sem restrições")
        xl = _limites(mop.xl, n_var)
        xu = _limites(mop.xu, n_var)

        self.F_gens, self.X_gens = [], []
        self.F_nd_gens, self.X_nd_gens = [], []
        self.F_dom_gens, self.X_dom_gens = [], []
        self.fes_gasto = 0

        def avaliar(X01):
            X_eval = xl + np.asarray(X01) * (xu - xl)
            n = len(X_eval)
            # A chamada a evaluation_benchmark e OBRIGATORIA (contagem de avaliacoes do framework).
            resultado = self.evaluation_benchmark(X_eval)
            F = _normalize_objectives(resultado["F"], n, n_obj)
            self.fes_gasto += n
            return F

        def registra(X01, F):
            X = xl + X01 * (xu - xl)
            nd = _nao_dominados(F)
            self.F_gens.append(F.copy()); self.X_gens.append(X.copy())
            self.F_nd_gens.append(F[nd].copy()); self.X_nd_gens.append(X[nd].copy())
            self.F_dom_gens.append(F[~nd].copy()); self.X_dom_gens.append(X[~nd].copy())
            pbar = get_active_pbar()
            if pbar:
                pbar.update_to(min(len(self.F_gens) - 1, self.generations))

        if self.seed is not None:
            np.random.seed(self.seed)
        X01, F, _ = execute(
            n_gen=self.generations + 1, n_pop=self.population, n_obj=n_obj, n_var=n_var,
            phase1=1.0, evaluate=avaliar, callback=registra, **self.ajustes,
        )

        pbar = get_active_pbar()
        if pbar:
            pbar.update_to(self.generations)

        F_final = F[_nao_dominados(F)]
        return (
            self.F_gens,
            self.X_gens,
            F_final,
            self.F_nd_gens,
            self.X_nd_gens,
            self.F_dom_gens,
            self.X_dom_gens,
        )