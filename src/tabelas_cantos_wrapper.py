import os
import sys

import numpy as np
import moeabench as mb
from moeabench.progress import get_active_pbar

sys.path.append(os.path.abspath("."))
from src.tabelas_cantos import executa, frentes_com_restricao


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


AJUSTES_VALIDOS = ("K", "rho", "p_aleatorio", "frac_fase1", "frac_tabelas", "eixo_variavel", "intervalo_eixo")


class TabelasCantos(mb.moeas.BaseMoea):
    """Busca de cantos por tabelas adaptativas (fase 1) e distribuição por ângulo (fase 2).
    O algoritmo está em src/tabelas_cantos.py.

    Orçamento: population * (generations + 1) avaliações. A fase 1 usa até frac_fase1 dele.

    Uso:
        alg = TabelasCantos(problem=..., population=..., generations=..., seed=...)
        alg = TabelasCantos(..., K=8, frac_fase1=0.6)       # ajustes pontuais
    Ajustes aceitos: K, rho, p_aleatorio, frac_fase1, frac_tabelas, eixo_variavel, intervalo_eixo.
    eixo_variavel=True reestima o eixo central a cada intervalo_eixo do orçamento (padrão: eixo fixo).
    Trata restrições (factível primeiro). Depois da execução, alg.info traz as âncoras
    permanentes (permanentes_X, permanentes_F), o melhor de cada tabela (ancoras_X,
    ancoras_F), o fim da fase 1 (fim_fase1, em avaliações) e as avaliações gastas.
    """

    def __init__(self, problem=None, population=None, generations=None, seed=None, **ajustes):
        super().__init__(problem, population, generations, seed)
        desconhecidos = set(ajustes) - set(AJUSTES_VALIDOS)
        if desconhecidos:
            raise ValueError("ajustes desconhecidos: %s. Validos: %s"
                             % (", ".join(sorted(desconhecidos)), ", ".join(AJUSTES_VALIDOS)))
        self.ajustes = ajustes
        self.name = "TabelasCantos"

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
            # A chamada a evaluation_benchmark e OBRIGATORIA (contagem de avaliacoes do framework).
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

        def registra(X01, F, CV):
            X = xl + np.asarray(X01) * (xu - xl)
            nd = frentes_com_restricao(F, CV) == 0
            self.F_gens.append(np.asarray(F).copy()); self.X_gens.append(X.copy())
            self.F_nd_gens.append(F[nd].copy()); self.X_nd_gens.append(X[nd].copy())
            self.F_dom_gens.append(F[~nd].copy()); self.X_dom_gens.append(X[~nd].copy())
            pbar = get_active_pbar()
            if pbar:   # progresso pelas avaliacoes gastas, convertidas em geracoes
                pbar.update_to(min(max(self.fes_gasto // self.population - 1, 0), self.generations))

        X01, F, CV, self.info = executa(
            avaliar, n_obj=n_obj, n_var=n_var, N=self.population, n_ger=self.generations,
            seed=self.seed, callback=registra, **self.ajustes,frac_fase1=0.5,
        )

        pbar = get_active_pbar()
        if pbar:       # fecha a barra em 100%
            pbar.update_to(self.generations)

        F_final = F[frentes_com_restricao(F, CV) == 0]
        return (
            self.F_gens,
            self.X_gens,
            F_final,
            self.F_nd_gens,
            self.X_nd_gens,
            self.F_dom_gens,
            self.X_dom_gens,
        )