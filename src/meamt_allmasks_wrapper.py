from typing import Any

import numpy as np
import random
import sys
import os
import moeabench as mb
from moeabench.progress import get_active_pbar
sys.path.append(os.path.abspath("."))

from src.meamt_allmasks import executa

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
            return int(value) # type: ignore
    return 0


def _limites(valor, n_var):
    v = np.asarray(valor, dtype=float)
    return np.full(n_var, float(v)) if v.ndim == 0 else v.reshape(n_var)



class MULTICORNER(mb.moeas.BaseMoea):
    def __init__(self, problem=None, population=None, generations=None, seed=None):
        super().__init__(problem, population, generations, seed)
        self.name = "MULTICORNER"
    
    def evaluation(self):
        mop = self.get_problem()
        n_gen = self.generations
        n_pop = self.population
        n_obj = mop.M
        n_var = mop.N
        n_constraints = _number_of_constraints(mop)
        xl = _limites(mop.xl, n_var)
        xu = _limites(mop.xu, n_var)
        
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            
        self.F_gens = []
        self.X_gens = []
        self.F_nd_gens = []
        self.X_nd_gens = []
        self.F_dom_gens = []
        self.X_dom_gens = []
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
        
        def snapshot_callback(pool_x, pool_f, fronts):
            self.F_gens.append(pool_f); self.X_gens.append(pool_x)
            self.F_nd_gens.append(pool_f[fronts[0]]); self.X_nd_gens.append(pool_x[fronts[0]])

            if len(fronts) > 1:
                dom_idx = np.concatenate(fronts[1:]).astype(int)
                self.F_dom_gens.append(pool_f[dom_idx]); self.X_dom_gens.append(pool_x[dom_idx])

            pbar = get_active_pbar()
            if pbar:
                pbar.update_to(len(self.F_gens) - 1)
        
        
        F_final = executa(
            avaliar=avaliar,
            n_gen=n_gen,
            mu_fase1 = 0.6,
            n_obj=n_obj,
            snapshot_callback=snapshot_callback,
            n_pop=n_pop,
            n_var=n_var
        )
        F_final = np.array(F_final)
        
        return(
            self.F_gens,
            self.X_gens,
            F_final,
            self.F_nd_gens,
            self.X_nd_gens,
            self.F_dom_gens,
            self.X_dom_gens,
        )    
        
    