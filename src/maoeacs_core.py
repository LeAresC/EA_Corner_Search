"""MaOEA-CS -- reimplementacao a partir do artigo, o mais fiel possivel.

Cai, X.; Sun, H.; Zhu, C.; Li, Z.; Zhang, Q. Locating the boundaries of Pareto
fronts: a many-objective evolutionary algorithm based on corner solution search.
arXiv:1806.02967, 2018.

NAO e o codigo dos autores. Cada passo segue o texto; os pontos em que o artigo e
ambiguo ou omisso estao marcados com [DECISAO]:

 [DECISAO 1] ABS, Algoritmo 6, linha 14: o artigo escreve theta_i = max{...}, mas
     define theta_i como o angulo MINIMO ate o vizinho mais proximo ja selecionado.
     So "min" e consistente com essa definicao; usamos min.
 [DECISAO 2] Troca de delta (Eq. 11): o texto nao diz se ocorre uma unica vez. Se
     fosse reavaliada a cada geracao, delta oscilaria entre 0,9 e 0,1. Fazemos uma
     unica troca.
 [DECISAO 3] Mutacao exploitativa (Eq. 10): implementada exatamente como impressa,
     com os dois "rand" independentes; valores fora dos limites sao cortados (o
     artigo nao diz o que fazer).
 [DECISAO 4] Busca exploratoria: o artigo nao especifica a selecao de pais; usamos
     pares sorteados da populacao.
 [DECISAO 5] "Solucoes mais proximas de z*" (Algoritmo 5): distancia euclidiana nos
     objetivos brutos, como esta escrito.
 [DECISAO 6] Restricoes: o artigo nao as trata. Como extensao, a ordenacao nao-
     dominada usa a regra de Deb (factivel vence infactivel; entre infactiveis, a
     menor violacao). Sem restricoes (CV = 0), o comportamento e o do artigo.
 [DECISAO 7] Divisao por zero: protegida com 1e-12 na Eq. 11 e na Eq. 13.

Parametros do artigo (Secao IV-B): delta = 0,9; SBX com p_c = 1 e eta_c = 20;
mutacao polinomial com p_m = 1/D e eta_m = 20; limiar de troca 0,001*m; len = 50;
populacao 25*m; orcamento max(100000, 10000*D).
"""
import numpy as np


# ------------------------------------------------------------------ utilitarios
def frentes(F):
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=int)
    le = (F[:, None, :] <= F[None, :, :]).all(2)
    lt = (F[:, None, :] < F[None, :, :]).any(2)
    dom = le & lt
    cnt = dom.sum(0)
    rank = np.full(n, -1)
    r, atual = 0, np.flatnonzero(cnt == 0)
    while len(atual):
        rank[atual] = r
        cnt = cnt - dom[atual].sum(0)
        cnt[rank >= 0] = -1
        atual = np.flatnonzero(cnt == 0)
        r += 1
    return rank


def frentes_com_restricao(F, CV):          # [DECISAO 6]
    rank = np.empty(len(F), dtype=int)
    fac = CV <= 0
    if fac.any():
        rank[fac] = frentes(F[fac])
        base = rank[fac].max() + 1
    else:
        base = 0
    if (~fac).any():
        _, nivel = np.unique(CV[~fac], return_inverse=True)
        rank[~fac] = base + nivel
    return rank


def sbx(A, B, rng, eta=20.0, pc=1.0):
    """SBX com limites (Deb), dominio [0, 1]."""
    x1, x2 = np.minimum(A, B), np.maximum(A, B)
    dx = np.maximum(x2 - x1, 1e-14)
    u = rng.random(A.shape)

    def bq(beta):
        alpha = 2.0 - beta ** -(eta + 1)
        return np.where(u <= 1 / alpha, (u * alpha) ** (1 / (eta + 1)), (1 / (2 - u * alpha)) ** (1 / (eta + 1)))

    c1 = np.clip(0.5 * (x1 + x2 - bq(1 + 2 * x1 / dx) * dx), 0, 1)
    c2 = np.clip(0.5 * (x1 + x2 + bq(1 + 2 * (1 - x2) / dx) * dx), 0, 1)
    tr = rng.random(A.shape) < 0.5
    c1, c2 = np.where(tr, c2, c1), np.where(tr, c1, c2)
    faz = (rng.random(A.shape) < 0.5) & (np.abs(A - B) > 1e-14) & (rng.random(A.shape[:-1] + (1,)) < pc)
    return np.where(faz, c1, A), np.where(faz, c2, B)


def mutacao_polinomial(X, rng, eta=20.0):
    """Mutacao polinomial com limites (Deb), p_m = 1/D."""
    m = rng.random(X.shape) < 1.0 / X.shape[-1]
    u = rng.random(X.shape)
    dq1 = (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dq2 = 1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(np.where(m, X + np.where(u < 0.5, dq1, dq2), X), 0, 1)


def mutacao_exploitativa(X, rng, fe, max_fe):
    """Eq. (10): x_c = x + rnd * (ub - lb), com rnd = 0,5 (rand - 0,5)(1 - rand^alpha)
    e alpha = 0,7 * (-(1 - fe/max_fe)); cada componente mutado com p_m = 1/D.
    Dominio interno [0, 1], entao ub - lb = 1. [DECISAO 3]"""
    alpha = 0.7 * (-(1.0 - fe / max_fe))
    m = rng.random(X.shape) < 1.0 / X.shape[-1]
    r1 = rng.random(X.shape)
    r2 = np.maximum(rng.random(X.shape), 1e-300)
    rnd = 0.5 * (r1 - 0.5) * (1.0 - r2 ** alpha)
    return np.clip(np.where(m, X + rnd, X), 0, 1)


# ------------------------------------------------------------------ Algoritmo 1
def busca_de_cantos(F):
    """Corner Solution Search. F: objetivos de um conjunto nao-dominado.
    Devolve (indices dos cantos em F, nadir estimado pela Eq. 9)."""
    d = np.sqrt(np.maximum((F ** 2).sum(1, keepdims=True) - F ** 2, 0))   # dist perpendicular a cada eixo
    P1 = list(dict.fromkeys(int(i) for i in np.argmin(d, 0)))              # Eq. (7)
    nad = F[P1].max(0)                                                     # Eq. (9) sobre P1
    Pc = list(P1)
    for j in np.argmin(F, 0):                                              # Eq. (8)
        if (F[j] > nad).any() and int(j) not in Pc:
            Pc.append(int(j))
    Pc = np.array(Pc)
    return Pc, F[Pc].max(0)                                                # nadir = max sobre P_c (Eq. 9)


# ------------------------------------------------------------------ Algoritmo 6
def abs_selecao(Fq, idx_cantos, zst, znad, N):
    """Angle-based selection. Fq: objetivos do conjunto Q; idx_cantos: posicoes
    dos cantos em Q. Devolve as posicoes selecionadas (N)."""
    Fn = (Fq - zst) / np.maximum(znad - zst, 1e-12)                        # Eq. (13)   [DECISAO 7]
    U = Fn / np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-12)
    sel = list(dict.fromkeys(int(i) for i in idx_cantos))
    if len(sel) >= N:
        return np.array(sel[:N])
    theta = np.arccos(np.clip(U @ U[sel].T, -1, 1)).min(1)                 # Eq. (14), linhas 5-7
    theta[sel] = -1.0
    while len(sel) < N:
        k = int(np.argmax(theta))
        sel.append(k)
        theta = np.minimum(theta, np.arccos(np.clip(U @ U[k], -1, 1)))     # [DECISAO 1]: min
        theta[sel] = -1.0
    return np.array(sel)


# ------------------------------------------------------------------ Algoritmo 5
def dsa_selecao(RX, RF, RC, N):
    rank = frentes_com_restricao(RF, RC)
    r1 = np.flatnonzero(rank == 0)
    F1 = RF[r1]
    zst = F1.min(0)                                                        # Eq. (12)
    pc_loc, znad = busca_de_cantos(F1)                                     # linhas 3-4
    Pc = r1[pc_loc]
    if len(r1) > N:
        fora = (F1 > znad).any(1)
        p_out, p_in = r1[fora], r1[~fora]
        if len(p_in) > N:
            pos_cantos = [int(np.flatnonzero(p_in == c)[0]) for c in Pc if c in p_in]
            idx = p_in[abs_selecao(RF[p_in], pos_cantos, zst, znad, N)]
        elif len(p_in) < N:
            d = np.linalg.norm(RF[p_out] - zst, axis=1)                    # [DECISAO 5]
            idx = np.concatenate([p_in, p_out[np.argsort(d)[:N - len(p_in)]]])
        else:
            idx = p_in
    elif len(r1) < N:
        resto = np.setdiff1d(np.arange(len(RF)), r1)
        d = np.linalg.norm(RF[resto] - zst, axis=1)                        # [DECISAO 5]
        idx = np.concatenate([r1, resto[np.argsort(d)[:N - len(r1)]]])
    else:
        idx = r1
    return idx, Pc, znad


# ------------------------------------------------------------------ Algoritmo 2
def executa(avaliar, n_obj, n_var, N, max_fe, seed=None, delta=0.9, limiar=None, len_=50,
            callback=None):
    """avaliar(X01) -> (F, CV), com X01 em [0,1]^n_var.
    Devolve (X01, F, CV) da populacao final e um dicionario com a geracao da troca de delta."""
    M, D = n_obj, n_var
    rng = np.random.default_rng(seed)
    limiar = 0.001 * M if limiar is None else limiar
    fe = 0

    def aval(X):
        nonlocal fe
        F, CV = avaliar(X)
        fe += len(X)
        return np.asarray(F, float), np.asarray(CV, float).reshape(-1)

    # ---- Algoritmo 3: inicializacao
    X = rng.random((N, D))
    F, CV = aval(X)
    rank = frentes_com_restricao(F, CV)
    nd = np.flatnonzero(rank == 0)                                         # P = NONDOMINATED-SELECTION(P)
    X, F, CV = X[nd], F[nd], CV[nd]
    pc_loc, znad = busca_de_cantos(F)
    Xc = X[pc_loc]

    historico_nadir = [znad.copy()]
    trocou, ger, ger_troca = False, 0, None
    while True:
        # ---- Algoritmo 4: reproducao
        if rng.random() < delta:
            reps = len(X) // len(Xc)                                        # floor(|P| / |P_c|)
            if reps == 0 or fe + reps * len(Xc) > max_fe:
                break
            QX = mutacao_exploitativa(np.repeat(Xc, reps, axis=0), rng, fe, max_fe)
        else:
            if fe + N > max_fe:
                break
            pa, pb = rng.integers(0, len(X), (2, (N + 1) // 2))            # [DECISAO 4]
            c1, c2 = sbx(X[pa], X[pb], rng)
            QX = mutacao_polinomial(np.vstack([c1, c2]), rng)[:N]
        QF, QC = aval(QX)

        # ---- Algoritmo 5: selecao DSA sobre R = P U Q
        RX = np.vstack([X, QX]); RF = np.vstack([F, QF]); RC = np.concatenate([CV, QC])
        RX, ui = np.unique(RX, axis=0, return_index=True)
        RF, RC = RF[ui], RC[ui]
        idx, Pc, znad = dsa_selecao(RX, RF, RC, N)
        X, F, CV = RX[idx], RF[idx], RC[idx]
        Xc = RX[Pc]
        ger += 1

        # ---- Eq. (11): troca de delta
        historico_nadir.append(znad.copy())
        if not trocou and len(historico_nadir) > len_:
            ant = historico_nadir[-1 - len_]
            dt = np.max(np.abs(znad - ant) / np.maximum(np.abs(ant), 1e-12))  # [DECISAO 7]
            if dt < limiar:
                delta = 1.0 - delta                                         # [DECISAO 2]
                trocou, ger_troca = True, ger
        if callback is not None:
            callback(X, F, CV, frentes_com_restricao(F, CV), len(Pc))
    return X, F, CV, dict(geracoes=ger, troca_delta=ger_troca, avaliacoes=fe)