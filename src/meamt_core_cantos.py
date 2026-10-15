"""CCSGA -- Continuous Corner Search Genetic Algorithm (versao congelada).

Tres fases, com o orcamento de avaliacoes dividido entre elas:

Fase 1 -- convergencia (orcamento adaptativo, ate frac_fase1_max)
    As 2m mascaras de nivel 1 e m-1 evoluem em paralelo, com SBX e mutacao
    polinomial. Cada mascara S minimiza
        V0(x) = max_{i em S} f_i(x) + rho * soma_i f_i(x)
    (objetivos medidos a partir de min(0, menor valor observado)). As mascaras
    cooperam de duas formas: filhos compartilhados (cada mascara escolhe os
    sobreviventes entre os seus individuos e todos os filhos da geracao, pelo seu
    criterio) e pais cruzados (em frac_pais_cruzados dos cruzamentos, o segundo pai
    vem de outra mascara). A fase termina quando todas as mascaras passam W geracoes
    sem melhorar o melhor V0 em mais de tol_fase1 (relativo), ou em frac_fase1_max do
    orcamento; do restante, parte_fase2 vai para a fase 2 e o resto para a fase 3.
    Os melhores de cada mascara em V0 sao os cantos-base.

Fase 2 -- busca de cantos (ate frac_fase2 do orcamento)
    * Candidatos: ate m ao mesmo tempo. Cada um e o nao-dominado de maior soma de
      angulos aos cantos atuais, a mais de tau graus dos cantos e dos outros
      candidatos.
    * Filhos de cada geracao:
        - frac_candidatos: divididos entre os candidatos; de cada um, metade por
          cruzamento direcional  y = c + r (c - x_p), r ~ U[0,1], x_p sorteado da
          populacao; metade por mutacao direcional  y = c + sigma * xi * d,
          xi ~ U[0,5; 1,5], d aprendida dos avancos (d <- d/2 + passo/2) e sigma
          ajustado pela regra de 1/5. Sem direcao aprendida, d = c - vizinho menos
          periferico. Todos seguidos da mutacao polinomial.
        - frac_refino: busca por coordenadas nos cantos (base e promovidos): cada
          filho muda uma variavel de um canto por +-h; cada canto tem seu ponteiro
          e seu passo h, que cai a metade a cada varredura completa (2D) sem melhora.
        - o resto: cruzamento normal (SBX + mutacao polinomial).
    * Regra de estagnacao: se alguem domina o candidato, o dominador (mais proximo
      em angulo) vira o candidato e a contagem zera; senao, se ha um nao-dominado a
      menos de tau graus com soma de angulos maior por mais de eps, ele vira o
      candidato e a contagem zera; senao, a contagem aumenta. Com K geracoes sem
      avanco, o candidato e promovido a canto. Candidatos a menos de tau/2 de um
      canto sao descartados.
    * Ao atingir max_cantos (fracao da populacao), a fase 2 para de buscar e usa
      todos os filhos no refinamento ate o fim dela.
    * Termina pelo orcamento ou apos L geracoes sem candidato elegivel.

Fase 3 -- distribuicao (o restante do orcamento)
    Todos os filhos por cruzamento normal. Cantos congelados: so trocados por quem
    os domine. Filtro do nadir: o nadir e estimado como no MaOEA-CS (nao-dominados
    mais proximos de cada eixo, estendidos pelos minimos de cada objetivo) e
    estendido pelos cantos; quem passa dele so entra se sobrar vaga.

Em todas as fases: selecao por nao-dominancia com restricoes (regra de Deb) e
selecao por angulo (VaEA/MaOEA-CS) com os cantos como ancoras.
"""
import itertools
import numpy as np

# ======================================================================
# Parametros (todos reunidos aqui)
# ======================================================================
PARAMETROS = dict(
    # orcamento
    frac_fase1_max=0.6,    # orcamento maximo da fase 1 (ela termina antes, por estagnacao)
    parte_fase2=0.4 / (1 - 0.3),   # fracao do restante para a fase 2 (a fase 3 fica com o resto): 4/7
    # fase 1
    rho=1e-4,              # peso do termo de soma no criterio V0
    frac_pais_cruzados=0.25,  # fracao dos cruzamentos com o segundo pai de outra mascara
    W=20,                  # geracoes sem melhora para considerar uma mascara estagnada
    tol_fase1=1e-4,        # melhora relativa minima do melhor V0 de uma mascara
    # fase 2: divisao dos filhos
    frac_candidatos=0.6,   # fracao dos filhos para os operadores dos candidatos
    frac_refino=0.3,       # fracao dos filhos para a busca por coordenadas nos cantos
    # fase 2: candidatos e promocao
    tau=10.0,              # angulo minimo (graus) de um candidato aos cantos e a outros candidatos
    eps=0.1,               # melhora minima (graus) na soma de angulos para contar como avanco
    K=10,                  # geracoes sem avanco para promover um candidato
    L=20,                  # geracoes sem candidato elegivel para encerrar a fase 2
    max_cantos=0.25,       # numero maximo de cantos, como fracao da populacao
    # fase 2: operadores direcionais dos candidatos
    beta=0.5,              # suavizacao da direcao aprendida (d <- (1-beta) d + beta passo)
    sigma_min=0.1, sigma_max=4.0,   # limites do passo da mutacao direcional
    # fase 2: busca por coordenadas
    h0=0.05,               # passo inicial
    h_min=1e-4,            # passo minimo
    # operadores geneticos
    eta_c=20.0, p_c=0.9,   # SBX
    eta_m=20.0,            # mutacao polinomial (probabilidade 1/D por variavel)
)


# ======================================================================
# Funcoes auxiliares
# ======================================================================
def _mascaras(M, niveis):
    return [S for k in sorted(set(niveis)) for S in itertools.combinations(range(M), k)]


def _matriz_ativos(ms, M):
    A = np.zeros((len(ms), M), dtype=bool)
    for i, S in enumerate(ms):
        A[i, list(S)] = True
    return A


def _v0(F, ACT, rho):
    """V0 = max dos ativos + rho * soma de todos. F: (n, M); ACT: (K, M) -> (K, n)."""
    b = np.where(ACT[:, None, :], F[None], -np.inf).max(2)
    return b + rho * F.sum(1)[None]


def _frentes(F):
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=int)
    le = (F[:, None, :] <= F[None, :, :]).all(2)
    lt = (F[:, None, :] < F[None, :, :]).any(2)
    dom = le & lt
    cnt = dom.sum(0)
    rank = np.full(n, -1)
    r = 0
    atual = np.flatnonzero(cnt == 0)
    while len(atual):
        rank[atual] = r
        cnt = cnt - dom[atual].sum(0)
        cnt[rank >= 0] = -1
        atual = np.flatnonzero(cnt == 0)
        r += 1
    return rank


def frentes_com_restricao(F, CV):
    """Factiveis ordenados em frentes; infactiveis depois, por violacao crescente."""
    rank = np.empty(len(F), dtype=int)
    fac = CV <= 0
    if fac.any():
        rank[fac] = _frentes(F[fac])
        base = rank[fac].max() + 1
    else:
        base = 0
    if (~fac).any():
        _, nivel = np.unique(CV[~fac], return_inverse=True)
        rank[~fac] = base + nivel
    return rank


def _sbx(A, B, rng, eta, pc):
    """SBX com limites (Deb e Agrawal), dominio [0, 1]."""
    x1, x2 = np.minimum(A, B), np.maximum(A, B)
    dx = np.maximum(x2 - x1, 1e-14)
    u = rng.random(A.shape)

    def beta_q(beta):
        alpha = 2.0 - beta ** -(eta + 1)
        return np.where(u <= 1 / alpha, (u * alpha) ** (1 / (eta + 1)), (1 / (2 - u * alpha)) ** (1 / (eta + 1)))

    c1 = np.clip(0.5 * (x1 + x2 - beta_q(1 + 2 * x1 / dx) * dx), 0, 1)
    c2 = np.clip(0.5 * (x1 + x2 + beta_q(1 + 2 * (1 - x2) / dx) * dx), 0, 1)
    troca = rng.random(A.shape) < 0.5
    c1, c2 = np.where(troca, c2, c1), np.where(troca, c1, c2)
    faz = (rng.random(A.shape) < 0.5) & (np.abs(A - B) > 1e-14) & (rng.random(A.shape[:-1] + (1,)) < pc)
    return np.where(faz, c1, A), np.where(faz, c2, B)


def _mut_polinomial(X, rng, eta):
    """Mutacao polinomial com limites (Deb), probabilidade 1/D por variavel."""
    m = rng.random(X.shape) < 1.0 / X.shape[-1]
    u = rng.random(X.shape)
    dq1 = (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dq2 = 1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(np.where(m, X + np.where(u < 0.5, dq1, dq2), X), 0, 1)


def _selecao_por_angulo(Fn, ancoras, n):
    """Selecao por angulo: as ancoras entram primeiro; depois, um por vez, o de maior
    angulo ao selecionado mais proximo."""
    U = Fn / np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-12)
    sel = list(dict.fromkeys(int(a) for a in ancoras))[:n]
    if len(sel) >= n:
        return np.array(sel[:n])
    if not sel:
        sel = [int(np.argmin(np.linalg.norm(Fn, axis=1)))]
    cmax = (U @ U[sel].T).max(1)
    cmax[sel] = 2.0
    while len(sel) < n:
        j = int(np.argmin(cmax))
        sel.append(j)
        cmax = np.maximum(cmax, U @ U[j])
        cmax[j] = 2.0
    return np.array(sel)


def _unit(F, zst, nad):
    U = (F - zst) / np.maximum(nad - zst, 1e-12)
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)


def _ang(U, V):
    return np.degrees(np.arccos(np.clip(U @ V.T, -1, 1)))


# ======================================================================
# Algoritmo
# ======================================================================
def executa(avaliar, n_obj, n_var, N, n_ger, seed=None, callback=None, **ajustes):
    """avaliar(X01) -> (F, CV), com X01 em [0,1]^n_var (CV = 0 sem restricoes).
    Orcamento: N * (n_ger + 1) avaliacoes. ajustes: substitui valores de PARAMETROS.
    Devolve (X, F, CV, info) da populacao final."""
    desconhecidos = set(ajustes) - set(PARAMETROS)
    if desconhecidos:
        raise ValueError("parametros desconhecidos: %s" % ", ".join(sorted(desconhecidos)))
    P = dict(PARAMETROS, **ajustes)
    M, D = n_obj, n_var
    rng = np.random.default_rng(seed)
    maxfe = N * (n_ger + 1); fes = 0
    z_baixo = np.zeros(M)
    tau, K, eps = P["tau"], P["K"], P["eps"]
    max_cantos = max(1, int(P["max_cantos"] * N)) if P["max_cantos"] <= 1 else int(P["max_cantos"])
    info = dict(promocoes=[], fim_fase1=None, fim_fase2=None, inicio_so_refino=None)

    sbx = lambda A, B: _sbx(A, B, rng, P["eta_c"], P["p_c"])
    mutpol = lambda X: _mut_polinomial(X, rng, P["eta_m"])


    def aval(X):
        nonlocal fes, z_baixo
        F, CV = avaliar(X); F = np.asarray(F, float); CV = np.asarray(CV, float).reshape(-1)
        fes += len(X); z_baixo = np.minimum(z_baixo, F.min(0)); return F, CV

    def v0(F, CV, ACT):
        c = _v0(F - z_baixo, ACT, P["rho"])
        fac = CV <= 0
        return np.where(fac[None, :], c, np.inf) if fac.any() else np.broadcast_to(CV[None, :], c.shape).copy()

    def cruz_normal(PX, n):
        pa, pb = rng.integers(0, len(PX), (2, (n + 1) // 2)); c1, c2 = sbx(PX[pa], PX[pb])
        return mutpol(np.vstack([c1, c2]))[:n]

    def nadir_maoeacs(F1):
        dist_eixo = np.sqrt(np.maximum((F1 ** 2).sum(1, keepdims=True) - F1 ** 2, 0))
        nadf = F1[np.argmin(dist_eixo, 0)].max(0)
        for j in np.argmin(F1, 0):
            if (F1[j] > nadf).any():
                nadf = np.maximum(nadf, F1[j])
        return nadf

    def selecao(RF, rank, nd, ancoras, zst, esc, nadf=None):
        if nadf is not None and len(nd) > N:                      # filtro do nadir (fase 3)
            escf = np.maximum(nadf - zst, 1e-12)
            dentro = nd[(RF[nd] <= nadf + 1e-9).all(1)]
            anc_d = [int(a) for a in ancoras if rank[a] == 0 and (RF[a] <= nadf + 1e-9).all()]
            pool = np.array(list(dict.fromkeys(anc_d + [int(i) for i in dentro])))
            if len(pool) >= N:
                return pool[_selecao_por_angulo((RF[pool] - zst) / escf, list(range(len(anc_d))), N)]
            fora = np.setdiff1d(nd, pool)
            dfo = np.linalg.norm((RF[fora] - zst) / escf, axis=1)
            return np.concatenate([pool, fora[np.argsort(dfo)[:N - len(pool)]]]).astype(int)
        if len(nd) <= N:
            esc_ = list(nd); r = 1
            while len(esc_) < N:
                fr = np.flatnonzero(rank == r); r += 1
                if len(fr) == 0: break
                if len(esc_) + len(fr) <= N: esc_ += list(fr)
                else:
                    pool = np.array(esc_ + list(fr)); s = _selecao_por_angulo((RF[pool] - zst) / esc, list(range(len(esc_))), N); esc_ = list(pool[s])
            return np.array(esc_)
        anc = [int(a) for a in ancoras if rank[a] == 0]
        pool = np.array(list(dict.fromkeys(anc + [int(i) for i in nd])))
        return pool[_selecao_por_angulo((RF[pool] - zst) / esc, list(range(len(anc))), N)]

    # ================= fase 1: convergencia
    m1 = _mascaras(M, {1, M - 1}); A1 = _matriz_ativos(m1, M); K1 = len(m1)
    mu = max(4, N // K1); ar = np.arange(K1)[:, None]
    X = rng.random((K1, mu, D)); F, CV = aval(X.reshape(-1, D)); F = F.reshape(K1, mu, M); CV = CV.reshape(K1, mu)
    ger1 = max(0, int(P["frac_fase1_max"] * maxfe) // (K1 * mu) - 1)
    chaves = lambda F, CV: np.stack([v0(F[i], CV[i], A1[i:i + 1])[0] for i in range(K1)])
    melhor_v = np.full(K1, np.inf); sem_melhora_m = np.zeros(K1, int)
    for _ in range(ger1):
        if fes + K1 * mu > maxfe: break
        k = chaves(F, CV)

        def torneio():
            i, j = rng.integers(0, mu, (2, K1, mu)); return np.where(k[ar, i] <= k[ar, j], i, j)

        XA, XB = X[ar, torneio()], X[ar, torneio()]
        if K1 > 1:                                   # pais cruzados: segundo pai de outra mascara
            troca = rng.random((K1, mu)) < P["frac_pais_cruzados"]
            outra = (np.arange(K1)[:, None] + rng.integers(1, K1, (K1, mu))) % K1
            XB = np.where(troca[:, :, None], X[outra, rng.integers(0, mu, (K1, mu))], XB)
        C, _ = sbx(XA, XB); C = mutpol(C)
        FC, CVC = aval(C.reshape(-1, D))
        # filhos compartilhados: cada mascara escolhe entre os seus individuos e todos os filhos da geracao
        Ct = np.broadcast_to(C.reshape(1, -1, D), (K1, K1 * mu, D))
        Ft = np.broadcast_to(FC.reshape(1, -1, M), (K1, K1 * mu, M)); CVt = np.broadcast_to(CVC.reshape(1, -1), (K1, K1 * mu))
        X2 = np.concatenate([X, Ct], 1); F2 = np.concatenate([F, Ft], 1); CV2 = np.concatenate([CV, CVt], 1)
        o = np.argsort(chaves(F2, CV2), 1)[:, :mu]; X, F, CV = X2[ar, o], F2[ar, o], CV2[ar, o]
        # termino: todas as mascaras W geracoes sem melhorar o melhor V0 em mais de tol_fase1
        bv = chaves(F, CV).min(1)
        with np.errstate(invalid="ignore"):
            rel = (melhor_v - bv) / np.maximum(np.abs(melhor_v), 1e-12)
        melhorou_m = ~np.isfinite(melhor_v) | (rel > P["tol_fase1"])
        sem_melhora_m = np.where(melhorou_m, 0, sem_melhora_m + 1)
        melhor_v = np.where(melhorou_m, bv, melhor_v)
        if (sem_melhora_m >= P["W"]).all():
            break
        if callback is not None:                     # registra tambem as geracoes da fase 1
            PXm, PFm, PCm = X.reshape(-1, D), F.reshape(-1, M), CV.reshape(-1)
            callback(PXm, PFm, PCm, frentes_com_restricao(PFm, PCm), 0)
    PX, PF, PCV = X.reshape(-1, D), F.reshape(-1, M), CV.reshape(-1)
    if len(PX) < N and fes + (N - len(PX)) <= maxfe:
        XR = rng.random((N - len(PX), D)); FR, CR = aval(XR); PX, PF, PCV = np.vstack([PX, XR]), np.vstack([PF, FR]), np.concatenate([PCV, CR])
    elif len(PX) > N:
        o = rng.choice(len(PX), N, replace=False); PX, PF, PCV = PX[o], PF[o], PCV[o]
    info["fim_fase1"] = fes

    # ================= estado das fases 2 e 3
    prom = []                       # cantos promovidos (x)
    base_x = []                     # cantos-base (x)
    cands = []                      # candidatos
    arqX, arqF, arqC = np.zeros((0, D)), np.zeros((0, M)), np.zeros(0)
    fim_f2 = int(fes + (maxfe - fes) * P["parte_fase2"])   # o restante e dividido entre as fases 2 e 3
    sem_eleg = 0; fase = 2; ger = 0
    est_ref = {}                    # busca por coordenadas: bytes do canto -> dict(ptr, h, tent, n)
    so_refino = False
    cantos, RX, RF = [], np.zeros((0, D)), np.zeros((0, M))

    def filhos_coordenadas(alvos, n_r):
        Y = np.repeat(np.asarray(alvos)[np.arange(n_r) % len(alvos)], 1, axis=0).copy()
        for t in range(n_r):
            a = alvos[t % len(alvos)]
            st = est_ref.setdefault(a.tobytes(), dict(ptr=int(rng.integers(2 * D)), h=P["h0"], tent=0, n=0))
            q = st["ptr"]; st["ptr"] = (q + 1) % (2 * D); st["n"] += 1
            Y[t, q // 2] = np.clip(Y[t, q // 2] + (st["h"] if q % 2 == 0 else -st["h"]), 0, 1)
        return Y

    while fes + N <= maxfe:
        ger += 1
        if fase == 2 and (fes >= fim_f2 or (sem_eleg >= P["L"] and not so_refino)):
            fase = 3; info["fim_fase2"] = fes; cands = []
        # ----- reproducao
        if fase == 2:
            n_c = int(round(P["frac_candidatos"] * N)) if (cands and not so_refino) else 0
            n_r = int(round(P["frac_refino"] * N)) if (base_x or prom) else 0
            if so_refino and (base_x or prom):
                n_r = N
            n_n = N - n_c - n_r
            partes = [cruz_normal(PX, n_n)] if n_n > 0 else []
            if n_c > 0:
                por = [n_c // len(cands) + (1 if i < n_c % len(cands) else 0) for i in range(len(cands))]
                for cd, nc in zip(cands, por):
                    if nc == 0: continue
                    c = cd["x"]
                    n_dir = nc // 2 if np.any(cd["d"]) or np.any(cd["boot"]) else 0
                    n_cx = nc - n_dir
                    if n_cx > 0:                                       # cruzamento direcional
                        xp = PX[rng.integers(0, len(PX), n_cx)]; r_ = rng.random((n_cx, 1))
                        partes.append(mutpol(np.clip(c[None] + r_ * (c[None] - xp), 0, 1)))
                    if n_dir > 0:                                      # mutacao direcional
                        dv = cd["d"] if np.any(cd["d"]) else cd["boot"]
                        Yd = mutpol(np.clip(c[None] + cd["sigma"] * rng.uniform(0.5, 1.5, (n_dir, 1)) * dv[None], 0, 1))
                        partes.append(Yd); cd["dirf"] = {y.tobytes() for y in Yd}; cd["ndir"] = n_dir
                    else:
                        cd["dirf"] = set(); cd["ndir"] = 0
            if n_r > 0:
                partes.append(filhos_coordenadas(list(base_x) + list(prom), n_r))
            QX = np.vstack(partes)[:N]
        else:
            QX = cruz_normal(PX, N)
        QF, QC = aval(QX)
        RX = np.vstack([PX, QX, arqX]); RF = np.vstack([PF, QF, arqF]); RC = np.concatenate([PCV, QC, arqC])
        RX, ui = np.unique(RX, axis=0, return_index=True); RF, RC = RF[ui], RC[ui]
        pos = {RX[i].tobytes(): i for i in range(len(RX))}
        rank = frentes_com_restricao(RF, RC); nd = np.flatnonzero(rank == 0); F1 = RF[nd]
        zst, nad = F1.min(0), F1.max(0); esc = np.maximum(nad - zst, 1e-12); Und = _unit(F1, zst, nad)

        def dominador(i):
            dm = (F1 <= RF[i]).all(1) & (F1 < RF[i]).any(1)
            if not dm.any(): return None
            k_ = np.flatnonzero(dm); return int(nd[k_[np.argmin(_ang(Und[k_], _unit(RF[i:i + 1], zst, nad))[:, 0])]])

        # ----- cantos-base: vagas V0 (fase 2); na fase 3, so por dominancia
        if fase == 2:
            base = list(dict.fromkeys(int(i) for i in np.argmin(v0(RF, RC, A1), 1)))
        else:
            base = []
            for x in base_x:
                i = pos.get(x.tobytes())
                if i is None: continue
                j = dominador(i) if rank[i] > 0 else None
                base.append(j if j is not None else i)
            base = list(dict.fromkeys(base))
        # ----- promovidos: troca por dominador; fusao a menos de tau/2
        pidx = []
        for x in prom:
            i = pos.get(x.tobytes())
            if i is None: continue
            j = dominador(i) if rank[i] > 0 else None
            pidx.append(j if j is not None else i)
        fund = []
        for j in sorted(dict.fromkeys(pidx), key=lambda j: np.linalg.norm((RF[j] - zst) / esc)):
            if j in base: continue
            if all(_ang(_unit(RF[j:j + 1], zst, nad), _unit(RF[k:k + 1], zst, nad))[0, 0] >= tau / 2 for k in fund): fund.append(j)
        pidx = fund
        # ----- candidatos (fase 2)
        if fase == 2:
            cantos = list(dict.fromkeys(base + pidx)); Uc = _unit(RF[cantos], zst, nad); soma_nd = _ang(Und, Uc).sum(1)
            novos = []
            for cd in cands:
                i = pos.get(cd["x"].tobytes())
                if i is None: continue
                ui_ = _unit(RF[i:i + 1], zst, nad); ang_c = _ang(Und, ui_)[:, 0]; soma_c = _ang(ui_, Uc).sum()
                j = dominador(i)
                if j is not None:                                     # dominancia primeiro
                    i = j; cd["idade"] = 0
                else:
                    viz = np.flatnonzero((ang_c < tau) & (soma_nd > soma_c + eps))
                    if len(viz):                                      # avanco para a periferia
                        jn = int(nd[viz[np.argmax(soma_nd[viz])]])
                        cd["d"] = (1 - P["beta"]) * cd["d"] + P["beta"] * (RX[jn] - cd["x"]); i = jn; cd["idade"] = 0
                    else:                                             # estagnacao
                        cd["idade"] += 1
                    if cd["ndir"] > 0:                                # regra de 1/5
                        suc = sum(1 for k_ in viz if RX[int(nd[k_])].tobytes() in cd["dirf"]); p = suc / cd["ndir"]
                        cd["sigma"] = float(np.clip(cd["sigma"] * np.exp((1 / 3) * (p - 0.2) / 0.8), P["sigma_min"], P["sigma_max"]))
                cd["x"] = RX[i].copy()
                if _ang(_unit(RF[i:i + 1], zst, nad), Uc).min() < tau / 2: continue        # encostou num canto
                if cd["idade"] >= K:                                                     # promocao
                    if len(set(base + pidx)) < max_cantos:
                        pidx.append(i); info["promocoes"].append(ger)
                    continue
                novos.append(cd)
            cands = novos
            if len(set(base + pidx)) >= max_cantos:
                so_refino = True; cands = []
                if info["inicio_so_refino"] is None: info["inicio_so_refino"] = fes
            # novos candidatos
            cantos = list(dict.fromkeys(base + pidx)); A = _ang(Und, _unit(RF[cantos], zst, nad))
            ocupados = [pos[cd["x"].tobytes()] for cd in cands if cd["x"].tobytes() in pos]
            eleg = list(np.flatnonzero(A.min(1) >= tau)[np.argsort(-A[A.min(1) >= tau].sum(1))])
            for e in eleg:
                if len(cands) >= M or so_refino: break
                ie = int(nd[e])
                if ocupados and _ang(Und[e:e + 1], _unit(RF[ocupados], zst, nad)).min() < tau: continue
                cands.append(dict(x=RX[ie].copy(), idade=0, d=np.zeros(D), sigma=1.0, dirf=set(), ndir=0, boot=np.zeros(D)))
                ocupados.append(ie)
            sem_eleg = sem_eleg + 1 if (not cands and not so_refino) else 0
            for cd in cands:                      # direcao de partida: do vizinho menos periferico ao candidato
                i = pos.get(cd["x"].tobytes()); cd["boot"] = np.zeros(D)
                if i is None or rank[i] != 0: continue
                perto = np.flatnonzero((_ang(Und, _unit(RF[i:i + 1], zst, nad))[:, 0] < tau) & (nd != i))
                if len(perto):
                    v = perto[np.argmin(_ang(Und[perto], _unit(RF[cantos], zst, nad)).sum(1))]; cd["boot"] = RX[i] - RX[nd[v]]
        # ----- selecao
        cantos = list(dict.fromkeys(base + pidx))
        if len(cantos) > N // 2:              # salvaguarda (com max_cantos <= N/2, nao ocorre)
            ordem = np.array(cantos); s = _selecao_por_angulo((RF[ordem] - zst) / esc, list(range(min(len(base), N // 2))), N // 2); ancoras = list(ordem[s])
        else:
            ancoras = cantos
        nadf = None
        if fase == 3:                         # filtro do nadir, estendido pelos cantos
            nadf = nadir_maoeacs(F1)
            if cantos:
                nadf = np.maximum(nadf, RF[cantos].max(0))
        idx = selecao(RF, rank, nd, ancoras, zst, esc, nadf)
        PX, PF, PCV = RX[idx], RF[idx], RC[idx]
        base_x = [RX[i].copy() for i in base]; prom = [RX[i].copy() for i in pidx]
        atuais = {x.tobytes() for x in base_x + prom}       # passos da busca por coordenadas
        for kb in list(est_ref):
            st = est_ref[kb]
            if kb not in atuais:
                del est_ref[kb]; continue                   # canto trocado: o novo ponto comeca do zero
            st["tent"] += st["n"]; st["n"] = 0
            if st["tent"] >= 2 * D:                         # varredura completa sem melhora
                st["h"] = max(st["h"] / 2, P["h_min"]); st["tent"] = 0
        guardar = list(dict.fromkeys(cantos + [pos[cd["x"].tobytes()] for cd in cands if cd["x"].tobytes() in pos]))
        arqX, arqF, arqC = RX[guardar].copy(), RF[guardar].copy(), RC[guardar].copy()
        if callback is not None:
            callback(PX, PF, PCV, frentes_com_restricao(PF, PCV), len(ancoras))
    info["cantos_F"] = RF[cantos].copy() if ger > 0 else np.zeros((0, M))
    info["cantos_X"] = RX[cantos].copy() if ger > 0 else np.zeros((0, D))
    if info["fim_fase2"] is None: info["fim_fase2"] = fes
    return PX, PF, PCV, info