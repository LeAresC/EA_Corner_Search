from collections import namedtuple

import numpy as np

JANELA = 40
TOLERANCIA = 1e-3


Tabela = namedtuple("Tabela", ["X", "F", "CV"])


def sbx_crossover(pais_a, pais_b, pc=0.9, eta=20):
    x1, x2 = np.minimum(pais_a, pais_b), np.maximum(pais_a, pais_b)
    dx = np.maximum(x2 - x1, 1e-14)
    u = np.random.rand(*pais_a.shape)

    def beta_q(beta):
        alpha = 2.0 - beta ** -(eta + 1)
        return np.where(u <= 1 / alpha, (u * alpha) ** (1 / (eta + 1)), (1 / (2 - u * alpha)) ** (1 / (eta + 1)))

    c1 = np.clip(0.5 * (x1 + x2 - beta_q(1 + 2 * x1 / dx) * dx), 0, 1)
    c2 = np.clip(0.5 * (x1 + x2 + beta_q(1 + 2 * (1 - x2) / dx) * dx), 0, 1)
    troca = np.random.rand(*pais_a.shape) < 0.5
    c1, c2 = np.where(troca, c2, c1), np.where(troca, c1, c2)
    cruza = (np.random.rand(*pais_a.shape) < 0.5) & (np.abs(pais_a - pais_b) > 1e-14)\
            & (np.random.rand(pais_a.shape[0], 1) < pc)
    return np.where(cruza, c1, pais_a), np.where(cruza, c2, pais_b)


def polynomial_mutation(X, eta=20):
    muta = np.random.rand(*X.shape) < 1.0 / X.shape[1]
    u = np.random.rand(*X.shape)
    dq1 = (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dq2 = 1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(np.where(muta, X + np.where(u < 0.5, dq1, dq2), X), 0, 1)


def _frentes(F):
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=int)
    menor_ou_igual = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    domina = menor_ou_igual & ~menor_ou_igual.T
    dominado_por = domina.sum(0)
    rank = np.full(n, -1)
    nivel = 0
    atual = np.flatnonzero(dominado_por == 0)
    while len(atual):
        rank[atual] = nivel
        dominado_por = dominado_por - domina[atual].sum(0)
        dominado_por[rank >= 0] = -1
        atual = np.flatnonzero(dominado_por == 0)
        nivel += 1
    return rank


def frentes_com_restricao(F, CV):
    rank = np.empty(len(F), dtype=int)
    factivel = CV <= 0
    if factivel.any():
        rank[factivel] = _frentes(F[factivel])
        proximo = rank[factivel].max() + 1
    else:
        proximo = 0
    if (~factivel).any():
        _, nivel = np.unique(CV[~factivel], return_inverse=True)
        rank[~factivel] = proximo + nivel
    return rank


def unitarios(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def angulos(U, V):
    return np.arccos(np.clip(U @ V.T, -1, 1))


def ponto_ideal(F, CV):
    factivel = CV <= 0
    return (F[factivel] if factivel.any() else F).min(axis=0)


def primeira_frente(F, CV):
    factivel = CV <= 0
    mascara = np.zeros(len(F), dtype=bool)
    if factivel.any():
        G = F[factivel]
        menor_ou_igual = (G[:, None, :] <= G[None, :, :]).all(axis=2)
        domina = menor_ou_igual & ~menor_ou_igual.T
        mascara[np.flatnonzero(factivel)] = ~domina.any(axis=0)
    elif len(F):
        mascara = CV == CV.min()
    return mascara


def referencia(F, CV, frente=None):
    ideal = ponto_ideal(F, CV)
    if frente is None:
        frente = primeira_frente(F, CV)
    escala = np.maximum(F[frente].max(axis=0) - ideal, 1e-12)
    return ideal, escala


def nota(F_transladado, CV, w, rho):
    tchebycheff = (F_transladado / w).max(axis=1) + rho * F_transladado.sum(axis=1)
    return np.where(CV <= 0, tchebycheff, 1e12 + CV)


def escolhe_ancoras_com_memoria(F_transladado, K, direcoes_antigas, livres, repulsoras, eixo=None):
    M = F_transladado.shape[1]
    U = unitarios(F_transladado)
    eixo = np.ones(M) / np.sqrt(M) if eixo is None else eixo
    ao_centro = angulos(U, eixo[None])[:, 0]
    soma = ao_centro.copy()
    menor = np.full(len(U), np.inf)
    if len(repulsoras):
        A = angulos(U, repulsoras)
        soma += A.sum(axis=1)
        menor = A.min(axis=1)

    def registra(nova):
        nonlocal soma, menor
        a_ = angulos(U, U[nova:nova + 1])[:, 0]
        soma = soma + a_
        menor = np.minimum(menor, a_)

    ancoras = [-1] * K
    escolhidas = []

    presas = [k for k in range(K) if direcoes_antigas is not None and k not in livres]
    if presas:
        D = direcoes_antigas[presas]
        coincide = angulos(D, D) < 1e-6
        np.fill_diagonal(coincide, True)
        grupo = np.array([int(np.flatnonzero(coincide[i])[0]) for i in range(len(presas))])
        celula = grupo[angulos(U, D).argmin(axis=1)]
        for i, k in enumerate(presas):
            candidatos = np.flatnonzero(celula == grupo[i])
            candidatos = candidatos[~np.isin(candidatos, escolhidas)]
            if len(candidatos) == 0:
                continue
            nova = int(candidatos[np.argmax(soma[candidatos])])
            ancoras[k] = nova
            escolhidas.append(nova)
            registra(nova)

    for k in range(K):
        if ancoras[k] >= 0:
            continue
        escore = ao_centro + np.where(np.isfinite(menor), menor, 0.0)
        escore[escolhidas] = -np.inf
        nova = int(np.argmax(escore))
        ancoras[k] = nova
        escolhidas.append(nova)
        registra(nova)
    return np.array(ancoras)


def monta_tabelas(X, F, CV, K, mu, rho, direcoes_antigas=None, livres=(), F_permanentes=None, eixo=None):
    M = F.shape[1]
    ideal, escala = referencia(F, CV)
    F_transladado = (F - ideal) / escala
    repulsoras = np.empty((0, M))
    if F_permanentes is not None and len(F_permanentes):
        repulsoras = unitarios(np.maximum((F_permanentes - ideal) / escala, 0))
    factiveis = np.flatnonzero(CV <= 0)
    elegiveis = factiveis if len(factiveis) >= K else np.arange(len(F))
    antigas = direcoes_antigas if len(elegiveis) >= K else None
    eixo_central = np.ones(M) / np.sqrt(M) if eixo is None else eixo
    ancoras = elegiveis[escolhe_ancoras_com_memoria(F_transladado[elegiveis], K, antigas, set(livres), repulsoras, eixo=eixo_central)]
    direcoes = np.vstack([eixo_central, np.maximum(unitarios(F_transladado[ancoras]), 1e-6)])
    tabelas = []
    for w in direcoes:
        melhores = np.argsort(nota(F_transladado, CV, w, rho), kind="stable")[:mu]
        tabelas.append(Tabela(X[melhores], F[melhores], CV[melhores]))
    return direcoes, tabelas


def gera_filhos(tabelas, direcoes, mu, rho, p_aleatorio, rng, normalizacao=None):
    if normalizacao is None:
        normalizacao = referencia(np.vstack([t.F for t in tabelas]), np.concatenate([t.CV for t in tabelas]))
    ideal, escala = normalizacao
    populacao = np.vstack([t.X for t in tabelas])
    filhos = []
    for k, tabela in enumerate(tabelas):
        notas = nota((tabela.F - ideal) / escala, tabela.CV, direcoes[k], rho)

        i, j = rng.integers(0, len(tabela.X), (2, mu))
        pais_a = tabela.X[np.where(notas[i] <= notas[j], i, j)].copy()
        i, j = rng.integers(0, len(tabela.X), (2, mu))
        pais_b = tabela.X[np.where(notas[i] <= notas[j], i, j)].copy()

        for r in np.flatnonzero(rng.random(mu) < p_aleatorio):
            pais_a[r] = populacao[rng.integers(len(populacao))]
            pais_b[r] = populacao[rng.integers(len(populacao))]
        filho, _ = sbx_crossover(pais_a, pais_b)
        filhos.append(polynomial_mutation(filho))
    return np.vstack(filhos)


def uniao(*conjuntos):
    X, posicoes = np.unique(np.vstack([c[0] for c in conjuntos]), axis=0, return_index=True)
    F = np.vstack([c[1] for c in conjuntos])[posicoes]
    CV = np.concatenate([c[2] for c in conjuntos])[posicoes]
    return X, F, CV


def situacao_das_tabelas(tabelas, eixo=None):
    M = tabelas[0].F.shape[1]
    ideal, escala = referencia(np.vstack([t.F for t in tabelas]), np.concatenate([t.CV for t in tabelas]))
    perifericas = tabelas[1:]
    melhores = np.array([t.F[0] for t in perifericas])
    eixo_central = np.ones(M) / np.sqrt(M) if eixo is None else eixo
    angulo_ao_centro = np.degrees(angulos(unitarios((melhores - ideal) / escala), eixo_central[None])[:, 0])
    melhor_ponto = [(t.F[0], t.CV[0], t.X[0]) for t in perifericas]
    return angulo_ao_centro, melhor_ponto, (ideal, escala)


def nota_do_ponto(ponto, w, normalizacao, rho):
    F, CV = ponto[0], ponto[1]
    ideal, escala = normalizacao
    return nota((F[None] - ideal) / escala, np.array([CV]), w, rho)[0]


def eixo_pelas_extremidades(F_extremidades, F, CV):
    ideal, escala = referencia(F, CV)
    return unitarios(np.maximum(((F_extremidades - ideal) / escala).mean(axis=0), 1e-6)[None])[0]


def selecao_por_frentes(F, CV, ancoras, n):
    if n <= 0 or len(F) == 0:
        return []
    rank = frentes_com_restricao(F, CV)
    ideal, escala = referencia(F, CV, frente=(rank == 0))
    F_normalizado = (F - ideal) / escala
    escolhidos, r = [], 0
    while len(escolhidos) < n:
        frente = np.flatnonzero(rank == r); r += 1
        if len(frente) == 0:
            break
        if len(escolhidos) + len(frente) <= n:
            escolhidos += [int(f) for f in frente]
            continue
        na_frente = set(int(f) for f in frente)
        primeiros = list(dict.fromkeys(escolhidos + [a for a in ancoras if a in na_frente]))
        resto = [int(f) for f in frente if int(f) not in set(primeiros)]
        grupo = np.array(primeiros + resto)
        if len(primeiros) <= n:
            escolhidos = [int(i) for i in grupo[truncamento_knn_angular(F_normalizado[grupo], len(primeiros), n)]]
        else:
            escolhidos = [int(i) for i in grupo[selecao_por_angulo(F_normalizado[grupo], list(range(len(primeiros))), n)]]
    return escolhidos


def truncamento_knn_angular(F, n_protegidos, n, k=3):
    m = len(F)
    if m <= n:
        return np.arange(m)
    if n <= k:
        return truncamento_knn_angular_simples(F, n_protegidos, n, k)
    U = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    D = np.arccos(np.clip(U @ U.T, -1, 1)); np.fill_diagonal(D, np.inf)
    viz = np.argpartition(D, k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(D, viz, axis=1)
    o = np.argsort(dist, axis=1); viz = np.take_along_axis(viz, o, axis=1); dist = np.take_along_axis(dist, o, axis=1)
    vivo = np.ones(m, dtype=bool); restantes = m
    while restantes > n:
        cand = np.flatnonzero(vivo[n_protegidos:]) + n_protegidos
        chave = dist[cand]
        r = cand[np.lexsort(tuple(chave[:, j] for j in range(k - 1, -1, -1)))[0]]
        vivo[r] = False; restantes -= 1; D[:, r] = np.inf
        afetados = np.flatnonzero(vivo & (viz == r).any(axis=1))
        if len(afetados):
            linhas = D[afetados]
            v = np.argpartition(linhas, k - 1, axis=1)[:, :k]
            d = np.take_along_axis(linhas, v, axis=1); o = np.argsort(d, axis=1)
            viz[afetados] = np.take_along_axis(v, o, axis=1); dist[afetados] = np.take_along_axis(d, o, axis=1)
    return np.flatnonzero(vivo)


def truncamento_knn_angular_simples(F, n_protegidos, n, k=3):
    U = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    D = np.arccos(np.clip(U @ U.T, -1, 1)); np.fill_diagonal(D, np.inf)
    vivo = np.ones(len(F), dtype=bool)
    while vivo.sum() > n:
        cand = np.flatnonzero(vivo[n_protegidos:]) + n_protegidos
        Dc = D[np.ix_(cand, np.flatnonzero(vivo))]
        kk = min(k, Dc.shape[1] - 1)
        viz = np.sort(np.partition(Dc, kk, axis=1)[:, :kk + 1], axis=1)[:, :k]
        ordem = np.lexsort(tuple(viz[:, j] for j in range(viz.shape[1] - 1, -1, -1)))
        vivo[cand[ordem[0]]] = False
    return np.flatnonzero(vivo)


def selecao_por_angulo(F_normalizado, iniciais, n):
    U = unitarios(F_normalizado)
    escolhidos = list(dict.fromkeys(int(a) for a in iniciais))[:n]
    if len(escolhidos) >= n:
        return np.array(escolhidos[:n])
    if not escolhidos:
        escolhidos = [int(np.argmin(np.linalg.norm(F_normalizado, axis=1)))]
    maior_cosseno = (U @ U[escolhidos].T).max(axis=1)
    maior_cosseno[escolhidos] = 2.0
    while len(escolhidos) < n:
        novo = int(np.argmin(maior_cosseno))
        escolhidos.append(novo)
        maior_cosseno = np.maximum(maior_cosseno, U @ U[novo])
        maior_cosseno[novo] = 2.0
    return np.array(escolhidos)


def executa(avaliar, n_obj, n_var, N, n_ger, seed=0, K=None, rho=0.1, p_aleatorio=0.1,
            frac_fase1=0.7, frac_tabelas=0.1, intervalo_eixo=0.1, callback=None):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    K = 2 * n_obj if K is None else int(K)
    n_tabelas = K + 1
    mu = max(2, N // n_tabelas)
    orcamento = N * (n_ger + 1)
    avaliacoes = 0

    def avalia(X):
        nonlocal avaliacoes
        F, CV = avaliar(X)
        avaliacoes += len(X)
        return np.asarray(F, float), np.asarray(CV, float).reshape(-1)


    X = rng.random((n_tabelas * mu, n_var))
    F, CV = avalia(X)
    ideal_inicial, escala_inicial = referencia(F, CV)
    centroide = np.maximum(((F - ideal_inicial) / escala_inicial).mean(axis=0), 1e-6)
    eixo = centroide / np.linalg.norm(centroide)
    proxima_atualizacao = intervalo_eixo * orcamento

    def atualiza_eixo(F_atual, CV_atual):
        nonlocal eixo, proxima_atualizacao
        if avaliacoes >= proxima_atualizacao:
            extremidades = [t.F[0] for t in tabelas[1:]] + [p[0] for p in permanentes]
            eixo = eixo_pelas_extremidades(np.array(extremidades), F_atual, CV_atual)
            while proxima_atualizacao <= avaliacoes:
                proxima_atualizacao += intervalo_eixo * orcamento
    direcoes, tabelas = monta_tabelas(X, F, CV, K, mu, rho, eixo=eixo)
    angulo_ao_centro, melhor_ponto, normalizacao = situacao_das_tabelas(tabelas, eixo=direcoes[0])
    historico = [[(melhor_ponto[k], angulo_ao_centro[k])] for k in range(K)]
    permanentes = []
    livres = set()
    direcoes_permanentes = []


    while avaliacoes + n_tabelas * mu <= frac_fase1 * orcamento:


        filhos = gera_filhos(tabelas, direcoes, mu, rho, p_aleatorio, rng, normalizacao=normalizacao)
        F_filhos, CV_filhos = avalia(filhos)
        candidatos = uniao(*[(t.X, t.F, t.CV) for t in tabelas], (filhos, F_filhos, CV_filhos))


        F_perm = np.array([p[0] for p in permanentes]) if permanentes else None
        atualiza_eixo(candidatos[1], candidatos[2])
        direcoes, tabelas = monta_tabelas(*candidatos, K, mu, rho, direcoes_antigas=unitarios(direcoes[1:]),
                                          livres=livres, F_permanentes=F_perm, eixo=eixo)
        livres = set()


        angulo_ao_centro, melhor_ponto, normalizacao = situacao_das_tabelas(tabelas, eixo=direcoes[0])
        for k in range(K):
            w = direcoes[k + 1]
            if not historico[k]:
                historico[k] = [(melhor_ponto[k], angulo_ao_centro[k])]
                continue
            recordista_anterior, recorde_anterior = historico[k][-1]
            recorde = max(angulo_ao_centro[k], recorde_anterior)
            atual = melhor_ponto[k]
            recordista = atual if nota_do_ponto(atual, w, normalizacao, rho) <= nota_do_ponto(recordista_anterior, w, normalizacao, rho) else recordista_anterior
            historico[k].append((recordista, recorde))
            if len(historico[k]) > JANELA:
                recordista_antigo, recorde_antigo = historico[k].pop(0)
                angulo_subiu = recorde - recorde_antigo > TOLERANCIA * abs(recorde_antigo)
                nota_antiga = nota_do_ponto(recordista_antigo, w, normalizacao, rho)
                nota_melhorou = nota_antiga - nota_do_ponto(recordista, w, normalizacao, rho) > TOLERANCIA * abs(nota_antiga)
                if not (angulo_subiu or nota_melhorou):
                    permanentes.append(recordista)
                    ideal_n, escala_n = normalizacao
                    direcoes_permanentes.append(np.maximum(unitarios(np.maximum((recordista[0][None] - ideal_n) / escala_n, 0))[0], 1e-6))
                    historico[k] = []
                    livres.add(k)

        if callback is not None:
            callback(*uniao(*[(t.X, t.F, t.CV) for t in tabelas]))

    fim_fase1 = avaliacoes


    n_antes = len(permanentes)
    if permanentes:
        F_todos = np.vstack([np.array([p[0] for p in permanentes])] + [t.F for t in tabelas])
        CV_todos = np.concatenate([np.array([p[1] for p in permanentes])] + [t.CV for t in tabelas])
        na_frente = primeira_frente(F_todos, CV_todos)[:len(permanentes)]
        permanentes = [p for p, ok in zip(permanentes, na_frente) if ok]
        direcoes_permanentes = [w for w, ok in zip(direcoes_permanentes, na_frente) if ok]


    def conjunto(*partes):
        partes = [p for p in partes if len(p[0])]
        return uniao(*partes)

    def permanentes_como_conjunto():
        if not permanentes:
            return (np.empty((0, n_var)), np.empty((0, n_obj)), np.empty(0))
        return (np.array([p[2] for p in permanentes]), np.array([p[0] for p in permanentes]),
                np.array([p[1] for p in permanentes]))

    P = conjunto(*[(t.X, t.F, t.CV) for t in tabelas], permanentes_como_conjunto())
    n_filhos_tabelas = int(round(frac_tabelas * N))
    while avaliacoes + N <= orcamento:

        filhos_tabelas = gera_filhos(tabelas, direcoes, mu, rho, p_aleatorio, rng)
        filhos_tabelas = filhos_tabelas[rng.permutation(len(filhos_tabelas))[:n_filhos_tabelas]]
        i, j = rng.integers(0, len(P[0]), (2, N - len(filhos_tabelas)))
        filho, _ = sbx_crossover(P[0][i], P[0][j])
        filhos = np.vstack([filhos_tabelas, polynomial_mutation(filho)])
        F_filhos, CV_filhos = avalia(filhos)
        Q = (filhos, F_filhos, CV_filhos)


        candidatos = conjunto(*[(t.X, t.F, t.CV) for t in tabelas], P, Q)
        F_perm = np.array([p[0] for p in permanentes]) if permanentes else None
        atualiza_eixo(candidatos[1], candidatos[2])
        direcoes, tabelas = monta_tabelas(*candidatos, K, mu, rho, direcoes_antigas=unitarios(direcoes[1:]),
                                          livres=set(), F_permanentes=F_perm, eixo=eixo)


        R = conjunto(P, Q, *[(t.X, t.F, t.CV) for t in tabelas], permanentes_como_conjunto())
        ideal, escala = referencia(R[1], R[2])
        R_normalizado = (R[1] - ideal) / escala


        for p in range(len(permanentes)):
            w = direcoes_permanentes[p]
            notas_R = nota(R_normalizado, R[2], w, rho)
            melhor = int(np.argmin(notas_R))
            F_p, CV_p, X_p = permanentes[p]
            if notas_R[melhor] < nota((F_p[None] - ideal) / escala, np.array([CV_p]), w, rho)[0]:
                permanentes[p] = (R[1][melhor], R[2][melhor], R[0][melhor])


        melhores_tabelas = (np.array([t.X[0] for t in tabelas[1:]]), np.array([t.F[0] for t in tabelas[1:]]),
                            np.array([t.CV[0] for t in tabelas[1:]]))
        S = conjunto(P, Q, melhores_tabelas, permanentes_como_conjunto())
        posicao = {S[0][k].tobytes(): k for k in range(len(S[0]))}
        ancoras = [posicao[x.tobytes()] for x in melhores_tabelas[0]] + [posicao[p[2].tobytes()] for p in permanentes]
        ancoras = list(dict.fromkeys(ancoras))
        escolhidos = np.array(selecao_por_frentes(S[1], S[2], ancoras, N))
        P = (S[0][escolhidos], S[1][escolhidos], S[2][escolhidos])

        if callback is not None:
            callback(*P)


    perifericas = tabelas[1:]
    info = dict(ancoras_X=np.array([t.X[0] for t in perifericas]),
                ancoras_F=np.array([t.F[0] for t in perifericas]),
                permanentes_X=np.array([p[2] for p in permanentes]) if permanentes else np.empty((0, n_var)),
                permanentes_F=np.array([p[0] for p in permanentes]) if permanentes else np.empty((0, n_obj)),
                fim_fase1=fim_fase1, permanentes_descartadas=n_antes - len(permanentes),
                avaliacoes=avaliacoes, K=K)
    return P[0], P[1], P[2], info