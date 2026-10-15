"""MEAMT classico -- Multiobjective Evolutionary Algorithm with Many Tables.

Reimplementacao em NumPy da versao com tabelas (DEAP) usada no inicio do projeto,
com a possibilidade de escolher quais tabelas participam.

IDEIA
-----
Cada tabela e uma subpopulacao associada a um subconjunto S dos objetivos (uma
"mascara"). Todas evoluem ao mesmo tempo, compartilhando os filhos:

  * a selecao de sobreviventes de cada tabela e o NSGA-II aplicado SO aos
    objetivos de S (dominancia e crowding distance calculados em S);
  * os pais vem de qualquer tabela, escolhida por torneio de score;
  * cada filho e oferecido a todas as tabelas;
  * a tabela 0 e um arquivo com o NSGA-II em todos os objetivos.

POR QUE AS TABELAS CONSERVAM BORDAS E, EM ALGUNS CASOS, CANTOS
----------------------------------------------------------------
A tabela S guarda solucoes nao-dominadas no subproblema "minimizar f_S",
ignorando os demais objetivos. Uma solucao assim e (fracamente) Pareto-otima
no problema completo e fica sempre na BORDA da fronteira:

  * Tabela de um objetivo, S = {f_i}: guarda as solucoes de f_i minimo. Isso e
    uma borda inteira da fronteira (no DTLZ2 com 3 objetivos, a curva f1 = 0,
    que liga dois cantos). Dentro da borda nao ha pressao de espalhamento,
    porque todos os pontos empatam em f_i.

  * Tabela de m-1 objetivos, S = todos menos f_j: guarda as solucoes em que os
    objetivos de S sao minimos AO MESMO TEMPO. Em fronteiras como a do DTLZ2,
    isso e um unico ponto -- um CANTO, onde f_j e maximo. Em outras fronteiras
    pode ser uma regiao de borda (por isso "em alguns casos" cantos).

  * Tabela com todos os objetivos: e o NSGA-II comum e cobre o interior.

Essa observacao -- tabelas parciais conservam a borda, e as de m-1 objetivos
chegam a cantos -- foi o ponto de partida da abordagem por cantos com o
criterio de Tchebycheff (MEAMT por cantos / CCSGA).

ESCOLHA DAS TABELAS (parametro `tabelas`)
-----------------------------------------
  "todas"            todas as 2^m - 1 mascaras (cresce exponencialmente com m)
  {"niveis": [1, 2]} todas as mascaras com 1 e 2 objetivos ativos
  [(1,), (1, 2), (1, 2, 3)]
                     mascaras explicitas, com os objetivos numerados a partir
                     de 1 (f1, f2, ...)

MECANISMO DE SCORE (alocacao="score", padrao, como na versao original)
-----------------------------------------------------------------------
  * a cada geracao, o score de cada tabela decai (fator `decaimento`);
  * cada filho que sobrevive numa tabela paga "royalties" a tabela de onde veio
    o seu pai: 1 + bonus de diversidade (crowding distance infinita -> 1;
    finita -> min(4, 5 * crowding));
  * as vagas sao realocadas proporcionalmente ao score, com piso `min_vagas`.
Com alocacao="fixa", todas as tabelas tem o mesmo tamanho o tempo todo e o
score nao e usado (versao mais simples, util para explicar as tabelas).
"""
import itertools
import numpy as np

PARAMETROS = dict(
    tabelas="todas",       # "todas", {"niveis": [...]} ou lista de mascaras (objetivos numerados a partir de 1)
    incluir_completa=True, # garante a tabela com todos os objetivos (o NSGA-II comum)
    alocacao="score",      # "score" (versao original) ou "fixa"
    decaimento=0.95,       # fator de decaimento do score por geracao
    min_vagas=3,           # piso de vagas por tabela (alocacao por score)
    cxpb=1.0,              # probabilidade de cruzamento (SBX)
    eta_c=20.0,            # SBX
    eta_m=20.0,            # mutacao polinomial (probabilidade 1/D por variavel)
)


# ======================================================================
# Auxiliares
# ======================================================================
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

def _sde_crowding(F, k=None):
    """
    Calcula o SDE (Shift-Based Density Estimation) para a matriz de objetivos F.
    Retorna um vetor de 'distâncias de densidade': quanto MAIOR o valor, MAIS isolado
    e melhor convergido é o indivíduo.
    """
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    
    # Se k não for especificado, usa a raiz quadrada do tamanho da população
    if k is None:
        k = int(np.sqrt(n))
        
    # Matriz 3D de deslocamento: Shift_F[i, j, m] = max(F[j, m], F[i, m])
    # Comparando cada indivíduo j deslocado em relação à referência i
    shifted_F = np.maximum(F[:, None, :], F[None, :, :])
    
    # Distância de cada indivíduo j para a referência i
    # dist_matrix[i, j] é a distância SDE de j em relação a i
    diff = shifted_F - F[:, None, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Define a diagonal como infinito para ignorar a distância de um indivíduo para ele mesmo
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Para cada indivíduo, ordena as distâncias SDE até os vizinhos
    dist_ordenadas = np.sort(dist_matrix, axis=1)
    
    # A densidade SDE é a distância até o k-ésimo vizinho mais próximo
    # Maior distância = menor densidade = melhor preservação
    sde_cd = dist_ordenadas[:, k - 1]
    return sde_cd


def _crowding(F):
    """Crowding distance do NSGA-II (como no DEAP), nos objetivos dados."""
    n, m = F.shape
    d = np.zeros(n)
    if n == 0:
        return d
    for j in range(m):
        o = np.argsort(F[:, j], kind="stable")
        d[o[0]] = d[o[-1]] = np.inf
        amp = F[o[-1], j] - F[o[0], j]
        if amp == 0:
            continue
        d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / (m * amp)
    return d


def selecao_nsga2(F, CV, k, ativos=None):
    """NSGA-II nos objetivos `ativos` (todos, se None). Devolve (indices, crowding)."""
    Fa = F if ativos is None else F[:, ativos]
    num_totais = F.shape[1]
    
    if ativos is None or len(ativos) == num_totais:
        Fa = F
        Fi = F  # Se todos estão ativos, usa o F completo
    else:
        Fa = F[:, ativos]
        # Identifica os índices dos objetivos inativos
        inativos = [i for i in range(num_totais) if i not in ativos]
        Fi = F[:, inativos]
    rank = frentes_com_restricao(Fa, CV)
    cd = np.zeros(len(F))
    for r in np.unique(rank):
        fr = np.flatnonzero(rank == r)
        cd[fr] = _sde_crowding(Fi[fr])
    ordem = np.lexsort((-cd, rank))
    return ordem[:k], cd


def _sbx(A, B, rng, eta, pc):
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
    m = rng.random(X.shape) < 1.0 / X.shape[-1]
    u = rng.random(X.shape)
    dq1 = (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dq2 = 1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(np.where(m, X + np.where(u < 0.5, dq1, dq2), X), 0, 1)


def resolve_tabelas(tabelas, M, incluir_completa=True):
    """Converte a especificacao de tabelas numa lista de mascaras (indices a partir de 0)."""
    if tabelas == "todas":
        ms = [S for k in range(1, M + 1) for S in itertools.combinations(range(M), k)]
    elif isinstance(tabelas, dict) and "niveis" in tabelas:
        niveis = sorted(set(int(k) for k in tabelas["niveis"]))
        if any(k < 1 or k > M for k in niveis):
            raise ValueError("niveis devem estar entre 1 e %d" % M)
        ms = [S for k in niveis for S in itertools.combinations(range(M), k)]
    else:
        ms = []
        for S in tabelas:
            S = tuple(sorted(set(int(a) for a in S)))
            if not S or S[0] < 1 or S[-1] > M:
                raise ValueError("mascara invalida %r: use objetivos de 1 a %d" % (S, M))
            ms.append(tuple(a - 1 for a in S))
        ms = list(dict.fromkeys(ms))
    if incluir_completa and tuple(range(M)) not in ms:
        ms.append(tuple(range(M)))
    return ms


def nome_tabela(S):
    return "{" + ", ".join("f%d" % (a + 1) for a in S) + "}"


# ======================================================================
# Algoritmo
# ======================================================================
def executa(avaliar, n_obj, n_var, N, n_ger, seed=None, callback=None, **ajustes):
    """avaliar(X01) -> (F, CV), com X01 em [0,1]^n_var (CV = 0 sem restricoes).
    N: tamanho da populacao (total das tabelas e do arquivo). Orcamento:
    N * (n_ger + 1) avaliacoes. Devolve (X, F, CV, info) do arquivo (tabela 0).
    info["tabelas"]: para cada tabela, as solucoes que ela guarda no fim."""
    desconhecidos = set(ajustes) - set(PARAMETROS)
    if desconhecidos:
        raise ValueError("parametros desconhecidos: %s" % ", ".join(sorted(desconhecidos)))
    P = dict(PARAMETROS, **ajustes)
    if P["alocacao"] not in ("score", "fixa"):
        raise ValueError("alocacao deve ser 'score' ou 'fixa'")
    M, D = n_obj, n_var
    rng = np.random.default_rng(seed)
    maxfe = N * (n_ger + 1); fes = 0
    masks = resolve_tabelas(P["tabelas"], M, P["incluir_completa"])
    T = len(masks)
    if T * P["min_vagas"] > N:
        raise ValueError("%d tabelas x min_vagas=%d excedem a populacao N=%d: escolha menos tabelas ou aumente N"
                         % (T, P["min_vagas"], N))

    def aval(X):
        nonlocal fes
        F, CV = avaliar(X); fes += len(X)
        return np.asarray(F, float), np.asarray(CV, float).reshape(-1)

    # ----- inicializacao: populacao aleatoria; cada tabela seleciona a sua parte
    X0 = rng.random((N, D)); F0, C0 = aval(X0)
    tam = np.full(T, N // T); tam[: N - tam.sum()] += 1
    tab = []
    for t, S in enumerate(masks):
        idx, _ = selecao_nsga2(F0, C0, tam[t], list(S))
        tab.append([X0[idx], F0[idx], C0[idx]])
    score = np.zeros(T)
    rank0 = frentes_com_restricao(F0, C0)
    AX, AF, AC = X0[rank0 == 0], F0[rank0 == 0], C0[rank0 == 0]

    while fes + N <= maxfe:
        if P["alocacao"] == "score":
            score *= P["decaimento"]
        # ----- pais: torneio de score entre duas tabelas, individuo sorteado na vencedora
        n_par = (N + 1) // 2
        cheias = np.array([len(tb[0]) > 0 for tb in tab])
        r1, r2 = rng.integers(0, T, (2, 2 * n_par))
        venc = np.where(~cheias[r1], r2, np.where(~cheias[r2], r1, np.where(score[r1] >= score[r2], r1, r2)))
        pais = np.array([tab[t][0][rng.integers(len(tab[t][0]))] for t in venc])
        c1, c2 = _sbx(pais[:n_par], pais[n_par:], rng, P["eta_c"], P["cxpb"])
        QX = _mut_polinomial(np.vstack([c1, c2]), rng, P["eta_m"])[:N]
        origem = np.concatenate([venc[:n_par], venc[n_par:]])[:N]
        QF, QC = aval(QX)
        # ----- arquivo (tabela 0): nao-dominados de todos os objetivos, truncados por NSGA-II
        UX, UF, UC = np.vstack([AX, QX]), np.vstack([AF, QF]), np.concatenate([AC, QC])
        r = frentes_com_restricao(UF, UC); nd = np.flatnonzero(r == 0)
        if len(nd) > N:
            s, _ = selecao_nsga2(UF[nd], UC[nd], N); nd = nd[s]
        AX, AF, AC = UX[nd], UF[nd], UC[nd]
        # ----- cada tabela: uniao com os filhos elegíveis e NSGA-II nos seus objetivos
        max_tabelas_por_filho = P.get("max_tabelas", 2)  # Limite desejado (ex: 1 para exclusividade total)
        usos_filhos = np.zeros(N, dtype=int)
        
        # Embaralha a ordem de acesso das tabelas a cada geração para não favorecer a tabela 0
        ordem_tabelas = rng.permutation(T)

        for t in ordem_tabelas:
            S = masks[t]
            n_ant = len(tab[t][0])
            
            # Filtra apenas os filhos de QX que ainda não atingiram o limite de pertencimento
            elegiveis = np.flatnonzero(usos_filhos < max_tabelas_por_filho)
            
            if len(elegiveis) == 0:
                # Se nenhum filho puder mais entrar, a tabela apenas faz a seleção interna da sua população
                UX, UF, UC = tab[t][0], tab[t][1], tab[t][2]
                idx, cd = selecao_nsga2(UF, UC, tam[t], list(S))
                tab[t] = [UX[idx], UF[idx], UC[idx]]
                continue

            # Concatena a população atual da tabela apenas com os filhos elegíveis
            UX = np.vstack([tab[t][0], QX[elegiveis]])
            UF = np.vstack([tab[t][1], QF[elegiveis]])
            UC = np.concatenate([tab[t][2], QC[elegiveis]])
            
            idx, cd = selecao_nsga2(UF, UC, tam[t], list(S))
            
            # Identifica quais dos filhos elegíveis foram efetivamente aceitos
            filhos_locais = idx[idx >= n_ant] - n_ant
            filhos_globais = elegiveis[filhos_locais]
            
            # Incrementa o contador de tabelas em que esses filhos entraram
            usos_filhos[filhos_globais] += 1

            if P["alocacao"] == "score":
                filhos = idx[idx >= n_ant]
                bonus = np.where(np.isinf(cd[filhos]), 1.0, np.minimum(4.0, 5.0 * cd[filhos]))
                np.add.at(score, origem[filhos_globais], 1.0 + bonus)

            tab[t] = [UX[idx], UF[idx], UC[idx]] 
        # ----- realocacao das vagas pelo score (maiores restos), com piso min_vagas
        if P["alocacao"] == "score":
            piso = P["min_vagas"]; resto = N - piso * T
            peso = np.maximum(score, 1e-9); exato = peso / peso.sum() * resto
            extra = np.floor(exato).astype(int); falta = resto - extra.sum()
            extra[np.argsort(-(exato - extra), kind="stable")[:falta]] += 1
            tam = piso + extra
        if callback is not None:
            callback(AX, AF, AC, frentes_com_restricao(AF, AC), T)

    info = dict(mascaras=masks, nomes=[nome_tabela(S) for S in masks], score_final=score.copy(), tamanhos_finais=tam.copy(),
                tabelas={nome_tabela(S): dict(X=tab[t][0].copy(), F=tab[t][1].copy(), CV=tab[t][2].copy()) for t, S in enumerate(masks)})
    return AX, AF, AC, info