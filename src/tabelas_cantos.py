"""Busca de cantos por tabelas adaptativas (fase 1) e distribuição por ângulo (fase 2).

Um CANTO é um máximo local do ângulo ao eixo central (1, ..., 1) sobre a fronteira de Pareto.

ESTRUTURA
    K + 1 tabelas, cada uma com mu = N // (K + 1) indivíduos:
      - a tabela central mira sempre o eixo central;
      - as K tabelas periféricas miram as direções de K âncoras. Cada tabela periférica tem
        identidade fixa (a tabela k é sempre a mesma).
    Âncoras permanentes: pontos onde tabelas convergiram. A lista cresce ao longo da busca.

    Eixo central: (1, ..., 1), nos objetivos normalizados. Opcionalmente (eixo_variavel=True), ele é
    reestimado a cada intervalo_eixo do orçamento, como a direção da média dos melhores pontos das
    tabelas periféricas e das âncoras permanentes.

    Os objetivos são sempre normalizados: F' = (F - ideal) / (nadir - ideal), com o ideal e o
    nadir estimados pela primeira frente não-dominada (factível) do conjunto em questão. Com
    isso, a busca é invariante à escala dos objetivos. Ângulos e notas são calculados sobre F'.

    A nota de um ponto x numa tabela de direção w é o Tchebycheff aumentado
        nota(x) = max_i F'_i(x) / w_i + rho * sum_i F'_i(x).
    Menor é melhor. Pontos factíveis vêm sempre antes; entre infactíveis, vale a menor violação.

FASE 1 (até frac_fase1 do orçamento): busca dos cantos, a cada geração
    1. Pais e filhos: cada tabela gera mu filhos. Para cada filho, com probabilidade
       1 - p_aleatorio os dois pais saem de torneios binários dentro da tabela; com probabilidade
       p_aleatorio, os dois são sorteados da população inteira (a união das tabelas). O filho é o
       SBX entre os dois pais, seguido da mutação polinomial.
    2. Candidatos: os membros de todas as tabelas mais todos os filhos.
    3. Direções. Escore pela soma = ângulo ao centro + soma dos ângulos às âncoras (permanentes
       e já escolhidas nesta geração); escore pelo menor ângulo = ângulo ao centro + menor ângulo
       a uma dessas âncoras.
         - Uma tabela não liberada escolhe a nova âncora só dentro da sua célula: os candidatos
           mais próximos da sua âncora anterior do que da âncora anterior de qualquer outra tabela
           (âncoras anteriores coincidentes compartilham a célula). Vence o maior escore pela soma.
         - Uma tabela recém-liberada, ou com a célula vazia, e todas na primeira geração, escolhem
           entre todos os candidatos, pelo maior escore pelo menor ângulo.
    4. Sobreviventes: cada tabela fica com os mu candidatos de menor nota na sua direção.
    5. Convergência de cada tabela. Cada tabela periférica guarda o recorde do ângulo do seu
       melhor ponto ao eixo central e o ponto recordista (o melhor ponto que ela já teve, nos
       objetivos originais; a cada geração, trocado pelo melhor ponto atual se este tiver nota
       menor, com os dois avaliados com a normalização e a direção atuais). Com JANELA gerações
       de histórico, a tabela é comparada com ela mesma JANELA gerações atrás: progrediu se o
       recorde de ângulo subiu mais de TOLERANCIA (relativo) ou se o recordista melhorou a nota
       em mais de TOLERANCIA (o antigo reavaliado com a normalização e a direção atuais).
       Se não progrediu, o recordista vira âncora permanente (com a sua direção nesse momento,
       que fica fixa), a tabela é liberada e o seu histórico recomeça.
    A fase 1 termina ao atingir frac_fase1 do orçamento. Só as âncoras permanentes não-dominadas
    (em relação às próprias âncoras e aos membros das tabelas) passam para a fase 2.

FASE 2 (o resto do orçamento): distribuição
    População principal P, de N indivíduos (no início, os membros das tabelas e as âncoras
    permanentes). A cada geração:
    1. Pais e filhos: frac_tabelas dos filhos vêm das tabelas (como na fase 1); o resto vem da
       população, com os dois pais sorteados ao acaso. SBX e mutação polinomial.
    2. As tabelas continuam: direções (com memória) e membros, entre tabelas, população e filhos.
       Na fase 2 não há novas fixações nem liberações.
    3. Âncoras permanentes: cada uma é trocada pelo ponto de menor nota na sua direção fixa, se
       esse ponto tiver nota menor que ela (com a normalização da geração). A direção não muda.
    4. Sobreviventes: frentes não-dominadas (factíveis primeiro); a frente que não cabe inteira é
       truncada pela seleção por ângulo (ABS), com os já escolhidos e as âncoras (o melhor de cada
       tabela e as âncoras permanentes) entrando primeiro.

RESULTADO
    A população final (N indivíduos) e, em info, as âncoras permanentes, o melhor ponto de cada
    tabela, o momento do fim da fase 1 e as avaliações gastas.
"""
from collections import namedtuple

import numpy as np

JANELA = 40          # gerações entre as duas situações comparadas na convergência de uma tabela
TOLERANCIA = 1e-3    # progresso relativo mínimo (no recorde de ângulo ou na nota) que conta

# Uma tabela é um conjunto de indivíduos (variáveis X, objetivos F, violação CV),
# sempre ordenado da melhor para a pior nota na direção da tabela.
Tabela = namedtuple("Tabela", ["X", "F", "CV"])


# =============================================================================
# Operadores genéticos (domínio [0, 1])
# =============================================================================
def sbx_crossover(pais_a, pais_b, pc=0.9, eta=20):
    """SBX com limites (Deb e Agrawal). Cada variável cruza com probabilidade 0,5."""
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
    cruza = (np.random.rand(*pais_a.shape) < 0.5) & (np.abs(pais_a - pais_b) > 1e-14) \
            & (np.random.rand(pais_a.shape[0], 1) < pc)
    return np.where(cruza, c1, pais_a), np.where(cruza, c2, pais_b)


def polynomial_mutation(X, eta=20):
    """Mutação polinomial com limites (Deb), probabilidade 1/n_var por variável."""
    muta = np.random.rand(*X.shape) < 1.0 / X.shape[1]
    u = np.random.rand(*X.shape)
    dq1 = (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dq2 = 1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(np.where(muta, X + np.where(u < 0.5, dq1, dq2), X), 0, 1)


# =============================================================================
# Ordenação não-dominada (usada pelo wrapper para separar a frente final)
# =============================================================================
def _frentes(F):
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=int)
    menor_ou_igual = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    domina = menor_ou_igual & ~menor_ou_igual.T          # i domina j: i <= j em tudo e j não é <= i em tudo
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
    """Factíveis ordenados em frentes não-dominadas; infactíveis depois, por violação crescente."""
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


# =============================================================================
# Geometria
# =============================================================================
def unitarios(V):
    """Cada linha de V dividida pela sua norma."""
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def angulos(U, V):
    """Ângulos (em radianos) entre as linhas de U e as de V, ambos unitários."""
    return np.arccos(np.clip(U @ V.T, -1, 1))


def ponto_ideal(F, CV):
    """Menor valor de cada objetivo entre os pontos factíveis (ou entre todos, se não houver)."""
    factivel = CV <= 0
    return (F[factivel] if factivel.any() else F).min(axis=0)


def primeira_frente(F, CV):
    """Máscara da primeira frente, igual a frentes_com_restricao(F, CV) == 0, sem ordenar o resto:
    os factíveis que nenhum outro factível domina; se não houver factíveis, os de menor violação."""
    factivel = CV <= 0
    mascara = np.zeros(len(F), dtype=bool)
    if factivel.any():
        G = F[factivel]
        menor_ou_igual = (G[:, None, :] <= G[None, :, :]).all(axis=2)
        domina = menor_ou_igual & ~menor_ou_igual.T      # i domina j: i <= j em tudo e j não é <= i em tudo
        mascara[np.flatnonzero(factivel)] = ~domina.any(axis=0)
    elif len(F):
        mascara = CV == CV.min()
    return mascara


def referencia(F, CV, frente=None):
    """Ideal e escala (nadir - ideal) para normalizar os objetivos. O nadir é o máximo de cada
    objetivo na primeira frente não-dominada (factível); a máscara da frente pode ser passada
    pronta, para não ser recalculada."""
    ideal = ponto_ideal(F, CV)
    if frente is None:
        frente = primeira_frente(F, CV)
    escala = np.maximum(F[frente].max(axis=0) - ideal, 1e-12)
    return ideal, escala


def nota(F_transladado, CV, w, rho):
    """Tchebycheff aumentado na direção w; infactíveis recebem uma nota enorme mais a violação."""
    tchebycheff = (F_transladado / w).max(axis=1) + rho * F_transladado.sum(axis=1)
    return np.where(CV <= 0, tchebycheff, 1e12 + CV)


# =============================================================================
# Passos do algoritmo
# =============================================================================
def escolhe_ancoras(F_transladado, K):
    """Escolhe K âncoras, uma a uma, pelo maior escore:
    ângulo ao eixo central + soma dos ângulos às âncoras já escolhidas."""
    M = F_transladado.shape[1]
    U = unitarios(F_transladado)
    escore = angulos(U, (np.ones(M) / np.sqrt(M))[None])[:, 0].copy()
    ancoras = []
    for _ in range(min(K, len(F_transladado))):
        disponivel = escore.copy()
        disponivel[ancoras] = -np.inf
        nova = int(np.argmax(disponivel))
        ancoras.append(nova)
        escore += angulos(U, U[nova:nova + 1])[:, 0]
    return np.array(ancoras)


def escolhe_ancoras_com_memoria(F_transladado, K, direcoes_antigas, livres, repulsoras, eixo=None):
    """Escolhe as K âncoras (uma por tabela periférica).

    Há dois escores para cada candidato:
      - escore pela soma:          ângulo ao centro + soma dos ângulos às âncoras;
      - escore pelo menor ângulo:  ângulo ao centro + menor ângulo a uma âncora;
    onde "âncoras" são as âncoras permanentes e as já escolhidas nesta geração.

    Uma tabela não liberada escolhe só dentro da sua célula (os candidatos mais próximos da sua
    âncora anterior do que da âncora anterior de qualquer outra tabela não liberada; tabelas com
    âncoras anteriores coincidentes compartilham a célula), pelo maior escore pela soma.
    Tabelas liberadas, células vazias e a primeira geração escolhem entre todos os candidatos,
    pelo maior escore pelo menor ângulo.
    Devolve, para cada tabela k, o índice da sua nova âncora."""
    M = F_transladado.shape[1]
    U = unitarios(F_transladado)
    eixo = np.ones(M) / np.sqrt(M) if eixo is None else eixo
    ao_centro = angulos(U, eixo[None])[:, 0]
    soma = ao_centro.copy()                       # escore pela soma
    menor = np.full(len(U), np.inf)               # menor ângulo a uma âncora
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
    # tabelas não liberadas: dentro da célula, pelo escore pela soma
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
    # tabelas liberadas, células vazias e primeira geração: entre todos, pelo menor ângulo
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
    """Passos 3 e 4: escolhe as direções e faz cada tabela ficar com os mu melhores candidatos.
    Devolve as direções (a primeira é a do eixo central) e as tabelas."""
    M = F.shape[1]
    ideal, escala = referencia(F, CV)
    F_transladado = (F - ideal) / escala                        # objetivos normalizados
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
    """Passo 1: cada tabela gera mu filhos por SBX e mutação polinomial. Para cada filho, com
    probabilidade 1 - p_aleatorio os dois pais saem de torneios binários dentro da tabela; com
    probabilidade p_aleatorio, os dois são sorteados da população inteira."""
    if normalizacao is None:
        normalizacao = referencia(np.vstack([t.F for t in tabelas]), np.concatenate([t.CV for t in tabelas]))
    ideal, escala = normalizacao
    populacao = np.vstack([t.X for t in tabelas])
    filhos = []
    for k, tabela in enumerate(tabelas):
        notas = nota((tabela.F - ideal) / escala, tabela.CV, direcoes[k], rho)
        # pais A e B: torneios binários dentro da tabela (vence a menor nota)
        i, j = rng.integers(0, len(tabela.X), (2, mu))
        pais_a = tabela.X[np.where(notas[i] <= notas[j], i, j)].copy()
        i, j = rng.integers(0, len(tabela.X), (2, mu))
        pais_b = tabela.X[np.where(notas[i] <= notas[j], i, j)].copy()
        # com probabilidade p_aleatorio, os dois pais são sorteados da população inteira
        for r in np.flatnonzero(rng.random(mu) < p_aleatorio):
            pais_a[r] = populacao[rng.integers(len(populacao))]
            pais_b[r] = populacao[rng.integers(len(populacao))]
        filho, _ = sbx_crossover(pais_a, pais_b)
        filhos.append(polynomial_mutation(filho))
    return np.vstack(filhos)


def uniao(*conjuntos):
    """Junta conjuntos (X, F, CV), removendo indivíduos repetidos."""
    X, posicoes = np.unique(np.vstack([c[0] for c in conjuntos]), axis=0, return_index=True)
    F = np.vstack([c[1] for c in conjuntos])[posicoes]
    CV = np.concatenate([c[2] for c in conjuntos])[posicoes]
    return X, F, CV


def situacao_das_tabelas(tabelas, eixo=None):
    """Para a verificação de parada: o ângulo (em graus) do melhor ponto de cada tabela periférica
    ao eixo central, o melhor ponto (F, CV) de cada uma e a normalização atual (ideal, escala)."""
    M = tabelas[0].F.shape[1]
    ideal, escala = referencia(np.vstack([t.F for t in tabelas]), np.concatenate([t.CV for t in tabelas]))
    perifericas = tabelas[1:]
    melhores = np.array([t.F[0] for t in perifericas])
    eixo_central = np.ones(M) / np.sqrt(M) if eixo is None else eixo
    angulo_ao_centro = np.degrees(angulos(unitarios((melhores - ideal) / escala), eixo_central[None])[:, 0])
    melhor_ponto = [(t.F[0], t.CV[0], t.X[0]) for t in perifericas]
    return angulo_ao_centro, melhor_ponto, (ideal, escala)


def nota_do_ponto(ponto, w, normalizacao, rho):
    """Nota de um ponto (F, CV) na direção w, com a normalização (ideal, escala) dada."""
    F, CV = ponto[0], ponto[1]
    ideal, escala = normalizacao
    return nota((F[None] - ideal) / escala, np.array([CV]), w, rho)[0]


def eixo_pelas_extremidades(F_extremidades, F, CV):
    """Direção da média das extremidades já encontradas (melhores das tabelas e âncoras
    permanentes), com os objetivos normalizados pela referência dos candidatos."""
    ideal, escala = referencia(F, CV)
    return unitarios(np.maximum(((F_extremidades - ideal) / escala).mean(axis=0), 1e-6)[None])[0]


def selecao_por_angulo(F_normalizado, iniciais, n):
    """ABS: os pontos em iniciais entram primeiro (até n); depois, um por vez, entra o ponto de
    maior ângulo ao ponto já escolhido mais próximo dele."""
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


# =============================================================================
# O algoritmo
# =============================================================================
def executa(avaliar, n_obj, n_var, N, n_ger, seed=0, K=None, rho=0.1, p_aleatorio=0.1,
            frac_fase1=0.7, frac_tabelas=0.1, eixo_variavel=False, intervalo_eixo=0.1, callback=None):
    """Busca de cantos por tabelas adaptativas (fase 1) seguida de distribuição por ângulo (fase 2).

    avaliar(X) -> (F, CV), com X em [0, 1]^n_var. Orçamento: N * (n_ger + 1) avaliações.
    K: número de tabelas periféricas (padrão 2M).
    rho: peso do termo de soma do Tchebycheff aumentado (0 = Tchebycheff puro).
    p_aleatorio: probabilidade de os dois pais de um filho de tabela virem da população inteira.
    frac_fase1: fração do orçamento usada pela fase 1.
    frac_tabelas: fração dos filhos da fase 2 gerada pelas tabelas (o resto vem da população).
    eixo_variavel: se True, o eixo central é reestimado a cada intervalo_eixo do orçamento, como a
        direção da média (nos objetivos normalizados) dos melhores pontos das tabelas periféricas e
        das âncoras permanentes; até a primeira atualização, é (1, ..., 1). Se False (padrão), o eixo
        é sempre (1, ..., 1).
    intervalo_eixo: fração do orçamento entre atualizações do eixo (só com eixo_variavel=True).
    callback(X, F, CV), se dado, é chamado a cada geração com a população atual.

    Devolve (X, F, CV, info): a população final (N indivíduos) e, em info, as âncoras
    permanentes, o melhor de cada tabela, o momento do fim da fase 1 e as avaliações gastas.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)                       # o SBX e a mutação usam o gerador global
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

    # ---------- inicialização
    X = rng.random((n_tabelas * mu, n_var))
    F, CV = avalia(X)
    eixo = np.ones(n_obj) / np.sqrt(n_obj)              # até a primeira atualização, o eixo é (1, ..., 1)
    proxima_atualizacao = intervalo_eixo * orcamento if eixo_variavel else np.inf
    def atualiza_eixo(F_, CV_):
        nonlocal eixo, proxima_atualizacao
        if avaliacoes >= proxima_atualizacao:
            extremidades = [t.F[0] for t in tabelas[1:]] + [p[0] for p in permanentes]
            eixo = eixo_pelas_extremidades(np.array(extremidades), F_, CV_)
            while proxima_atualizacao <= avaliacoes:
                proxima_atualizacao += intervalo_eixo * orcamento
    direcoes, tabelas = monta_tabelas(X, F, CV, K, mu, rho, eixo=eixo)
    angulo_ao_centro, melhor_ponto, normalizacao = situacao_das_tabelas(tabelas, eixo=direcoes[0])
    historico = [[(melhor_ponto[k], angulo_ao_centro[k])] for k in range(K)]   # por tabela: (recordista, recorde de ângulo)
    permanentes = []                     # âncoras permanentes: (F, CV, X)
    livres = set()                       # tabelas liberadas, que recebem nova âncora na próxima geração
    direcoes_permanentes = []            # direção de cada âncora permanente (fixa)

    # ---------- fase 1: busca dos cantos
    while avaliacoes + n_tabelas * mu <= frac_fase1 * orcamento:
        # passos 1 e 2: filhos e candidatos
        # (a normalização das tabelas é a mesma calculada no fim da geração anterior)
        filhos = gera_filhos(tabelas, direcoes, mu, rho, p_aleatorio, rng, normalizacao=normalizacao)
        F_filhos, CV_filhos = avalia(filhos)
        candidatos = uniao(*[(t.X, t.F, t.CV) for t in tabelas], (filhos, F_filhos, CV_filhos))

        # passos 3 e 4: direções (memória, repulsão das permanentes, tabelas livres) e sobreviventes
        F_perm = np.array([p[0] for p in permanentes]) if permanentes else None
        atualiza_eixo(candidatos[1], candidatos[2])
        direcoes, tabelas = monta_tabelas(*candidatos, K, mu, rho, direcoes_antigas=unitarios(direcoes[1:]),
                                          livres=livres, F_permanentes=F_perm, eixo=eixo)
        livres = set()

        # passo 5: recordes de cada tabela e verificação individual de estagnação
        angulo_ao_centro, melhor_ponto, normalizacao = situacao_das_tabelas(tabelas, eixo=direcoes[0])
        for k in range(K):
            w = direcoes[k + 1]
            if not historico[k]:                                      # tabela recém-liberada: recomeça
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
                if not (angulo_subiu or nota_melhorou):               # a tabela k convergiu: fixa e libera
                    permanentes.append(recordista)
                    ideal_n, escala_n = normalizacao                       # direção da âncora, fixa daqui em diante
                    direcoes_permanentes.append(np.maximum(unitarios(np.maximum((recordista[0][None] - ideal_n) / escala_n, 0))[0], 1e-6))
                    historico[k] = []
                    livres.add(k)

        if callback is not None:
            callback(*uniao(*[(t.X, t.F, t.CV) for t in tabelas]))

    fim_fase1 = avaliacoes

    # ---------- transição: só as âncoras permanentes não-dominadas passam para a fase 2
    # (dominância em relação às próprias âncoras e aos membros das tabelas; factíveis primeiro)
    n_antes = len(permanentes)
    if permanentes:
        F_todos = np.vstack([np.array([p[0] for p in permanentes])] + [t.F for t in tabelas])
        CV_todos = np.concatenate([np.array([p[1] for p in permanentes])] + [t.CV for t in tabelas])
        na_frente = primeira_frente(F_todos, CV_todos)[:len(permanentes)]
        permanentes = [p for p, ok in zip(permanentes, na_frente) if ok]
        direcoes_permanentes = [w for w, ok in zip(direcoes_permanentes, na_frente) if ok]

    # ---------- fase 2: distribuição
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
        # pais e filhos: frac_tabelas dos filhos vem das tabelas, o resto da população (pais sorteados)
        filhos_tabelas = gera_filhos(tabelas, direcoes, mu, rho, p_aleatorio, rng)
        filhos_tabelas = filhos_tabelas[rng.permutation(len(filhos_tabelas))[:n_filhos_tabelas]]
        i, j = rng.integers(0, len(P[0]), (2, N - len(filhos_tabelas)))
        filho, _ = sbx_crossover(P[0][i], P[0][j])
        filhos = np.vstack([filhos_tabelas, polynomial_mutation(filho)])
        F_filhos, CV_filhos = avalia(filhos)
        Q = (filhos, F_filhos, CV_filhos)

        # as tabelas continuam: direções (com memória) e membros, entre tabelas, população e filhos
        candidatos = conjunto(*[(t.X, t.F, t.CV) for t in tabelas], P, Q)
        F_perm = np.array([p[0] for p in permanentes]) if permanentes else None
        atualiza_eixo(candidatos[1], candidatos[2])
        direcoes, tabelas = monta_tabelas(*candidatos, K, mu, rho, direcoes_antigas=unitarios(direcoes[1:]),
                                          livres=set(), F_permanentes=F_perm, eixo=eixo)

        # todos os pontos disponíveis e a normalização da geração
        R = conjunto(P, Q, *[(t.X, t.F, t.CV) for t in tabelas], permanentes_como_conjunto())
        ideal, escala = referencia(R[1], R[2])
        R_normalizado = (R[1] - ideal) / escala

        # âncoras permanentes: trocadas por um ponto mais convergido na sua direção (a direção não muda)
        for p in range(len(permanentes)):
            w = direcoes_permanentes[p]
            notas_R = nota(R_normalizado, R[2], w, rho)
            melhor = int(np.argmin(notas_R))
            F_p, CV_p, X_p = permanentes[p]
            if notas_R[melhor] < nota((F_p[None] - ideal) / escala, np.array([CV_p]), w, rho)[0]:
                permanentes[p] = (R[1][melhor], R[2][melhor], R[0][melhor])

        # sobreviventes: frentes não-dominadas; a frente que não cabe é truncada pelo ABS, âncoras primeiro
        melhores_tabelas = (np.array([t.X[0] for t in tabelas[1:]]), np.array([t.F[0] for t in tabelas[1:]]),
                            np.array([t.CV[0] for t in tabelas[1:]]))
        S = conjunto(P, Q, melhores_tabelas, permanentes_como_conjunto())
        posicao = {S[0][k].tobytes(): k for k in range(len(S[0]))}
        ancoras = [posicao[x.tobytes()] for x in melhores_tabelas[0]] + [posicao[p[2].tobytes()] for p in permanentes]
        ancoras = list(dict.fromkeys(ancoras))
        rank = frentes_com_restricao(S[1], S[2])
        ideal_s, escala_s = referencia(S[1], S[2], frente=(rank == 0))
        S_normalizado = (S[1] - ideal_s) / escala_s
        escolhidos, r = [], 0
        while len(escolhidos) < N:
            frente = np.flatnonzero(rank == r); r += 1
            if len(frente) == 0:
                break
            if len(escolhidos) + len(frente) <= N:
                escolhidos += list(frente)
                continue
            na_frente = set(int(f) for f in frente)
            primeiros = list(dict.fromkeys(escolhidos + [a for a in ancoras if a in na_frente]))
            resto = [int(f) for f in frente if int(f) not in set(primeiros)]
            grupo = np.array(primeiros + resto)
            escolhidos = list(grupo[selecao_por_angulo(S_normalizado[grupo], list(range(len(primeiros))), N)])
        escolhidos = np.array(escolhidos)
        P = (S[0][escolhidos], S[1][escolhidos], S[2][escolhidos])

        if callback is not None:
            callback(*P)

    # ---------- resultado
    perifericas = tabelas[1:]
    info = dict(ancoras_X=np.array([t.X[0] for t in perifericas]),
                ancoras_F=np.array([t.F[0] for t in perifericas]),
                permanentes_X=np.array([p[2] for p in permanentes]) if permanentes else np.empty((0, n_var)),
                permanentes_F=np.array([p[0] for p in permanentes]) if permanentes else np.empty((0, n_obj)),
                fim_fase1=fim_fase1, permanentes_descartadas=n_antes - len(permanentes),
                avaliacoes=avaliacoes, K=K)
    return P[0], P[1], P[2], info