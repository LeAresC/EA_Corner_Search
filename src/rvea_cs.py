import numpy as np


# =============================================================================
# Operadores genéticos
# =============================================================================
def sbx_crossover(pais_a, pais_b, pc=0.9, eta=20):
    """SBX com limites (Deb e Agrawal), domínio [0, 1]. Cada variável cruza com probabilidade 0,5."""
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
    """Mutação polinomial com limites (Deb), domínio [0, 1], probabilidade 1/n_var por variável."""
    muta = np.random.rand(*X.shape) < 1.0 / X.shape[1]
    u = np.random.rand(*X.shape)
    dq1 = (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dq2 = 1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(np.where(muta, X + np.where(u < 0.5, dq1, dq2), X), 0, 1)


def reproducao(pop, n_filhos, pc=0.9, eta=20):
    """Gera n_filhos por SBX + mutação polinomial, com pais sorteados ao acaso."""
    idx_a, idx_b = np.random.randint(0, len(pop), (2, (n_filhos + 1) // 2))
    c1, c2 = sbx_crossover(pop[idx_a], pop[idx_b], pc=pc, eta=eta)
    return polynomial_mutation(np.vstack([c1, c2]), eta=eta)[:n_filhos]


# =============================================================================
# Geometria no espaço dos objetivos
# =============================================================================
def geometria(fit_trans, eixo_central=None):
    """Tudo o que a seleção precisa, calculado uma vez por geração.
    fit_trans: objetivos transladados pelo ponto ideal (sem dividir pelo nadir, como no RVEA)."""
    n_obj = fit_trans.shape[1]
    norma = np.maximum(np.linalg.norm(fit_trans, axis=1), 1e-12)
    unitarios = fit_trans / norma[:, None]
    if eixo_central is None:
        eixo_central = np.ones(n_obj) / np.sqrt(n_obj)
    theta_c = np.arccos(np.clip(unitarios @ eixo_central, -1, 1))       # ângulo de cada ponto ao centro
    ang = np.arccos(np.clip(unitarios @ unitarios.T, -1, 1))            # ângulo entre todos os pares
    np.fill_diagonal(ang, 0.0)                                          # ângulo de cada ponto a si mesmo: exatamente 0
    ang_sem_diagonal = ang.copy()
    np.fill_diagonal(ang_sem_diagonal, np.inf)
    espacamento = np.median(ang_sem_diagonal.min(axis=1))              # ângulo mediano ao vizinho mais próximo
    return norma, theta_c, ang, espacamento


# =============================================================================
# Seleção
# =============================================================================
RAIO_MINIMO = 1e-6   # radianos: acima do ruído numérico, para pontos alinhados (mesma direção) se compararem
def escolhe_centro(norma, theta_c, espacamento, t, T, n_obj, alpha=2):
    """APD original do RVEA, nichado em volta do eixo central."""
    R = max(2.0 * espacamento, RAIO_MINIMO)
    celula = np.flatnonzero(theta_c <= theta_c.min() + R)
    apd = (1 + n_obj * (t / T) ** alpha * theta_c[celula] / R) * norma[celula]
    return celula[np.argmin(apd)]


def selecao(norma, theta_c, ang, espacamento, n_sel, i_centro, t, T, forca_premio=0.95, p=1.0, r=0.5, ancora="vencedor"):
    """Fluxo de cada passo:
    1) acha o mais longe pelo ângulo (p_gap);
    2) calcula o nicho do p_gap;
    3) escolhe o sobrevivente do nicho pelo APD modificado;
    4) ancora no sobrevivente (ou, com ancora="p_gap", no p_gap);
    5) recalcula o min_theta com a nova âncora; repete.
    Devolve os índices dos n_sel sobreviventes."""
    n = len(norma)
    g = (t / T) ** p                                      # força do prêmio: cresce de 0 a 1 ao longo da fase
    livre = np.ones(n, dtype=bool)                        # ainda não escolhidos como sobreviventes
    min_theta = theta_c.copy()                            # a primeira âncora é o próprio eixo central
    sobreviventes = [i_centro]
    livre[i_centro] = False

    while len(sobreviventes) < n_sel and livre.any():
        candidatos = np.flatnonzero(livre)
        p_gap = candidatos[np.argmax(min_theta[candidatos])]                          # 1) o mais longe
        R = max(r * min_theta[p_gap], espacamento, RAIO_MINIMO)                        # 2) nicho do p_gap
        nicho = candidatos[(ang[p_gap, candidatos] <= R) | (candidatos == p_gap)]
        u = (theta_c[nicho] - theta_c[p_gap]) / R                                      # 3) APD modificado
        d = (1 - forca_premio * g * u) * norma[nicho]
        vencedor = nicho[np.argmin(d)]
        sobreviventes.append(vencedor)
        livre[vencedor] = False
        a = p_gap if ancora == "p_gap" else vencedor
        np.minimum(min_theta, ang[a], out=min_theta)                                   # 4-5) ancora e recalcula
    return np.array(sobreviventes)


def execute(n_gen, n_pop, n_obj, n_var, phase1=0.6, evaluate=None, alpha=2, pc=0.9, eta=20,
            forca_premio=0.3, p=1.0, r=0.5, ancora="vencedor", recentrar=None, callback=None):
    """evaluate(X) recebe a matriz (n, n_var) em [0, 1] e devolve os objetivos (n, n_obj).
    callback(pop, pop_fit), se dado, é chamado com a população inicial e depois de cada geração.
    Devolve a população final, os seus objetivos e o número de avaliações gastas."""
    if evaluate is None:
        raise ValueError("evaluate não foi definido")

    pop = np.random.rand(n_pop, n_var)
    pop_fit = evaluate(pop)
    avaliacoes = n_pop
    if callback is not None:
        callback(pop, pop_fit)
    T = int(n_gen * phase1)
    eixo_central = np.ones(n_obj) / np.sqrt(n_obj)
    passo_recentrar = max(1, int(recentrar * T)) if recentrar else None

    for t in range(1, T):
        # 1) reprodução
        offspring = reproducao(pop, n_pop, pc=pc, eta=eta)
        offspring_fit = evaluate(offspring)
        avaliacoes += len(offspring)

        # 2) pais + filhos, sem duplicatas
        uniao, idx_unicos = np.unique(np.vstack((pop, offspring)), axis=0, return_index=True)
        uniao_fit = np.vstack((pop_fit, offspring_fit))[idx_unicos]

        # 3) translação pelo ponto ideal (como no RVEA, sem normalizar pelo nadir)
        z_ideal = uniao_fit.min(axis=0)
        fit_trans = uniao_fit - z_ideal

        # 4) centro e seleção
        if passo_recentrar and t % passo_recentrar == 0:          # eixo pela faixa dos objetivos da população
            faixa = np.maximum(pop_fit.max(axis=0) - pop_fit.min(axis=0), 1e-12)
            eixo_central = faixa / np.linalg.norm(faixa)
        norma, theta_c, ang, espacamento = geometria(fit_trans, eixo_central)
        i_centro = escolhe_centro(norma, theta_c, espacamento, t, T, n_obj, alpha)
        idx = selecao(norma, theta_c, ang, espacamento, n_pop, i_centro, t, T, forca_premio, p, r, ancora)
        pop, pop_fit = uniao[idx], uniao_fit[idx]
        if callback is not None:
            callback(pop, pop_fit)

    return pop, pop_fit, avaliacoes