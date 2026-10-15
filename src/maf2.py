# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from moeabench.mops import BaseMop

class MaF2(BaseMop):
    """
    MaF2 (DTLZ2BZ) benchmark problem from CEC 2017.
    Concave Pareto front with independent convergence barriers (g_i) for each objective.
    """
    def __init__(self, **kwargs):
        # Em problemas DTLZ/MaF, K costuma ser o número de variáveis de distância.
        # No CEC 2017, MaF2 geralmente utiliza K = 9 ou 10 para N = M + K - 1.
        self.K = kwargs.pop('K', 10)
        m_val = kwargs.get('M', 3)
        if 'N' not in kwargs:
            kwargs['N'] = m_val + self.K - 1
            
        super().__init__(**kwargs)

    def validate(self):
        """
        MaF2 exige no mínimo M variáveis de decisão:
        M-1 para definir a posição angular (theta) e pelo menos 1 para a distância (g).
        """
        super().validate()
        if self.N < self.M:
            raise ValueError(
                f"MaF2 exige N >= M variáveis para manter sua estrutura matemática.\n"
                f"M-1 variáveis são usadas para a posição esférica, "
                f"e o restante para a convergência g_i (fornecido N={self.N}, M={self.M})."
            )

    def evaluation(self, X, n_ieq_constr=0):
        X = np.atleast_2d(X)
        batch_size = X.shape[0]
        M = self.M
        D = self.N
        
        # 1. Variáveis de Posição (X_pos)
        X_pos = X[:, :M - 1]
        
        # Transformação de theta_i conforme especificação oficial do MaF2:
        # \theta_i = (\pi / 2) * ( (x_i / 2) + 1/4 )
        theta = (np.pi / 2.0) * ((X_pos / 2.0) + 0.25)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 2. Variáveis de Distância (X_dist)
        X_dist = X[:, M - 1:]
        
        # 3. Partição para calcular o vetor de convergência g_i[cite: 4]
        g_matrix = np.zeros((batch_size, M))
        term_floor = (D - M + 1) // M
        
        for i in range(M):
            # Mapeamento do índice 1-based da fórmula[cite: 4] para 0-based no subarray X_dist
            col_start = i * term_floor
            col_end = (i + 1) * term_floor if i < M - 1 else X_dist.shape[1]
            
            if col_start < X_dist.shape[1]:
                group_vars = X_dist[:, col_start:col_end]
                # A transformação \left( \frac{x_j}{2} + \frac{1}{4} \right) - 0.5[cite: 4]
                transformed = ((group_vars / 2.0) + 0.25) - 0.5
                g_matrix[:, i] = np.sum(transformed**2, axis=1)
            else:
                g_matrix[:, i] = 0.0

        # 4. Cálculo Vetorizado dos Objetivos Baseados em Seno/Cosseno[cite: 3]
        F = np.zeros((batch_size, M))
        
        for i in range(M):
            if i == 0:
                # f_1 = cos(\theta_1) ... cos(\theta_{M-1})[cite: 3]
                F[:, i] = np.prod(cos_theta, axis=1)
            elif i == M - 1:
                # f_M = sin(\theta_1)[cite: 3]
                F[:, i] = sin_theta[:, 0]
            else:
                # f_{M-i} intermediários[cite: 3]
                n_cos = M - 1 - i
                term_cos = np.prod(cos_theta[:, :n_cos], axis=1) if n_cos > 0 else 1.0
                term_sin = sin_theta[:, n_cos]
                F[:, i] = term_cos * term_sin
        
        # 5. Aplicação individual do g_i respectivo a cada objetivo[cite: 3]
        F = F * (1.0 + g_matrix)
        
        return {'F': F}

    def ps(self, n_points=100):
        """
        Amostragem analítica do Pareto Set (PS) do MaF2.
        """
        M = self.M
        N = self.N
        res = np.zeros((n_points, N))
        rng = np.random.RandomState(42)
        
        # As M-1 primeiras variáveis podem assumir qualquer valor em [0, 1]
        res[:, :M-1] = rng.random((n_points, M - 1))
        
        # Para que g_i = 0, a transformação (x_j/2 + 0.25) - 0.5 deve ser igual a 0.[cite: 4]
        # Resolvendo: x_j/2 + 0.25 = 0.5  =>  x_j/2 = 0.25  =>  x_j = 0.5
        # Portanto, o ótimo global no espaço de decisão é x_j = 0.5 para j >= M.
        res[:, M-1:] = 0.5
        
        return res

    def get_K(self):
        return self.K