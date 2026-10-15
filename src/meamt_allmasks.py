import numpy as np

# --- Chave de Prioridade (Deb / Violacao de Restricao) ---
def chaves(Tch, CV):
    return np.where(CV <= 0, Tch, 1e9 + CV)

# --- 1. SPEA2 Truncation Otimizado ---
def spea2_truncation(front_f, slots):
    n_points = len(front_f)
    if n_points <= slots:
        return np.arange(n_points, dtype=int)
    
    dists = np.linalg.norm(front_f[:, np.newaxis, :] - front_f[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)

    active = np.ones(n_points, dtype=bool)
    n_active = n_points

    while n_active > slots:
        sub_dists = dists[active][:, active]
        sorted_dists = np.sort(sub_dists, axis=1)
        worst_rel = np.lexsort(sorted_dists.T[::-1])[0]
        
        active_indices = np.where(active)[0]
        active[active_indices[worst_rel]] = False
        n_active -= 1

    return np.where(active)[0]

# --- 2. Remoção Rápida de Duplicados em Matrizes (com CV) ---
def remove_duplicates_fast(all_x, all_f_raw, all_cv):
    _, unique_indices = np.unique(
        np.ascontiguousarray(all_x).view(np.dtype((np.void, all_x.dtype.itemsize * all_x.shape[1]))), 
        return_index=True
    )
    unique_indices = np.sort(unique_indices)
    return all_x[unique_indices], all_f_raw[unique_indices], all_cv[unique_indices]

# --- 3. Fast Non-Dominated Sort Com Restrições (Deb) ---
def fast_nondenominated_sort_constrained(pop_f, pop_cv):
    n_pop = len(pop_f)
    
    diff = pop_f[:, None, :] - pop_f[None, :, :]
    obj_dom = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
    
    v_i = (pop_cv == 0)[:, None]
    v_j = (pop_cv == 0)[None, :]
    
    cv_i = pop_cv[:, None]
    cv_j = pop_cv[None, :]
    
    dom_matrix = (v_i & v_j & obj_dom) | (v_i & ~v_j) | (~v_i & ~v_j & (cv_i < cv_j))
    
    n = np.sum(dom_matrix, axis=0)
    S = [np.where(dom_matrix[p])[0] for p in range(n_pop)]

    fronts = []
    first_front = np.where(n == 0)[0].tolist()
    fronts.append(first_front)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts

# --- 4. Operadores Genéticos Vetorizados com Tratamento de Fronteira ---
def polynomial_mutation_batch(offspring, eta=20, bounds=(0, 1), indpb=None):
    mut = offspring.copy()
    low, up = bounds
    n, n_var = mut.shape
    if indpb is None:
        indpb = 1.0 / n_var
    
    mut_mask = np.random.rand(n, n_var) < indpb
    if not np.any(mut_mask):
        return mut

    x = mut[mut_mask]
    u = np.random.rand(len(x))
    
    delta1 = (x - low) / (up - low)
    delta2 = (up - x) / (up - low)
    
    delta_q = np.zeros_like(u)
    
    mask_low = u <= 0.5
    if np.any(mask_low):
        u_l = u[mask_low]
        d1 = delta1[mask_low]
        val = 2.0 * u_l + (1.0 - 2.0 * u_l) * ((1.0 - d1) ** (eta + 1.0))
        delta_q[mask_low] = (val ** (1.0 / (eta + 1.0))) - 1.0

    mask_high = ~mask_low
    if np.any(mask_high):
        u_h = u[mask_high]
        d2 = delta2[mask_high]
        val = 2.0 * (1.0 - u_h) + 2.0 * (u_h - 0.5) * ((1.0 - d2) ** (eta + 1.0))
        delta_q[mask_high] = 1.0 - (val ** (1.0 / (eta + 1.0)))

    mut[mut_mask] = x + delta_q * (up - low)
    return np.clip(mut, low, up)

def sbx_crossover_batch(p1, p2, pc=0.9, eta=20, bounds=(0, 1)):
    low, up = bounds
    n, n_var = p1.shape
    c1, c2 = p1.copy(), p2.copy()
    
    do_cross = (np.random.rand(n) <= pc)[:, None] & (np.random.rand(n, n_var) <= 0.5)
    if not np.any(do_cross):
        return c1, c2

    diff_mask = do_cross & (np.abs(p1 - p2) > 1e-14)
    if not np.any(diff_mask):
        return c1, c2

    x1 = np.minimum(p1[diff_mask], p2[diff_mask])
    x2 = np.maximum(p1[diff_mask], p2[diff_mask])
    
    u = np.random.rand(np.count_nonzero(diff_mask)) # type: ignore

    beta_low = 1.0 + (2.0 * (x1 - low) / (x2 - x1))
    alpha_low = 2.0 - beta_low ** (-(eta + 1.0))
    
    beta_up = 1.0 + (2.0 * (up - x2) / (x2 - x1))
    alpha_up = 2.0 - beta_up ** (-(eta + 1.0))

    betaq = np.zeros_like(u)
    
    mask_low = u <= (1.0 / alpha_low)
    betaq[mask_low] = (u[mask_low] * alpha_low[mask_low]) ** (1.0 / (eta + 1.0))
    betaq[~mask_low] = (1.0 / (2.0 - u[~mask_low] * alpha_low[~mask_low])) ** (1.0 / (eta + 1.0))

    c1_val = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
    
    mask_up = u <= (1.0 / alpha_up)
    betaq[mask_up] = (u[mask_up] * alpha_up[mask_up]) ** (1.0 / (eta + 1.0))
    betaq[~mask_up] = (1.0 / (2.0 - u[~mask_up] * alpha_up[~mask_up])) ** (1.0 / (eta + 1.0))

    c2_val = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

    swap = np.random.rand(np.count_nonzero(diff_mask)) <= 0.5 # type: ignore
    
    res1 = np.where(swap, c2_val, c1_val)
    res2 = np.where(swap, c1_val, c2_val)

    c1[diff_mask] = res1
    c2[diff_mask] = res2

    return np.clip(c1, low, up), np.clip(c2, low, up)

# --- 5. Funções de Avaliação Tchebycheff e Inicialização ---
def tch(F, mask_bool, rho):
    return F[:, mask_bool].max(axis=1) + rho * F.sum(axis=1)

def initiate(avaliar, n_pop, n_var, n_obj, rho, active_masks):
    n_masks = 2 ** n_obj - 1
    base, resto = divmod(n_pop, n_masks)
    tables_ind = [None] * (n_masks + 1)
    tables_fit = [None] * (n_masks + 1)
    tables_cv  = [None] * (n_masks + 1)
    tables_tch = [None] * (n_masks + 1)
    
    for mask in range(1, n_masks + 1):
        size = base + 1 if mask <= resto else base
        X = np.random.rand(size, n_var)
        F, CV = avaliar(X)
        tables_ind[mask] = X
        tables_fit[mask] = F
        tables_cv[mask]  = CV
        tables_tch[mask] = tch(F, active_masks[mask], rho)
        
    return tables_ind, tables_fit, tables_cv, tables_tch, n_masks

# --- Execução Principal ---
def executa(avaliar, n_obj, mu_fase1, n_var, n_pop, n_gen, rho=1e-4, pc=0.9, eta=20, snapshot_callback=None):
    active_masks = {
        m: np.array([(m >> i) & 1 for i in range(n_obj)], dtype=bool)
        for m in range(1, 2 ** n_obj)
    }

    tables_ind, tables_fit, tables_cv, tables_tch, n_masks = initiate(avaliar, n_pop, n_var, n_obj, rho, active_masks)
    fes_count = sum(len(tables_ind[m]) for m in range(1, n_masks + 1))

    if n_masks > n_pop:
        return 'Not enough population' 

    fase1_gen = int(n_gen * mu_fase1)
    
    # --- FASE 1: Convergência por Decomposição ---
    for gen in range(fase1_gen):
        p1_list, p2_list = [], []
        
        for mask in range(1, n_masks + 1):
            mu = len(tables_ind[mask])
            tch_arr = tables_tch[mask]
            cv_arr  = tables_cv[mask]
            
            keys = chaves(tch_arr, cv_arr)
            
            for _ in range(mu):
                i1, i2 = np.random.randint(0, mu, 2)
                p1_idx = i1 if keys[i1] <= keys[i2] else i2
                p1_list.append(tables_ind[mask][p1_idx])
                
                if np.random.rand() < 0.25 and n_masks > 1:
                    other = np.random.randint(1, n_masks + 1)
                    while other == mask:
                        other = np.random.randint(1, n_masks + 1)
                    p2_list.append(tables_ind[other][np.random.randint(len(tables_ind[other]))])
                else:
                    j1, j2 = np.random.randint(0, mu, 2)
                    p2_idx = j1 if keys[j1] <= keys[j2] else j2
                    p2_list.append(tables_ind[mask][p2_idx])

        P1 = np.array(p1_list)
        P2 = np.array(p2_list)
            
        C1, C2 = sbx_crossover_batch(P1, P2, pc, eta)
        offspring = np.vstack([C1, C2])
        offspring = polynomial_mutation_batch(offspring, eta, (0, 1), 1.0 / n_var)
        
        off_fit, off_cv = avaliar(offspring)
        fes_count += len(offspring)

        pool_x_sub  = np.vstack([tables_ind[m] for m in range(1, n_masks + 1)])
        pool_f_sub  = np.vstack([tables_fit[m] for m in range(1, n_masks + 1)])
        pool_cv_sub = np.hstack([tables_cv[m]  for m in range(1, n_masks + 1)])

        all_x  = np.vstack([pool_x_sub, offspring])
        all_f  = np.vstack([pool_f_sub, off_fit])
        all_cv = np.hstack([pool_cv_sub, off_cv])
        
        pool_x, pool_f, pool_cv = [], [], []

        for mask in range(1, n_masks + 1):
            mu = len(tables_ind[mask])
            all_tch = tch(all_f, active_masks[mask], rho)
            
            all_keys = chaves(all_tch, all_cv)
            best = np.argsort(all_keys, kind="stable")[:mu]
            
            tables_ind[mask] = all_x[best]
            tables_fit[mask] = all_f[best]
            tables_cv[mask]  = all_cv[best]
            tables_tch[mask] = all_tch[best]
            
            pool_x.append(tables_ind[mask])
            pool_f.append(tables_fit[mask])
            pool_cv.append(tables_cv[mask])
        
        if snapshot_callback is not None:
            pool_x_mat = np.vstack(pool_x)
            pool_f_mat = np.vstack(pool_f)
            pool_cv_mat = np.hstack(pool_cv)
            fronts = fast_nondenominated_sort_constrained(pool_f_mat, pool_cv_mat)
            snapshot_callback(pool_x_mat, pool_f_mat, fronts)

    # --- FIM DA FASE 1: Extração das Âncoras ---
    melhores = [int(np.argmin(chaves(tables_tch[m], tables_cv[m]))) for m in range(1, n_masks + 1)]
    anchors_x  = np.array([tables_ind[m][melhores[m - 1]] for m in range(1, n_masks + 1)])
    anchors_f  = np.array([tables_fit[m][melhores[m - 1]] for m in range(1, n_masks + 1)])
    anchors_cv = np.array([tables_cv[m][melhores[m - 1]] for m in range(1, n_masks + 1)])
    
    pop_x  = np.vstack([tables_ind[m] for m in range(1, n_masks + 1)])
    pop_f  = np.vstack([tables_fit[m] for m in range(1, n_masks + 1)])
    pop_cv = np.hstack([tables_cv[m] for m in range(1, n_masks + 1)])

    # --- FASE 2: Exploração e Espalhamento ---
    for gen in range(fase1_gen, n_gen):
        n = len(pop_x)
    
        i1, i2 = np.random.randint(0, n, (2, n // 2))
        P1 = pop_x[np.minimum(i1, i2)]

        # Torneio para P2: faz o mesmo para o segundo pai
        i3, i4 = np.random.randint(0, n, (2, n // 2))
        P2 = pop_x[np.minimum(i3, i4)]         
        C1, C2 = sbx_crossover_batch(P1, P2, pc, eta)
        offspring = np.vstack([C1, C2])
        
        if n % 2 != 0:
            p1_extra, p2_extra = pop_x[np.random.randint(0, n, 2)]
            c_extra, _ = sbx_crossover_batch(np.array([p1_extra]), np.array([p2_extra]), pc, eta)
            offspring = np.vstack([offspring, c_extra])

        offspring = polynomial_mutation_batch(offspring, eta, (0, 1), 1.0 / n_var)
        off_fit, off_cv = avaliar(offspring)
        fes_count += len(offspring)

        all_x = np.vstack([pop_x, offspring])
        all_f_raw = np.vstack([pop_f, off_fit])
        all_cv = np.hstack([pop_cv, off_cv])

        all_x, all_f_raw, all_cv = remove_duplicates_fast(all_x, all_f_raw, all_cv)
        fronts = fast_nondenominated_sort_constrained(all_f_raw, all_cv)
        
        # Normalização Isolada
        first_front_fit = all_f_raw[fronts[0]]
        v_znadir = np.max(first_front_fit, axis=0)
        v_zideal = np.min(first_front_fit, axis=0)
        denom = np.maximum(v_znadir - v_zideal, 1e-12)
        
        all_f_norm = (all_f_raw - v_zideal) / denom
        anchors_f_norm = (anchors_f - v_zideal) / denom

        # Atualização das Âncoras Com Restrições
        for mask in range(1, n_masks + 1):
            mask_tch = tch(all_f_norm, active_masks[mask], rho)
            mask_keys = chaves(mask_tch, all_cv)
            min_idx = np.argmin(mask_keys)
            new_min_key = mask_keys[min_idx]

            anchor_tch = tch(anchors_f_norm[mask - 1:mask], active_masks[mask], rho)[0]
            anchor_key = chaves(np.array([anchor_tch]), anchors_cv[mask - 1:mask])[0]

            if new_min_key < anchor_key:
                anchors_x[mask - 1] = all_x[min_idx]
                anchors_f[mask - 1] = all_f_raw[min_idx]
                anchors_cv[mask - 1] = all_cv[min_idx]

        # Seleção Ambiental
        next_pop_x  = []
        next_pop_f  = []
        next_pop_cv = []
        current_count = 0

        for i, front in enumerate(fronts):
            front_arr = np.array(front)
            if current_count + len(front_arr) <= n_pop:
                next_pop_x.append(all_x[front_arr])
                next_pop_f.append(all_f_raw[front_arr])
                next_pop_cv.append(all_cv[front_arr])
                current_count += len(front_arr)
            else:
                slots = n_pop - current_count
                chosen_relative_indexes = spea2_truncation(all_f_norm[front_arr], slots).astype(int)
                chosen_global_indexes = front_arr[chosen_relative_indexes]
                fronts[i] = list(chosen_global_indexes)
                next_pop_x.append(all_x[chosen_global_indexes])
                next_pop_f.append(all_f_raw[chosen_global_indexes])
                next_pop_cv.append(all_cv[chosen_global_indexes])
                break
            
        pop_x  = np.vstack(next_pop_x)
        pop_f  = np.vstack(next_pop_f)
        pop_cv = np.hstack(next_pop_cv)
        
        if snapshot_callback is not None:
            new_fronts = []
            offset = 0
            for block in next_pop_x:
                block_size = len(block)
                new_fronts.append(list(range(offset, offset + block_size)))
                offset += block_size
            
            snapshot_callback(pop_x, pop_f, new_fronts)

    return pop_f