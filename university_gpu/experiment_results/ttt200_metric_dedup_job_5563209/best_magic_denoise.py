def magic_denoise(X, budget_s=None, random_state=None, knn=None, t=None, n_pca=None, solver='exact', decay=0.1, knn_max=None, n_jobs=-1):
    if budget_s is None:
        budget_s = float('inf')
    
    if random_state is None:
        random_state = np.random.randint(0, 10000)
    
    if knn is None:
        knn = 5
    
    if t is None:
        t = 1.0
    
    if n_pca is None:
        n_pca = min(X.shape[0], X.shape[1])
    
    if solver == 'exact':
        X_scaled = StandardScaler().fit_transform(X)
        U, s, Vh = np.linalg.svd(X_scaled, full_matrices=False)
        X_denoised = (U @ np.diag(np.reciprocal(s))) @ Vh
        X_denoised = normalize(X_denoised, axis=1, norm='l2')
    elif solver == 'pca':
        pca = PCA(n_components=n_pca).fit(X)
        X_denoised = pca.transform(X)
        X_denoised = normalize(X_denoised, axis=1, norm='l2')
    else:
        raise ValueError("Unsupported solver type.")
    
    X_denoised = np.clip(X_denoised, 0.0, np.inf)
    
    # Calculate Poisson loss
    poisson_loss = np.mean(poisson.pmf(np.exp(X_denoised), X))
    
    while poisson_loss > 0.97:
        X_denoised *= 1.01
        poisson_loss = np.mean(poisson.pmf(np.exp(X_denoised), X))
    
    X_denoised = np.round(X_denoised)
    X_denoised[X_denoised < 0] = 0
    
    # Normalize counts based on variance stabilizing square root transformation
    variances = np.var(X_denoised, axis=1, keepdims=True)
    X_denoised /= np.sqrt(variances + 1e-6)
    
    return X_denoised