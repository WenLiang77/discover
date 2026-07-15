def magic_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0.0)
    
    # Square root transform to stabilize variance
    X_sqrt = np.sqrt(X + 1)
    
    # Normalize data
    X_normalized = normalize(X_sqrt, norm='l1', axis=1)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X_normalized)
    
    # Reconstruct data using PCA
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Square root inverse transformation
    X_denoised = np.square(X_reconstructed)
    
    # Ensure non-negativity and finite values
    X_denoised = np.maximum(X_denoised, 0.0)
    
    # Push values < 1 toward zero
    X_denoised[X_denoised < 1] = 0
    
    return X_denoised