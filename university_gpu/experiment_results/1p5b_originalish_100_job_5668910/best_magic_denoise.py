def magic_denoise(X, **kwargs):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0.0)
    
    # Normalize counts to mean 0 and std 1
    X = normalize(X, axis=1, norm='l2')
    
    # Apply square root transformation to stabilize Poisson distribution
    X = np.sqrt(X)
    
    # Apply Poisson loss constraint to ensure non-negative counts
    X[X < 1] = 0
    
    return X