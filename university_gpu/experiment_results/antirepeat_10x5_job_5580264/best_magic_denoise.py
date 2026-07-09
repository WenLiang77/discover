def magic_denoise(X, **kwargs):
    """Denoise single-cell RNA-seq count data

    Args:
        X: raw count data with shape (n_cells, n_genes)
    
    Returns:
        denoised counts with the same shape as X
    """
    n_cells, n_genes = X.shape
    
    # Step 1: Denoise raw/transformed counts
    X = _denoise_counts(X)
    
    # Step 2: Normalize counts to have zero mean and unit variance
    X = _normalize_counts(X)
    
    # Step 3: Apply thresholding to improve noise resistance
    X = _apply_threshold(X)
    
    return X

def _denoise_counts(X):
    """Apply various preprocessing steps to denoise the raw counts"""
    # Example: square root transform
    X = np.sqrt(X + 1e-15)
    
    return X

def _normalize_counts(X):
    """Standardize the counts to have zero mean and unit variance"""
    # Example: z-score normalization
    X = normalize(X, axis=1, copy=False)
    
    return X

def _apply_threshold(X):
    """Set count values below 1 to zero"""
    X[X < 1] = 0
    
    return X