def magic_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0.0)

    # Normalization: Logarithmic transformation
    X_log = np.log(X + 1.0)

    # Denoising: Subtract mean across genes
    denoised_X = X_log - np.mean(X_log, axis=1, keepdims=True)

    # Reverse normalization
    denoised_X = np.exp(denoised_X)

    # Apply Poisson normalization
    poisson_norm = calculate_poisson_normalization(denoised_X)
    denoised_X = apply_poisson_normalization(denoised_X, poisson_norm)

    # Clip to ensure non-negativity and zero values
    denoised_X = np.clip(denoised_X, 0.0, None)

    return denoised_X

def calculate_poisson_normalization(X):
    """
    Calculate the Poisson normalization factor for each cell
    :param X: numpy array of shape (n_cells, n_genes) - denoised counts
    :return: float - Poisson normalization factor
    """
    return np.mean(np.sum(X, axis=1))

def apply_poisson_normalization(X, poisson_norm):
    """
    Apply Poisson normalization to the denoised counts
    :param X: numpy array of shape (n_cells, n_genes) - denoised counts
    :param poisson_norm: float - Poisson normalization factor
    :return: numpy array of shape (n_cells, n_genes) - normalized counts
    """
    return X / poisson_norm