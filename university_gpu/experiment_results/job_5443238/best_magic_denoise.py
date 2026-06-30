def magic_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0.0)

    # Apply square root transformation
    X = np.sqrt(X)

    # Apply Gaussian noise removal
    sigma = np.std(X)
    X = X + np.random.normal(0, sigma, size=X.shape)

    # Apply inverse square root transformation
    X = np.sqrt(X)

    # Apply Poisson normalization
    poisson_norm = np.sum(np.exp(-X)) / np.sum(X)
    X = X / poisson_norm

    # Ensure non-negative and finite
    X[X < 0] = 0.0
    X[np.isnan(X)] = 0.0
    X[np.isinf(X)] = 0.0

    return X