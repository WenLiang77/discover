def magic_denoise(X, **kwargs):
    X = np.array(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0.0)

    # Apply square root transformation
    X = np.sqrt(X)

    # Add small positive constant to avoid numerical instability
    X += 1e-6

    # Normalize the data
    X /= np.sum(X, axis=1, keepdims=True)

    return X