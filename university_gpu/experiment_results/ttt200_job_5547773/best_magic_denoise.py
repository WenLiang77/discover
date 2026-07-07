def magic_denoise(X, **kwargs):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.maximum(X, 0.0)

    # Standardize data
    X_scaled = X / np.sum(X, axis=1, keepdims=True)

    # Apply square root transformation
    X_sqrt = np.sqrt(X_scaled)

    # Denoise by setting small values close to zero
    X_denoised = np.where(X_sqrt < 1.0, 0.0, X_sqrt)

    # Normalize back to log scale
    X_log = np.log(X_denoised + 1.0)

    return X_log