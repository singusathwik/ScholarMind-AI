import numpy as np


def paired_bootstrap(left, right, seed=42, repetitions=10000):
    differences = np.asarray(right) - np.asarray(left)
    if not len(differences):
        raise ValueError("No paired observations")
    rng = np.random.default_rng(seed)
    estimates = np.array([rng.choice(differences, len(differences)).mean() for _ in range(repetitions)])
    return {
        "mean_difference": float(differences.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "queries": len(differences),
        "seed": seed,
        "bootstrap_samples": repetitions,
        "method": "paired percentile bootstrap; unadjusted for multiple comparisons",
    }
