import random


def boot_ci(diffs, n=10000, seed=13):
    """95% bootstrap CI on the mean of paired differences."""
    if not diffs:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]
