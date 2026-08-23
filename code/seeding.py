import numpy as np

# Global seed
SEED = 2023

def create_seed_sequences(names: list[str]) -> dict[str, np.random.SeedSequence]:
    """Create a dictionary of seed sequences for reproducibility.

    Parameters
    ----------
    names : list[str]
        Names for which to create seed sequences.
    
    Returns
    -------
    dict[str, np.random.SeedSequence]
        A dictionary mapping each name to a unique seed sequence.
    """
    return dict(zip(names, np.random.SeedSequence(SEED).spawn(len(names))))

def rng(seed_seq: np.random.SeedSequence) -> np.random.Generator:
    """Create a random number generator from a seed sequence.

    Parameters
    ----------
    seed_seq : np.random.SeedSequence
        The seed sequence to use for creating the random number generator.
    
    Returns
    -------
    np.random.Generator
        A random number generator initialized with the given seed sequence.
    """
    return np.random.default_rng(seed_seq)