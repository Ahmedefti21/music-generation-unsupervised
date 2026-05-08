import numpy as np

def calculate_pitch_histogram(piano_roll):
    """Calculates the frequency of each pitch in a piano roll."""
    # Sum across time axis
    counts = np.sum(piano_roll > 0.5, axis=0)
    total = np.sum(counts)
    if total == 0:
        return np.zeros_like(counts)
    return counts / total

def histogram_intersection(h1, h2):
    """Calculates the intersection between two histograms (Pitch Similarity)."""
    return np.sum(np.minimum(h1, h2))
