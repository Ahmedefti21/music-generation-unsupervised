import numpy as np

def calculate_rhythm_diversity(piano_roll):
    """
    Calculates the rhythm diversity score based on note onset patterns.
    Higher score means more complex/varied rhythms.
    """
    # Detect onsets: 1 if note starts at this step
    padded = np.pad(piano_roll > 0.5, ((1, 0), (0, 0)), mode='constant')
    onsets = (np.diff(padded.astype(int), axis=0) == 1).any(axis=1)
    
    if np.sum(onsets) == 0:
        return 0.0
    
    # Calculate diversity as entropy of note durations or patterns
    # Simplified: ratio of active time steps to total onsets
    return np.std(onsets.astype(float))

def repetition_ratio(piano_roll, window_size=16):
    """Measures how much the rhythm repeats within a sequence."""
    padded = np.pad(piano_roll > 0.5, ((1, 0), (0, 0)), mode='constant')
    onsets = (np.diff(padded.astype(int), axis=0) == 1).any(axis=1).astype(int)
    
    if len(onsets) < window_size * 2:
        return 1.0
        
    w1 = onsets[:window_size]
    w2 = onsets[window_size:2*window_size]
    
    return np.mean(w1 == w2)
