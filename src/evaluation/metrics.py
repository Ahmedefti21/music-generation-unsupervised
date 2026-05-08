"""Evaluation metrics for music generation quality assessment.

Implements all metrics from project spec Section 5:
- Pitch Histogram Similarity
- Rhythm Diversity Score
- Repetition Ratio
- (Human Listening Score is collected separately via survey)
"""
import numpy as np
import pretty_midi
from pathlib import Path
from collections import Counter


def load_midi_notes(midi_path):
    """Load all notes from a MIDI file. Returns list of (pitch, duration) tuples."""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                duration = round(note.end - note.start, 2)
                notes.append((note.pitch, duration))
        return notes
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return []


def pitch_histogram(notes, num_pitches=128):
    """Compute normalized pitch histogram from notes list."""
    hist = np.zeros(num_pitches)
    for pitch, _ in notes:
        if 0 <= pitch < num_pitches:
            hist[pitch] += 1
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def pitch_histogram_similarity(notes_gen, notes_ref):
    """Pitch Histogram Similarity per project spec.

    H(p, q) = (1/2) * sum_i |p_i - q_i|
    Lower is better (0 = identical distributions).
    """
    p = pitch_histogram(notes_gen)
    q = pitch_histogram(notes_ref)
    return 0.5 * np.sum(np.abs(p - q))


def rhythm_diversity_score(notes):
    """Rhythm Diversity Score per project spec.

    D_rhythm = #unique_durations / #total_notes
    Higher is better (more rhythmic variety).
    """
    if len(notes) == 0:
        return 0.0
    durations = [d for _, d in notes]
    unique = len(set(durations))
    total = len(durations)
    return unique / total


def repetition_ratio(notes, pattern_len=4):
    """Repetition Ratio per project spec.

    R = #repeated_patterns / #total_patterns
    Lower is better (less repetitive).
    Uses n-gram patterns of pitch sequences.
    """
    if len(notes) < pattern_len:
        return 0.0
    pitches = [p for p, _ in notes]
    patterns = [tuple(pitches[i:i+pattern_len]) for i in range(len(pitches) - pattern_len + 1)]
    if len(patterns) == 0:
        return 0.0
    counts = Counter(patterns)
    repeated = sum(1 for v in counts.values() if v > 1)
    return repeated / len(counts)


def evaluate_midi_file(midi_path, reference_notes=None):
    """Compute all metrics for a single MIDI file.

    Args:
        midi_path: Path to MIDI file
        reference_notes: Optional reference notes for pitch similarity
    Returns:
        dict of metric name -> value
    """
    notes = load_midi_notes(midi_path)

    results = {
        "num_notes": len(notes),
        "rhythm_diversity": rhythm_diversity_score(notes),
        "repetition_ratio": repetition_ratio(notes),
    }

    if reference_notes is not None:
        results["pitch_histogram_similarity"] = pitch_histogram_similarity(notes, reference_notes)

    return results


def evaluate_folder(midi_folder, reference_notes=None):
    """Evaluate all MIDI files in a folder, return mean metrics."""
    midi_folder = Path(midi_folder)
    midi_files = list(midi_folder.glob("*.mid")) + list(midi_folder.glob("*.midi"))

    if not midi_files:
        print(f"No MIDI files found in {midi_folder}")
        return {}

    all_results = []
    for f in midi_files:
        r = evaluate_midi_file(f, reference_notes)
        if r["num_notes"] > 0:
            all_results.append(r)

    if not all_results:
        return {}

    # Average across files
    avg = {}
    for key in all_results[0]:
        avg[key] = np.mean([r[key] for r in all_results if key in r])

    return avg
