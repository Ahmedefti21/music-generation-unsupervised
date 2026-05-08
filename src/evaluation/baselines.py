"""Baseline music generators for comparison.

Implements:
1. Random Note Generator (Naive)
2. Markov Chain Music Model
"""
import numpy as np
import pretty_midi
from pathlib import Path
from collections import defaultdict
import random


def generate_random_midi(output_path, num_notes=80, duration=4.0,
                          pitch_range=(48, 84), fs=16, program=0):
    """Naive Random Note Generator baseline.

    Generates music by sampling pitches and durations uniformly at random.
    No musical structure or memory — serves as lower-bound baseline.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    current_time = 0.0
    note_duration = duration / num_notes

    for _ in range(num_notes):
        pitch = random.randint(pitch_range[0], pitch_range[1])
        velocity = random.randint(60, 100)
        start = current_time
        end = current_time + note_duration * random.uniform(0.5, 2.0)
        note = pretty_midi.Note(
            velocity=velocity, pitch=pitch,
            start=float(start), end=float(end)
        )
        instrument.notes.append(note)
        current_time += note_duration

    midi.instruments.append(instrument)
    midi.write(str(output_path))


class MarkovChainModel:
    """First-order Markov Chain Music Model.

    Learns transition probabilities P(next_pitch | current_pitch)
    and P(next_duration | current_duration) from training MIDI files.
    Generates music by sampling from learned transitions.
    """

    def __init__(self, order=2):
        self.order = order
        self.pitch_transitions = defaultdict(lambda: defaultdict(int))
        self.duration_transitions = defaultdict(lambda: defaultdict(int))
        self.start_pitches = []
        self.start_durations = []

    def _quantize_duration(self, dur):
        """Quantize duration to nearest 0.125s grid."""
        return round(dur / 0.125) * 0.125

    def fit(self, midi_files, max_files=200):
        """Train Markov model from list of MIDI files."""
        trained = 0
        for midi_path in midi_files[:max_files]:
            try:
                midi = pretty_midi.PrettyMIDI(str(midi_path))
                for instrument in midi.instruments:
                    if instrument.is_drum:
                        continue
                    notes = sorted(instrument.notes, key=lambda n: n.start)
                    if len(notes) < self.order + 1:
                        continue

                    pitches = [n.pitch for n in notes]
                    durations = [self._quantize_duration(n.end - n.start) for n in notes]

                    # Record start states
                    self.start_pitches.append(tuple(pitches[:self.order]))
                    self.start_durations.append(tuple(durations[:self.order]))

                    # Build transition counts
                    for i in range(len(pitches) - self.order):
                        state = tuple(pitches[i:i+self.order])
                        next_p = pitches[i+self.order]
                        self.pitch_transitions[state][next_p] += 1

                        dur_state = tuple(durations[i:i+self.order])
                        next_d = durations[i+self.order]
                        self.duration_transitions[dur_state][next_d] += 1

                trained += 1
            except Exception:
                continue

        print(f"Markov model trained on {trained} files.")
        print(f"  Pitch states: {len(self.pitch_transitions)}")
        print(f"  Duration states: {len(self.duration_transitions)}")

    def _sample_next(self, transitions, state):
        """Sample next token from transition distribution."""
        if state in transitions and transitions[state]:
            choices = list(transitions[state].keys())
            counts = np.array(list(transitions[state].values()), dtype=float)
            probs = counts / counts.sum()
            return np.random.choice(choices, p=probs)
        else:
            # Fallback: uniform random pitch in piano range
            if isinstance(list(transitions.keys())[0][0] if transitions else (60,), int):
                return random.randint(48, 84)
            return 0.25

    def generate(self, output_path, num_notes=120, program=0):
        """Generate a MIDI file using learned Markov transitions."""
        if not self.start_pitches:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Pick a random starting state
        start_idx = random.randint(0, len(self.start_pitches) - 1)
        pitch_state = list(self.start_pitches[start_idx])
        dur_state = list(self.start_durations[start_idx])

        pitches = list(pitch_state)
        durations = list(dur_state)

        # Generate sequence
        for _ in range(num_notes - self.order):
            next_p = self._sample_next(self.pitch_transitions, tuple(pitch_state[-self.order:]))
            next_d = self._sample_next(self.duration_transitions, tuple(dur_state[-self.order:]))
            pitches.append(next_p)
            durations.append(next_d if isinstance(next_d, float) else 0.25)
            pitch_state.append(next_p)
            dur_state.append(next_d)

        # Build MIDI
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)

        current_time = 0.0
        for pitch, dur in zip(pitches, durations):
            dur = max(0.1, float(dur))
            note = pretty_midi.Note(
                velocity=80,
                pitch=int(np.clip(pitch, 0, 127)),
                start=float(current_time),
                end=float(current_time + dur)
            )
            instrument.notes.append(note)
            current_time += dur

        midi.instruments.append(instrument)
        midi.write(str(output_path))
