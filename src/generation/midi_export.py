import numpy as np
import pretty_midi

def piano_roll_to_midi(piano_roll, fs=16, program=0):
    """
    Converts a binary piano roll (T, 88) into a pretty_midi object.
    Pitches are offset by 21 (piano keys).
    """
    notes, pitches = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    for pitch_idx in range(pitches):
        # Find start and end of note segments
        active = piano_roll[:, pitch_idx] > 0.5
        padded = np.pad(active, (1, 1), mode='constant')
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch_idx + 21,
                start=start / fs,
                end=end / fs
            )
            instrument.notes.append(note)
            
    pm.instruments.append(instrument)
    return pm

def save_midi(pm, path):
    pm.write(str(path))
