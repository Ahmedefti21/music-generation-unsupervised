from pathlib import Path
import numpy as np
import pretty_midi


def midi_to_binary_piano_roll(midi_path, fs=16):
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Merge all non-drum instruments into one piano roll
    all_rolls = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        roll = instrument.get_piano_roll(fs=fs)
        all_rolls.append(roll)

    if not all_rolls:
        return None

    merged_roll = np.maximum.reduce(all_rolls)
    binary_roll = (merged_roll > 0).astype(np.float32)

    return binary_roll


def segment_piano_roll(piano_roll, window_size=128, stride=64):
    windows = []

    if piano_roll is None:
        return windows

    total_steps = piano_roll.shape[1]

    for start in range(0, total_steps - window_size + 1, stride):
        window = piano_roll[:, start:start + window_size]
        windows.append(window)

    return windows


def process_dataset(midi_root, fs=16, window_size=128, stride=64, max_files=None):
    midi_root = Path(midi_root)
    midi_files = list(midi_root.rglob("*.midi")) + list(midi_root.rglob("*.mid"))

    if max_files is not None:
        midi_files = midi_files[:max_files]

    dataset_windows = []

    for i, midi_path in enumerate(midi_files):
        try:
            piano_roll = midi_to_binary_piano_roll(midi_path, fs=fs)
            windows = segment_piano_roll(
                piano_roll,
                window_size=window_size,
                stride=stride
            )
            dataset_windows.extend(windows)

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(midi_files)} files")

        except Exception as e:
            print(f"Skipping {midi_path.name} بسبب error: {e}")

    return np.array(dataset_windows, dtype=np.float32)

def piano_roll_to_midi(piano_roll, output_path, fs=16, program=0, threshold=0.1, min_note_duration=0.05):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Convert to binary note activity
    active = piano_roll > threshold

    num_pitches, num_steps = active.shape

    for pitch in range(num_pitches):
        time = 0

        while time < num_steps:
            if active[pitch, time]:
                start_time = time

                # continue while same pitch is active
                while time < num_steps and active[pitch, time]:
                    time += 1

                end_time = time

                start = start_time / fs
                end = end_time / fs

                if end - start >= min_note_duration:
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=int(pitch),
                        start=float(start),
                        end=float(end)
                    )
                    instrument.notes.append(note)
            else:
                time += 1

    midi.instruments.append(instrument)
    midi.write(str(output_path))

