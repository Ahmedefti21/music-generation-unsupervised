import pandas as pd
import matplotlib.pyplot as plt
import pretty_midi
import numpy as np
import glob
import os
from pathlib import Path

def run_eda():
    output_dir = Path('../../outputs/plots/eda')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = '../../data/raw_midi/maestro-v3.0.0/maestro-v3.0.0.csv'
    df = pd.read_csv(csv_path)
    
    # 1. Piece duration histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df['duration'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Piece Durations in MAESTRO Dataset')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(output_dir / 'duration_histogram.png')
    plt.close()
    
    # Analyze a subset of MIDI files for note-level stats
    midi_files = glob.glob('../../data/raw_midi/maestro-v3.0.0/*/*.midi') + glob.glob('../../data/raw_midi/maestro-v3.0.0/*/*.mid')
    # Use 50 files for speed
    sample_files = midi_files[:50]
    
    note_counts = []
    pitches = []
    velocities = []
    sparsities = []
    
    for f in sample_files:
        try:
            pm = pretty_midi.PrettyMIDI(f)
            notes = pm.instruments[0].notes
            note_counts.append(len(notes))
            
            for note in notes:
                pitches.append(note.pitch)
                velocities.append(note.velocity)
                
            # Sparsity
            pr = pm.get_piano_roll(fs=16)
            pr = pr[21:109, :] # 88 keys
            total_cells = pr.size
            active_cells = np.count_nonzero(pr)
            sparsity = 1.0 - (active_cells / total_cells)
            sparsities.append(sparsity)
        except Exception as e:
            print(f"Failed to process {f}: {e}")
            
    # 2. Note count per piece
    plt.figure(figsize=(10, 5))
    plt.hist(note_counts, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Total Notes per Recording (Sample)')
    plt.xlabel('Number of Notes')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(output_dir / 'note_count_histogram.png')
    plt.close()
    
    # 3. Pitch distribution
    plt.figure(figsize=(12, 5))
    plt.hist(pitches, bins=88, range=(21, 108), color='coral', edgecolor='black')
    plt.title('Pitch Frequency Distribution (Sample)')
    plt.xlabel('MIDI Pitch (21=A0, 108=C8)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(output_dir / 'pitch_distribution.png')
    plt.close()
    
    # 4. Velocity distribution
    plt.figure(figsize=(10, 5))
    plt.hist(velocities, bins=30, range=(0, 127), color='plum', edgecolor='black')
    plt.title('Velocity Distribution (Sample)')
    plt.xlabel('Velocity (0-127)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(output_dir / 'velocity_distribution.png')
    plt.close()
    
    print(f"Average Piano-Roll Sparsity (16fps): {np.mean(sparsities):.4f}")
    print("EDA plots generated in outputs/plots/eda/")

if __name__ == '__main__':
    run_eda()
