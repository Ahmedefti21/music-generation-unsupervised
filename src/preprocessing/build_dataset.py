import pandas as pd
import numpy as np
from pathlib import Path
from .piano_roll import midi_to_binary_piano_roll, segment_piano_roll

def build_dataset(csv_path, midi_root, out_dir, max_train=200, max_test=50):
    df = pd.read_csv(csv_path)
    
    train_files = df[df['split'] == 'train']['midi_filename'].tolist()
    val_files = df[df['split'] == 'validation']['midi_filename'].tolist()
    test_files = df[df['split'] == 'test']['midi_filename'].tolist()
    
    if max_train:
        train_files = train_files[:max_train]
    if max_test:
        val_files = val_files[:max_test]
        test_files = test_files[:max_test]
        
    midi_root = Path(midi_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def process_split(files, name):
        print(f"Processing {name} split ({len(files)} files)...")
        dataset_windows = []
        for i, f in enumerate(files):
            try:
                path = midi_root / f
                piano_roll = midi_to_binary_piano_roll(path, fs=16)
                windows = segment_piano_roll(piano_roll, window_size=128, stride=64)
                dataset_windows.extend(windows)
            except Exception as e:
                print(f"Failed to process {f}: {e}")
            if (i+1) % 50 == 0:
                print(f"  Processed {i+1} files")
                
        dataset = np.array(dataset_windows, dtype=np.float32)
        out_path = out_dir / f"{name}.npy"
        np.save(out_path, dataset)
        print(f"Saved {name} dataset to {out_path} with shape {dataset.shape}")
        
    process_split(train_files, 'train')
    process_split(val_files, 'val')
    process_split(test_files, 'test')

if __name__ == '__main__':
    csv_path = '../../data/raw_midi/maestro-v3.0.0/maestro-v3.0.0.csv'
    midi_root = '../../data/raw_midi/maestro-v3.0.0/'
    out_dir = '../../data/train_test_split/'
    # Using larger subset to give the model better data, but keeping it small enough to run quickly
    build_dataset(csv_path, midi_root, out_dir, max_train=200, max_test=50)
