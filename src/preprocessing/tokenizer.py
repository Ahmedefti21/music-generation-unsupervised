"""Tokenize MAESTRO MIDI files using miditok REMI for Task 3 Transformer."""
import pandas as pd
import numpy as np
from pathlib import Path
import miditok


def tokenize_maestro(maestro_root, output_dir, seq_len=512):
    """Tokenize all MAESTRO MIDI files and save chunked sequences per split."""
    maestro_root = Path(maestro_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata CSV
    csv_path = maestro_root / "maestro-v3.0.0.csv"
    df = pd.read_csv(csv_path)

    # Initialize REMI tokenizer with default config
    tokenizer = miditok.REMI()

    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split]
        all_chunks = []

        print(f"\nTokenizing {split} split ({len(split_df)} files)...")

        for idx, row in split_df.iterrows():
            midi_path = maestro_root / row["midi_filename"]

            if not midi_path.exists():
                print(f"  Skipping missing file: {midi_path}")
                continue

            try:
                tokens = tokenizer(midi_path)

                if len(tokens) == 0:
                    continue

                ids = tokens[0].ids

                # Chunk into fixed-length windows
                for start in range(0, len(ids) - seq_len + 1, seq_len):
                    chunk = ids[start:start + seq_len]
                    all_chunks.append(chunk)

            except Exception as e:
                print(f"  Error tokenizing {midi_path.name}: {e}")
                continue

        all_chunks = np.array(all_chunks, dtype=np.int64)

        # Map split name
        save_name = split if split != "validation" else "val"
        save_path = output_dir / f"tokens_{save_name}.npy"
        np.save(save_path, all_chunks)

        print(f"  Saved {save_name}: {all_chunks.shape} to {save_path}")

    # Save tokenizer for later use in generation
    tokenizer.save(str(output_dir / "remi_tokenizer.json"))
    print(f"\nTokenizer saved. Vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    tokenize_maestro(
        maestro_root="data/raw_midi/maestro-v3.0.0",
        output_dir="data/tokens",
        seq_len=512
    )
