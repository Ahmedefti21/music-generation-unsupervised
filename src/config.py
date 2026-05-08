from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi" / "maestro-v3.0.0"
TRAIN_TEST_SPLIT_DIR = DATA_DIR / "train_test_split"
TOKENS_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
GENERATED_MIDIS_DIR = OUTPUT_DIR / "generated_midis"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Hyperparameters
FS = 16  # Piano roll sampling frequency (frames per second)
PIANO_ROLL_WINDOW_SIZE = 128
TRANSFORMER_MAX_SEQ_LEN = 512

# Metrics config
EVAL_NUM_NOTES = 80
