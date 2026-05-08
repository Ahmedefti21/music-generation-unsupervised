import pretty_midi
from pathlib import Path

def get_midi_metadata(midi_path):
    """Extract basic metadata from a MIDI file."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        return {
            "duration": pm.get_end_time(),
            "instruments": [instr.name for instr in pm.instruments],
            "is_polyphonic": len(pm.instruments) > 0,
            "tempo": pm.estimate_tempo()
        }
    except Exception as e:
        return {"error": str(e)}

def validate_midi(midi_path):
    """Check if the MIDI file is valid and has at least one note."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in pm.instruments:
            if len(instrument.notes) > 0:
                return True
        return False
    except:
        return False

if __name__ == "__main__":
    # Test on a generated file
    test_file = Path("outputs/generated_midis/task3_transformer/task3_generated_1.mid")
    if test_file.exists():
        print(get_midi_metadata(test_file))
