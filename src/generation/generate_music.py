import sys
import torch
from pathlib import Path
import miditok
from miditok import TokSequence

# Add root to sys.path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root / "src"))

from models.transformer import MusicTransformer

def generate(checkpoint_path, out_path, num_tokens=500, temperature=1.0, top_k=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MusicTransformer(vocab_size=284).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    tokenizer = miditok.REMI()
    tokens = [4] # Bar token as seed
    
    print(f"Generating {num_tokens} tokens...")
    with torch.no_grad():
        for _ in range(num_tokens):
            context = torch.tensor([tokens[-511:]], dtype=torch.long, device=device)
            logits = model(context)
            next_logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[-1]
                next_logits = next_logits.masked_fill(next_logits < min_val, float("-inf"))
            
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

    # Save MIDI
    tok_seq = TokSequence(ids=tokens)
    tokenizer.complete_sequence(tok_seq)
    midi_obj = tokenizer([tok_seq])
    midi_obj.dump_midi(out_path)
    print(f"Saved generated MIDI to {out_path}")

if __name__ == "__main__":
    checkpoint = root / "outputs" / "checkpoints" / "transformer_best.pth"
    output = root / "outputs" / "generated_midis" / "script_generated.mid"
    
    if checkpoint.exists():
        generate(checkpoint, output)
    else:
        print(f"Checkpoint {checkpoint} not found. Please train the model first.")
