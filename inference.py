import torch
import argparse
import time
from config import Config
from model import CalculatorTransformer
from dataset import VOCAB, char_to_ix, ix_to_char, encode, decode

def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return config.device

def generate(model, prompt, device, max_len=64, debug=False):
    model.eval()
    
    # encode prompt
    input_ids = encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    pad_idx = char_to_ix['<pad>']
    
    if debug:
        print(f"--- Debug Mode ---")
        print(f"Prompt: {prompt}")
        print(f"Initial tokens: {input_ids}")
        print("-" * 20)
    
    generated_ids = list(input_ids)
    
    with torch.no_grad():
        for step in range(max_len - len(input_ids)):
            logits = model(x)
            # Take the prediction for the last token in the sequence
            next_token_logits = logits[0, -1, :]
            next_token_id = next_token_logits.argmax().item()
            
            if debug:
                probs = torch.softmax(next_token_logits, dim=0)
                top_prob, top_idx = probs.topk(3)
                print(f"Step {step+1}:")
                print(f"  Current sequence: '{decode(generated_ids)}'")
                print(f"  Top 3 predictions:")
                for i in range(3):
                    token_char = ix_to_char[top_idx[i].item()]
                    print(f"    '{token_char}': {top_prob[i].item()*100:.2f}%")
                print(f"  Chosen token: '{ix_to_char[next_token_id]}'")
                print("-" * 20)
                time.sleep(0.5) # slow down for visual debugging
            
            if next_token_id == pad_idx:
                if debug:
                    print("Reached <pad> token. Stopping.")
                break
                
            generated_ids.append(next_token_id)
            x = torch.tensor([generated_ids], dtype=torch.long).to(device)
            
    return decode(generated_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculator Transformer Inference")
    parser.add_argument("formula", type=str, help="Input formula, e.g., '123 + 456 ='")
    parser.add_argument("--debug", action="store_true", help="Print step-by-step decoding process")
    parser.add_argument("--model", type=str, default="model.pt", help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    # Remove spaces from input formula
    formula = args.formula.replace(" ", "")
    if not formula.endswith("="):
        print("Warning: Input formula should probably end with '='")
        
    config = Config()
    device = get_device(config)
    
    model = CalculatorTransformer(len(VOCAB), config).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded model from {args.model}")
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found. Please train the model first.")
        exit(1)
        
    print(f"\nEvaluating: {formula}")
    result = generate(model, formula, device, max_len=64, debug=args.debug)
    print(f"\nFinal Result: {result}")
