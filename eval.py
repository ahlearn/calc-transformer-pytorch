import torch
import random
from config import Config
from model import CalculatorTransformer
from dataset import VOCAB
from inference import generate
from tqdm import tqdm

def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return config.device

def eval_model(model, device, num_samples=1000):
    correct = 0
    
    print(f"Evaluating model on {num_samples} samples of up to 3-digit math...")
    
    for _ in tqdm(range(num_samples)):
        # Generate a test case: A + B or A - B
        op = random.choice(['+', '-'])
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        
        prompt = f"{a}{op}{b}="
        if op == '+':
            expected_answer = str(a + b)[::-1]
        else:
            expected_answer = str(a - b)[::-1]
            
        # Default generation gives us the full sequence e.g., "123+456=579"
        output = generate(model, prompt, device, max_len=32, debug=False)
        
        expected_full = prompt + expected_answer
        
        # the generation stops either at max_len or when it generates <pad>
        if output == expected_full:
            correct += 1
        elif expected_full in output:
            # handle case where it outputs trailing pads or something
            correct += 1
            
    accuracy = correct / num_samples * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{num_samples})")
    return accuracy

if __name__ == '__main__':
    config = Config()
    device = get_device(config)
        
    model = CalculatorTransformer(len(VOCAB), config).to(device)
    
    try:
        model.load_state_dict(torch.load("model.pt", map_location=device))
        print(f"Loaded model.pt for evaluation")
    except Exception as e:
        print(f"Could not load model: {e}")
        exit(1)
        
    eval_model(model, device)
