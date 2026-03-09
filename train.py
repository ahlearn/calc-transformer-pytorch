import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from config import Config
from dataset import CalculatorDataset, VOCAB
from model import CalculatorTransformer

def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return config.device

def train():
    config = Config()
    device = get_device(config)
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = CalculatorDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    # Model
    model = CalculatorTransformer(len(VOCAB), config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB.index('<pad>'))

    # Logging
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter('runs/calculator')

    # Training Loop
    model.train()
    step = 0
    running_loss = 0.0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        
        # logits: (batch_size, seq_len, vocab_size)
        # y: (batch_size, seq_len)
        loss = criterion(logits.view(-1, len(VOCAB)), y.view(-1))
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            print(f"Step {step}/{config.max_steps} - Loss: {avg_loss:.4f}")
            writer.add_scalar('Loss/train', avg_loss, step)
            running_loss = 0.0
            
            # Simple accuracy tracking (greedy match)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (y != VOCAB.index('<pad>'))
                correct = (preds == y)[mask].sum().item()
                total = mask.sum().item()
                accuracy = correct / total if total > 0 else 0
                writer.add_scalar('Accuracy/train', accuracy, step)

        if step % config.save_interval == 0:
            torch.save(model.state_dict(), 'model.pt')
            print(f"Model saved at step {step}")
            
            # Run evaluation for early stopping
            from eval import eval_model
            acc = eval_model(model, device, num_samples=200)
            writer.add_scalar('Accuracy/eval', acc, step)
            if acc == 100.0:
                print(f"\nReached 100.0% Validation Accuracy at step {step}! Mission Accomplished.")
                break

        if step >= config.max_steps:
            print("Training complete without reaching 100%.")
            break

    torch.save(model.state_dict(), 'model.pt')
    writer.close()

if __name__ == '__main__':
    train()
