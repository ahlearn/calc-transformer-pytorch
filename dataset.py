import torch
from torch.utils.data import IterableDataset
import random

VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '=', '<pad>']

char_to_ix = {ch: i for i, ch in enumerate(VOCAB)}
ix_to_char = {i: ch for i, ch in enumerate(VOCAB)}

def encode(s):
    return [char_to_ix[c] for c in s]

def decode(l):
    return ''.join([ix_to_char[i] for i in l])

class CalculatorDataset(IterableDataset):
    def __init__(self, config):
        self.max_digits = config.max_digits
        self.operators = config.operators
        self.batch_size = config.batch_size
        
        # Max context length calculation
        # e.g. 10 digits + 1 op + 10 digits + 1 op + 10 digits + 1 '=' + up to ~11 digits answer.
        # Let's say we do 2 to 3 operands.
        # A 3-operand formula: 10 + 1 + 10 + 1 + 10 + 1 = 33 chars left side.
        # The answer can be up to 11 digits + potential negative sign -> 12 chars.
        # Total max length roughly 45 chars.
        # We'll set a fixed sequence length to make batching easier.
        self.seq_len = 64
        
    def generate_formula(self):
        # Generate exactly 2 terms for simple addition and subtraction
        num_terms = 2
        terms = []
        for _ in range(num_terms):
            # random length 1 to max_digits
            num_digits = random.randint(1, self.max_digits)
            # first digit 1-9, rest 0-9
            if num_digits == 1:
                val = random.randint(0, 9)
            else:
                first = random.randint(1, 9)
                rest = [random.randint(0, 9) for _ in range(num_digits - 1)]
                val = int(str(first) + ''.join(map(str, rest)))
            terms.append(val)
            
        ops = [random.choice(self.operators) for _ in range(num_terms - 1)]
        
        formula_str = str(terms[0])
        total = int(terms[0])
        for i in range(num_terms - 1):
            formula_str += ops[i] + str(terms[i+1])
            val = int(terms[i+1])
            if ops[i] == '+':
                total += val
            elif ops[i] == '-':
                total -= val
                
        formula_str += "=" + str(total)
        return formula_str
        
    def __iter__(self):
        pad_idx = char_to_ix['<pad>']
        while True:
            s = self.generate_formula()
            encoded = encode(s)
            
            # Pad to seq_len + 1 (for input and target shift)
            if len(encoded) > self.seq_len + 1:
                encoded = encoded[:self.seq_len + 1]
            
            # If shorter, pad it
            encoded = encoded + [pad_idx] * (self.seq_len + 1 - len(encoded))
            
            encoded_t = torch.tensor(encoded, dtype=torch.long)
            x = encoded_t[:-1]
            y = encoded_t[1:]
            
            yield x, y
