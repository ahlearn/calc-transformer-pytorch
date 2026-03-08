import yaml

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        self.d_model = cfg['model']['d_model']
        self.nhead = cfg['model']['nhead']
        self.num_layers = cfg['model']['num_layers']
        self.dropout = cfg['model']['dropout']
        
        self.max_digits = cfg['data']['max_digits']
        self.operators = cfg['data']['operators']
        self.batch_size = cfg['data']['batch_size']
        
        self.lr = cfg['training']['lr']
        self.max_steps = cfg['training']['max_steps']
        self.log_interval = cfg['training']['log_interval']
        self.save_interval = cfg['training']['save_interval']
        self.device = cfg['training']['device']
        
    def __str__(self):
        return str(vars(self))
