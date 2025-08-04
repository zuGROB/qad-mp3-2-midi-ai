import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

CONFIG = {
    "PREPROCESSED_DIR": "preprocessed_data_1",
    "MODEL_SAVE_PATH": "audio_to_midi_v0.11.pth",
    "SEQ_LENGTH": 64,
    "INSTRUMENT_CLASSES": [ 'Drums', 'Acoustic Piano', 'Electric Piano', 'Organ', 'Acoustic Guitar', 'Clean Electric Guitar', 'Distortion Guitar', 'Acoustic Bass', 'Synth Bass', 'Strings Ensemble', 'Solo Strings', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Ethnic' ],
    "N_PITCHES": 128, "N_MELS": 256,
    "BATCH_SIZE": 32,
    "EPOCHS": 75,
    "LEARNING_RATE": 0.0001,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "GRAD_CLIP_VALUE": 1.0,
    "POS_WEIGHT": 50.0,
    "WARMUP_STEPS": 25000,
    "NUM_WORKERS": 4
}
CONFIG["N_CLASSES"] = len(CONFIG["INSTRUMENT_CLASSES"])
CONFIG["OUTPUT_NEURONS"] = CONFIG["N_CLASSES"] * CONFIG["N_PITCHES"]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if identity.shape[1] == out.shape[1]:
            out += identity
        return self.relu(out)

class ResCRNN_v11(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 64),
            nn.MaxPool2d((1, 2)),
            ConvBlock(64, 128),
            nn.MaxPool2d((1, 2)),
            ConvBlock(128, 256),
            nn.MaxPool2d((1, 2)),
            ConvBlock(256, 512),
            nn.MaxPool2d((1, 2)),
        )
        cnn_out_features = 512 * (config["N_MELS"] // 16)
        self.rnn = nn.GRU(cnn_out_features, 512, 3, batch_first=True, bidirectional=True, dropout=0.5)
        base_features_dim = 512 * 2
        self.fc_notes = nn.Linear(base_features_dim, config["OUTPUT_NEURONS"])

    def forward(self, x):
        b, s, _ = x.shape
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1, 3).reshape(b, s, -1)
        base_features, _ = self.rnn(x)
        notes_logits = self.fc_notes(base_features)
        notes_out = notes_logits.view(b, s, self.config["N_CLASSES"], self.config["N_PITCHES"])
        return notes_out

class MidiDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.files = sorted(glob.glob(os.path.join(config["PREPROCESSED_DIR"], "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No preprocessed files found in {config['PREPROCESSED_DIR']}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = torch.load(file_path, map_location='cpu')
        return data['spectrogram'].T, data['notes']

def collate_fn(batch):
    specs, notes = zip(*batch)
    specs = torch.stack(specs)
    return specs, notes

def train():
    if CONFIG['DEVICE'] != 'cuda':
        print("CRITICAL: This training script is optimized for CUDA. Oopsies, sorry not sorry. im fucking dumb for any other")
        return

    try:
        dataset = MidiDataset(CONFIG)
        print(f"Dataset found. Total chunks: {len(dataset)}")
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        return
        
    data_loader = DataLoader(
        dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=CONFIG['NUM_WORKERS'], 
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    model = ResCRNN_v11(CONFIG).to(CONFIG['DEVICE'])
    
    pos_weight_tensor = torch.tensor([CONFIG['POS_WEIGHT']], device=CONFIG['DEVICE'])
    criterion_bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    scaler = torch.cuda.amp.GradScaler()
    
    def lr_lambda(current_step: int):
        if current_step < CONFIG["WARMUP_STEPS"]:
            return float(current_step) / float(max(1, CONFIG["WARMUP_STEPS"]))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Starting training on {len(dataset)} samples...")
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}", leave=True)
        
        for specs, notes_list in progress_bar:
            specs = specs.to(CONFIG['DEVICE'], non_blocking=True)
            
            # Build notes_tensor on the fly on the GPU
            notes_true = torch.zeros(specs.shape[0], CONFIG["SEQ_LENGTH"], CONFIG["N_CLASSES"], CONFIG["N_PITCHES"], dtype=torch.float32, device=CONFIG['DEVICE'])
            for i in range(specs.shape[0]):
                for class_idx, pitch, start, end in notes_list[i]:
                    notes_true[i, start:end, class_idx, pitch] = 1.0

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                notes_logits_pred = model(specs)
                loss_notes = criterion_bce_with_logits(notes_logits_pred, notes_true)
                
            if torch.isnan(loss_notes) or torch.isinf(loss_notes):
                print("ERROR: Loss is NaN or Inf. Stopping training. Did you overclocked your GPU?")
                progress_bar.close()
                return

            scaler.scale(loss_notes).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_VALUE'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss_notes.item()
            
            # update progress bar
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{loss_notes.item():.4f}", avg_loss=f"{avg_loss:.4f}")

        torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])
    
    print(f"Training finished. Final model saved to {CONFIG['MODEL_SAVE_PATH']}. Yippee!!")

if __name__ == '__main__':
    train()