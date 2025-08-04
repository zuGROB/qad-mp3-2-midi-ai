import os
import numpy as np
import torch
import torch.nn as nn
import pretty_midi
import librosa
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


CONFIG = {
    "SAMPLE_RATE": 33075, "N_FFT": 4096, "HOP_LENGTH": 128, "N_MELS": 256,
    "SEQ_LENGTH": 64, "INFERENCE_BATCH_SIZE": 16,
    "INSTRUMENT_CLASSES": [
        'Drums', 'Acoustic Piano', 'Electric Piano', 'Organ', 'Acoustic Guitar',
        'Clean Electric Guitar', 'Distortion Guitar', 'Acoustic Bass', 'Synth Bass',
        'Strings Ensemble', 'Solo Strings', 'Brass', 'Reed', 'Pipe',
        'Synth Lead', 'Synth Pad', 'Ethnic'
    ],
    "N_PITCHES": 128, "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}
CONFIG["N_CLASSES"] = len(CONFIG["INSTRUMENT_CLASSES"])
CONFIG["OUTPUT_NEURONS"] = CONFIG["N_CLASSES"] * CONFIG["N_PITCHES"]

DEFAULT_PROGRAMS = {
    'Acoustic Piano': 0, 'Electric Piano': 4, 'Organ': 19, 'Acoustic Guitar': 25, 
    'Clean Electric Guitar': 27, 'Distortion Guitar': 30, 'Acoustic Bass': 32, 
    'Synth Bass': 38, 'Strings Ensemble': 48, 'Solo Strings': 40, 'Brass': 56, 
    'Reed': 64, 'Pipe': 73, 'Synth Lead': 80, 'Synth Pad': 88, 'Ethnic': 104, 'Drums': 0
}
# some ResCRNN shit
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
        notes_pred = torch.sigmoid(notes_logits)
        notes_out = notes_pred.view(b, s, self.config["N_CLASSES"], self.config["N_PITCHES"])
        return notes_out

def prediction_to_midi(pred_notes, fs, config, note_threshold=0.5, min_note_duration=0.01, min_track_notes=1):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for class_idx, class_name in enumerate(config["INSTRUMENT_CLASSES"]):
        instrument = pretty_midi.Instrument(program=DEFAULT_PROGRAMS.get(class_name, 0), is_drum=(class_name == 'Drums'), name=class_name)
        notes_roll = (pred_notes[:, class_idx, :] > note_threshold).astype(int)
        
        for pitch in range(config["N_PITCHES"]):
            padded = np.pad(notes_roll[:, pitch], (1, 1), 'constant'); changes = np.diff(padded)
            note_ons = np.where(changes == 1)[0]; note_offs = np.where(changes == -1)[0]
            
            for i in range(len(note_ons)):
                start_frame, end_frame = note_ons[i], note_offs[i] if i < len(note_offs) else notes_roll.shape[0]
                start_time, end_time = start_frame / fs, end_frame / fs
                duration = end_time - start_time
                if duration < min_note_duration: continue

                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
                instrument.notes.append(note)
        
        if len(instrument.notes) >= min_track_notes:
            pm.instruments.append(instrument)

    return pm

def transcribe_high_quality(audio_path, model_path, output_path, args):     # good lord.
    print(f"Loading model {model_path} on {CONFIG['DEVICE']}..."); model = ResCRNN_v11(CONFIG).to(CONFIG['DEVICE']); model.load_state_dict(torch.load(model_path, map_location=CONFIG['DEVICE'])); model.eval()
    
    if CONFIG['DEVICE'] == 'cuda' and torch.__version__ >= "2.0.0":
        try: model = torch.compile(model, backend="cudagraphs"); print("Model compiled!")
        except Exception as e: print(f"Failed to complile: {e}")
        
    print(f"Processing audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=CONFIG['SAMPLE_RATE'])
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=CONFIG['N_FFT'], hop_length=CONFIG['HOP_LENGTH'], n_mels=CONFIG['N_MELS'])
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    full_spec_tensor = torch.from_numpy(log_mel_spec.T).float()
    total_frames = full_spec_tensor.shape[0]
    final_notes = torch.zeros(total_frames, CONFIG['N_CLASSES'], CONFIG['N_PITCHES'])
    frame_counts = torch.zeros(total_frames)
    step = CONFIG['SEQ_LENGTH'] // 2; batches = []
    for start_frame in range(0, total_frames, step):
        if start_frame + CONFIG['SEQ_LENGTH'] > total_frames: continue
        batches.append(full_spec_tensor[start_frame : start_frame + CONFIG['SEQ_LENGTH']])
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for i in tqdm(range(0, len(batches), CONFIG['INFERENCE_BATCH_SIZE']), desc="Processing..."):
            batch_chunks = batches[i : i + CONFIG['INFERENCE_BATCH_SIZE']]; batch_tensor = torch.stack(batch_chunks).to(CONFIG['DEVICE'])
            notes_pred = model(batch_tensor)
            notes_pred = notes_pred.cpu().to(torch.float32)
            for j, note_chunk in enumerate(notes_pred):
                current_start_frame = (i + j) * step; current_end_frame = current_start_frame + CONFIG['SEQ_LENGTH']
                final_notes[current_start_frame:current_end_frame] += note_chunk; frame_counts[current_start_frame:current_end_frame] += 1
    frame_counts[frame_counts == 0] = 1; final_notes /= frame_counts.unsqueeze(-1).unsqueeze(-1)
    print("Cleaning up and creating a file...")
    fs = CONFIG['SAMPLE_RATE'] / CONFIG['HOP_LENGTH']
    midi_data = prediction_to_midi(final_notes.numpy(), fs, CONFIG, note_threshold=args.threshold, min_note_duration=args.min_duration, min_track_notes=args.min_notes)
    if midi_data:
        midi_data.write(output_path); print(f"Done!! Saved in {output_path}")
    else:
        print("Fatal error during MIDI creation. Session terminated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio to midi 'AI' thingy. Kinda.")
    parser.add_argument("audio_file", type=str, help="MP3 path")
    parser.add_argument("--model", type=str, default="audio_to_midi_v0.11.pth", help="Model path")
    parser.add_argument("--output", type=str, default="output_v0.11.mid", help="Output MIDI file path")
    parser.add_argument("--threshold", type=float, default=0.9, help="Activation threshold")
    parser.add_argument("--min-duration", type=float, default=0.1, help="Minimal note duartion")
    parser.add_argument("--min-notes", type=int, default=1, help="Minimal amount of notes in a track")
    args = parser.parse_args()
    if not os.path.exists(args.audio_file): print(f"Audio file not found: {args.audio_file}")
    elif not os.path.exists(args.model): print(f"Weights file not found: {args.model}.")
    else: transcribe_high_quality(args.audio_file, args.model, args.output, args)
