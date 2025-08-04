import os
import glob
import numpy as np
import torch
import librosa
import pretty_midi
from tqdm import tqdm
import warnings

CONFIG = {
    "INPUT_DIR": "input",
    "OUTPUT_DIR": "output",
    "PREPROCESSED_DIR": "preprocessed_data_1",
    "SAMPLE_RATE": 33075,
    "N_FFT": 4096,
    "HOP_LENGTH": 128,
    "N_MELS": 256,
    "SEQ_LENGTH": 64,
    "INSTRUMENT_CLASSES": [
        'Drums', 'Acoustic Piano', 'Electric Piano', 'Organ', 'Acoustic Guitar',
        'Clean Electric Guitar', 'Distortion Guitar', 'Acoustic Bass', 'Synth Bass',
        'Strings Ensemble', 'Solo Strings', 'Brass', 'Reed', 'Pipe',
        'Synth Lead', 'Synth Pad', 'Ethnic'
    ],
    "N_PITCHES": 128,
    "AUGMENT": True,            # AUGMENTATION TOGGLE! if your dataset is small like mine was, your model will suck balls anyways.
    "STRETCH_FACTORS": [0.95, 1.0, 1.05],
    "PITCH_SHIFTS": [-1, 0, 1],
}
CONFIG["CLASS_MAP"] = {name: i for i, name in enumerate(CONFIG["INSTRUMENT_CLASSES"])}
CONFIG["N_CLASSES"] = len(CONFIG["INSTRUMENT_CLASSES"])

def get_instrument_class_v5(instrument: pretty_midi.Instrument, config):
    if instrument.is_drum or 'drum' in instrument.name.lower(): return config["CLASS_MAP"].get('Drums', -1)
    p = instrument.program
    if 0 <= p <= 7: return config["CLASS_MAP"].get('Acoustic Piano', -1)
    if 8 <= p <= 15: return config["CLASS_MAP"].get('Electric Piano', -1)
    if 16 <= p <= 23: return config["CLASS_MAP"].get('Organ', -1)
    if 24 <= p <= 25: return config["CLASS_MAP"].get('Acoustic Guitar', -1)
    if 26 <= p <= 29: return config["CLASS_MAP"].get('Clean Electric Guitar', -1)
    if 30 <= p <= 31: return config["CLASS_MAP"].get('Distortion Guitar', -1)
    if 32 <= p <= 35: return config["CLASS_MAP"].get('Acoustic Bass', -1)
    if 36 <= p <= 39: return config["CLASS_MAP"].get('Synth Bass', -1)
    if 40 <= p <= 47: return config["CLASS_MAP"].get('Solo Strings', -1)
    if 48 <= p <= 55: return config["CLASS_MAP"].get('Strings Ensemble', -1)
    if 56 <= p <= 63: return config["CLASS_MAP"].get('Brass', -1)
    if 64 <= p <= 71: return config["CLASS_MAP"].get('Reed', -1)
    if 72 <= p <= 79: return config["CLASS_MAP"].get('Pipe', -1)
    if 80 <= p <= 103: return config["CLASS_MAP"].get('Synth Pad', -1)
    if 104 <= p <= 111: return config["CLASS_MAP"].get('Ethnic', -1)
    return -1

def process_and_save_chunks(mp3_path, midi_path, config, stretch_factor, pitch_shift):
    chunk_count = 0
    basename = os.path.splitext(os.path.basename(mp3_path))[0]
    try:
        y, sr = librosa.load(mp3_path, sr=config["SAMPLE_RATE"])
        if stretch_factor != 1.0:
            y = librosa.effects.time_stretch(y, rate=stretch_factor)
        if pitch_shift != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=config["N_FFT"], hop_length=config["HOP_LENGTH"], n_mels=config["N_MELS"])
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        pm = pretty_midi.PrettyMIDI(midi_path)
        if pitch_shift != 0:
            for instrument in pm.instruments:
                for note in instrument.notes:
                    note.pitch += pitch_shift
        if stretch_factor != 1.0:
            for instrument in pm.instruments:
                for note in instrument.notes:
                    note.start /= stretch_factor
                    note.end /= stretch_factor

        fs = config["SAMPLE_RATE"] / config["HOP_LENGTH"]
        seq_len = config["SEQ_LENGTH"]

        for start_frame in range(0, log_mel_spec.shape[1] - seq_len + 1, seq_len):
            end_frame = start_frame + seq_len
            start_time_sec = start_frame / fs
            end_time_sec = end_frame / fs

            spec_chunk = torch.from_numpy(log_mel_spec[:, start_frame:end_frame]).to(torch.float16)
            notes_list = []

            for instrument in pm.instruments:
                class_idx = get_instrument_class_v5(instrument, config)
                if class_idx == -1: continue
                instrument.remove_invalid_notes()
                for note in instrument.notes:
                    if note.pitch >= config["N_PITCHES"]: continue
                    if note.start < end_time_sec and note.end > start_time_sec:
                        note_start_in_chunk = max(0, int(round((note.start - start_time_sec) * fs)))
                        note_end_in_chunk = min(seq_len, int(round((note.end - start_time_sec) * fs)))
                        if note_end_in_chunk > note_start_in_chunk:
                            notes_list.append((class_idx, note.pitch, note_start_in_chunk, note_end_in_chunk))

            save_name = f"{basename}_s{int(stretch_factor*100)}_p{pitch_shift}_c{chunk_count}.pt"
            save_path = os.path.join(config["PREPROCESSED_DIR"], save_name)
            torch.save({'spectrogram': spec_chunk, 'notes': notes_list}, save_path)
            chunk_count += 1
        return chunk_count
    except Exception as e:
        print(f"Error processing {mp3_path} (s:{stretch_factor}, p:{pitch_shift}): {e}")
        return 0

def main():
    warnings.filterwarnings('ignore')
    print("Starting preprocessing...")
    os.makedirs(CONFIG["PREPROCESSED_DIR"], exist_ok=True)

    tasks = []
    mp3_files = glob.glob(os.path.join(CONFIG["INPUT_DIR"], "*.mp3"))
    for mp3_path in mp3_files:
        basename = os.path.splitext(os.path.basename(mp3_path))[0]
        midi_path = None
        for ext in ['.mid', '.midi']:
            potential_path = os.path.join(CONFIG["OUTPUT_DIR"], basename + ext)
            if os.path.exists(potential_path):
                midi_path = potential_path
                break
        if not midi_path:
            continue

        if CONFIG["AUGMENT"]:
            for factor in CONFIG["STRETCH_FACTORS"]:
                for shift in CONFIG["PITCH_SHIFTS"]:
                    tasks.append((mp3_path, midi_path, CONFIG, factor, shift))
        else:
            tasks.append((mp3_path, midi_path, CONFIG, 1.0, 0))

    total_chunks = 0
    for task in tqdm(tasks, desc="Creating augmented chunks"):
        total_chunks += process_and_save_chunks(*task)

    print(f"Preprocessing finished. Saved {total_chunks} chunks to {CONFIG['PREPROCESSED_DIR']}")

if __name__ == '__main__':
    main()
