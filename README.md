# MP3 to MIDI Converter AI [PROOF OF CONCEPT!]

This garbage uses a Convolutional Recurrent Neural Network (CRNN) to transcribe music from MP3 audio files into MIDI format.

Could be good, could be bad. Who knows? I don't have enough data to train it, god damn it!!

Testing model was trained on soundtracks from Touhou 3 and 4 and their MIDIs made by Blargzargo. This is just a PoC!

Model was trained with augmentation enabled on 37 pairs MP3-MIDI. 75 epochs, batch of 32, with warmup. Current settings in train.py, read it yourself.

Final avg loss at 0.0047. Looks stupidly ovberfitted, eh??

Weights: https://huggingface.co/KazamiYuuka/qad-mp3-2-midi-ai

# Test midis:

0415.mid - WAS in dataset for training. Threshold - 0.5, min duration - 0.01.

0518.mid - was NOT in dataset. Threshold - 0.9, min duration - 0.1.

## How it Works

1.  **Preprocessing (`preprocess.py`):** Audio files are converted into mel spectrograms, which are visual representations of the sound. These are saved as data chunks for training. Data augmentation (time stretching, pitch shifting) is used to create a larger, more robust dataset.
2.  **Training (`train.py`):** The `ResCRNN_v0.11` model learns to map the spectrogram chunks to MIDI notes. The trained model is saved as `audio_to_midi_v0.11.pth`.
3.  **Conversion (`convert_v11.py`):** The trained model analyzes a new MP3 file and predicts the MIDI notes, generating a `.mid` file.

## How to Use

1.  **Add Data:**
    *   Place your MP3 files in the `/input` folder.
    *   Place the corresponding ground-truth MIDI files in the `/output` folder.

2.  **Preprocess Data:**
    ```bash
    python preprocess.py
    ```

3.  **Train the Model:**
    ```bash
    python train.py
    ```

4.  **Convert an MP3 to MIDI:**
    ```bash
    python convert_v11.py "path/to/your/song.mp3"
    ```
