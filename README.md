# whispr

Real-time microphone transcription using [Whisper](https://github.com/openai/whisper) via [MLX](https://github.com/ml-explore/mlx), optimized for Apple Silicon.

## Installation

It is recommended to use a virtual environment to avoid conflicts with other Python packages.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# When you're done, deactivate with:
deactivate
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Audio buffer manipulation |
| `sounddevice` | Microphone capture via PortAudio |
| `mlx-whisper` | Whisper inference accelerated with Apple MLX |
| `argostranslate` | Offline translation of the transcription (optional, only when using `--target-lang`) |

Install all dependencies with:

```bash
pip install numpy sounddevice mlx-whisper argostranslate
```

> **Note:** `sounddevice` requires PortAudio. If you get an error, install it first:
> ```bash
> brew install portaudio
> ```

### Model

On first run, the model `mlx-community/whisper-medium-mlx` will be downloaded automatically from Hugging Face (~1.5 GB). No manual setup required.

## Usage

```bash
python transcribe.py
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--mic` | `-m` | system default | Audio input device ID |
| `--chunk` | `-c` | `5` | Capture chunk duration in seconds |
| `--output` | `-o` | — | `.txt` file to save the transcription |
| `--silence-threshold` | `-s` | `0.01` | RMS silence threshold (chunks below this are skipped) |
| `--source-lang` | `-f` | `pt` | Spoken/captured language (ISO code, e.g. `pt`, `en`, `es`) |
| `--target-lang` | `-t` | — | Translate the transcription into this language (offline). Omit for no translation |
| `--list-devices` | | — | List available audio devices and exit |
| `--debug` | | — | Show RMS level of each captured chunk |

### Examples

```bash
# Basic usage
python transcribe.py

# List available audio devices
python transcribe.py --list-devices

# Debug mode (shows RMS level per chunk)
python transcribe.py --debug
```

```bash
python transcribe.py -m 1 -c 3 -s 0.002
```

### Simultaneous translation

Transcribe in one language and display the text in another, fully offline:

```bash
# Listen in English, read the transcription in Portuguese
python transcribe.py -m 1 -f en -t pt
```

> **Note:** On the first run for a given language pair, argos-translate downloads
> the offline translation model (needs internet once). After that it works
> offline. When no direct model exists for the pair, it pivots through English
> automatically. Without `--target-lang` the tool only transcribes (no translation).
