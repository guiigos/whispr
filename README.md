# whispr

Real-time microphone transcription using [Whisper](https://github.com/openai/whisper) via [MLX](https://github.com/ml-explore/mlx), optimized for Apple Silicon.

## Installation

It is recommended to use a virtual environment to avoid conflicts with other Python packages.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# When you're done, deactivate with:
deactivate
```

> All subsequent commands (`pip install`, `python transcribe.py`) should be run with the virtual environment activated.

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Audio buffer manipulation |
| `sounddevice` | Microphone capture via PortAudio |
| `mlx-whisper` | Whisper inference accelerated with Apple MLX |

Install all dependencies with:

```bash
pip install numpy sounddevice mlx-whisper
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
| `--list-devices` | | — | List available audio devices and exit |
| `--debug` | | — | Show RMS level of each captured chunk |

### Examples

```bash
# Basic usage
python transcribe.py

# Save transcription to a file
python transcribe.py -o output.txt

# Use a specific microphone and 3-second chunks
python transcribe.py -m 2 -c 3

# List available audio devices
python transcribe.py --list-devices

# Debug mode (shows RMS level per chunk)
python transcribe.py --debug
```
