import sys
import argparse
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import mlx_whisper

MODEL_REPO = "mlx-community/whisper-medium-mlx"
SAMPLE_RATE = 16000

def list_devices():
  print(sd.query_devices())


def record_chunks(audio_queue: queue.Queue, chunk_seconds: int, device_id: int | None):
  device_info = sd.query_devices(device_id, "input")
  channels = min(int(device_info["max_input_channels"]), 2)

  def callback(indata, frames, time, status):
    if status:
      print(f"[warning] {status}", file=sys.stderr)

    mono = indata.mean(axis=1, keepdims=True)
    try:
      audio_queue.put_nowait(mono)
    except queue.Full:
      pass

  chunk_size = SAMPLE_RATE * chunk_seconds

  with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=channels,
    dtype="float32",
    blocksize=chunk_size,
    device=device_id,
    callback=callback,
  ):
    threading.Event().wait()


def transcribe_worker(audio_queue: queue.Queue, output_file: Path | None, silence_threshold: float, debug: bool):
  while True:
    chunk = audio_queue.get()
    audio = chunk[:, 0]

    rms = float(np.sqrt(np.mean(audio**2)))

    if debug:
      print(f"[debug] RMS={rms:.5f}  queue={audio_queue.qsize()}  {'OK' if rms >= silence_threshold else 'silence, skipping'}")

    if rms < silence_threshold:
      continue

    try:
      result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_REPO,
        language="pt",
      )
      text = result.get("text", "").strip()
    except Exception as e:
      print(f"[error] {e}", file=sys.stderr)
      continue

    if not text:
      continue

    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {text}"
    print(line, flush=True)

    if output_file:
      with open(output_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
  parser = argparse.ArgumentParser(
    description="Real-time microphone transcription with Whisper (Apple Silicon)."
  )
  parser.add_argument(
    "--mic", "-m", type=int, default=None,
    help="Audio input device ID (default: system microphone)",
  )
  parser.add_argument(
    "--list-devices", action="store_true",
    help="List available audio devices and exit",
  )
  parser.add_argument(
    "--chunk", "-c", type=int, default=5,
    help="Capture chunk duration in seconds (default: 5)",
  )
  parser.add_argument(
    "--output", "-o", type=str, default=None,
    help=".txt file to save the transcription (optional)",
  )
  parser.add_argument(
    "--silence-threshold", "-s", type=float, default=0.01,
    help="RMS silence threshold (default: 0.01)",
  )
  parser.add_argument(
    "--debug", action="store_true",
    help="Show the RMS level of each captured chunk",
  )

  args = parser.parse_args()

  if args.list_devices:
    list_devices()
    return

  output_file = Path(args.output) if args.output else None
  if output_file:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving transcription to: {output_file}")

  print(f"Loading model via MLX ({MODEL_REPO})...")
  # mlx_whisper loads the model on first use — warming up here
  mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL_REPO, language="pt")
  print("Model loaded. Listening to microphone... (Ctrl+C to stop)\n")

  audio_queue: queue.Queue = queue.Queue(maxsize=10)

  worker = threading.Thread(
    target=transcribe_worker,
    args=(audio_queue, output_file, args.silence_threshold, args.debug),
    daemon=True,
  )
  worker.start()

  recorder = threading.Thread(
    target=record_chunks,
    args=(audio_queue, args.chunk, args.mic),
    daemon=True,
  )
  recorder.start()

  try:
    recorder.join()
  except KeyboardInterrupt:
    print("\nTranscription stopped.")


if __name__ == "__main__":
  main()
