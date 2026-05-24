import sys
import argparse
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import mlx_whisper
import argostranslate.package
import argostranslate.translate

MODEL_REPO = "mlx-community/whisper-medium-mlx"
SAMPLE_RATE = 16000

def list_devices():
  print(sd.query_devices())


def ensure_translation(source: str, target: str):
  """Install the offline argos-translate package(s) for source -> target.

  Pivots through English when there is no direct package for the pair.
  Requires internet only on the first run to download the model; afterwards
  translation works fully offline.
  """
  installed = {
    (pkg.from_code, pkg.to_code)
    for pkg in argostranslate.package.get_installed_packages()
  }

  argostranslate.package.update_package_index()
  available = argostranslate.package.get_available_packages()

  def install_pair(a: str, b: str) -> bool:
    if (a, b) in installed:
      return True
    pkg = next((p for p in available if p.from_code == a and p.to_code == b), None)
    if pkg is None:
      return False
    print(f"Downloading translation model {a} -> {b} (first run only)...")
    argostranslate.package.install_from_path(pkg.download())
    return True

  if install_pair(source, target):
    return

  # No direct package: pivot through English.
  if install_pair(source, "en") and install_pair("en", target):
    return

  raise RuntimeError(
    f"No offline translation available for {source} -> {target} "
    f"(directly or via English)."
  )


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


def transcribe_worker(audio_queue: queue.Queue, output_file: Path | None, silence_threshold: float, debug: bool, source: str, target: str | None):
  translate = bool(target) and target != source

  while True:
    chunk = audio_queue.get()
    audio = chunk[:, 0]

    # Loudest 0.5s sub-window, so detection doesn't depend on chunk length:
    # a short utterance keeps the same level whether the chunk is 3s or 5s.
    frame = SAMPLE_RATE // 2
    if len(audio) >= frame:
      frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
      level = float(np.sqrt(np.mean(frames**2, axis=1)).max())
    else:
      level = float(np.sqrt(np.mean(audio**2)))

    if debug:
      peak = float(np.max(np.abs(audio)))
      print(f"[debug] level={level:.5f}  peak={peak:.5f}  queue={audio_queue.qsize()}  {'OK' if level >= silence_threshold else 'silence, skipping'}")

    if level < silence_threshold:
      continue

    # Normalize gain so weak speech reaches Whisper at a usable level
    # (only after the silence gate, to avoid amplifying pure noise).
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
      audio = audio / peak * 0.95

    try:
      result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_REPO,
        language=source,
      )
      text = result.get("text", "").strip()
    except Exception as e:
      print(f"[error] {e}", file=sys.stderr)
      continue

    if not text:
      continue

    if translate:
      if debug:
        print(f"[debug] {source}: {text}")
      try:
        text = argostranslate.translate.translate(text, source, target).strip()
      except Exception as e:
        print(f"[error] translation failed: {e}", file=sys.stderr)
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
    "--source-lang", "-f", type=str, default="pt",
    help="Spoken/captured language (ISO code, e.g. pt, en, es). Default: pt",
  )
  parser.add_argument(
    "--target-lang", "-t", type=str, default=None,
    help="Translate the transcription into this language (e.g. en). Default: no translation",
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

  if args.target_lang and args.target_lang != args.source_lang:
    print(f"Preparing offline translation {args.source_lang} -> {args.target_lang}...")
    ensure_translation(args.source_lang, args.target_lang)

  print(f"Loading model via MLX ({MODEL_REPO})...")
  # mlx_whisper loads the model on first use — warming up here
  mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL_REPO, language="pt")
  print("Model loaded. Listening to microphone... (Ctrl+C to stop)\n")

  audio_queue: queue.Queue = queue.Queue(maxsize=10)

  worker = threading.Thread(
    target=transcribe_worker,
    args=(audio_queue, output_file, args.silence_threshold, args.debug, args.source_lang, args.target_lang),
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
