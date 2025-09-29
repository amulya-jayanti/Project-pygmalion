#!/usr/bin/env python3
"""
echo_push.py — push each new runner reply to Tavus CVI (Echo Mode)

Watches outbox/<run_id>/tavus/queue.jsonl and, for every new line:
- Sends a Tavus Interaction: message_type="conversation", event_type="conversation.echo"
- By default posts the reply text (properties.text)
- With --use_audio, base64-encodes the ElevenLabs WAV/MP3 and sends as properties.audio

Requires: httpx
    pip install httpx
"""

import os
import time
import json
import argparse
import base64
from pathlib import Path

import httpx


def log(msg: str):
    print(f"[echo] {msg}", flush=True)


def b64_file(path: Path) -> tuple[str, str]:
    """Returns (mime_type, base64_str). Supports .wav and .mp3."""
    ext = path.suffix.lower()
    if ext == ".wav":
        mime = "audio/wav"
    elif ext == ".mp3":
        mime = "audio/mpeg"
    else:
        raise ValueError(f"Unsupported audio type: {ext}")
    data = path.read_bytes()
    return mime, base64.b64encode(data).decode("ascii")


def send_echo_text(interactions_url: str, api_key: str, text: str, conv_id: str | None = None):
    payload = {
        "message_type": "conversation",
        "event_type": "conversation.echo",
        # some deployments accept conversation_id in body; harmless if ignored
        **({"conversation_id": conv_id} if conv_id else {}),
        "properties": {"text": text},
    }
    headers = {
        "x-api-key": api_key,          # Tavus CVI uses x-api-key
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        r = client.post(interactions_url, headers=headers, json=payload)
        r.raise_for_status()


def send_echo_audio(interactions_url: str, api_key: str, audio_path: Path, conv_id: str | None = None):
    mime, b64 = b64_file(audio_path)
    payload = {
        "message_type": "conversation",
        "event_type": "conversation.echo",
        **({"conversation_id": conv_id} if conv_id else {}),
        "properties": {
            "audio": {
                "mime_type": mime,
                "base64": b64
            }
        },
    }
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(interactions_url, headers=headers, json=payload)
        r.raise_for_status()


def _process_line(line: str,
                  interactions_url: str,
                  api_key: str,
                  run_dir: Path,
                  use_audio: bool,
                  conv_id_hint: str | None,
                  startup: bool = False):
    line = line.strip()
    if not line:
        return
    try:
        obj = json.loads(line)
    except Exception as e:
        log(f"skip (bad json): {e}")
        return

    reply = (obj.get("reply_text") or "").strip()
    turn_idx = obj.get("turn_index")
    # optional conversation_id hint (some deployments accept it in body)
    conv_id = obj.get("conversation_id") or conv_id_hint

    # Construct default audio path from known structure:
    # outbox/<run_id>/audio/turn_XXX.wav  (or .mp3)
    audio_path = None
    if isinstance(turn_idx, int):
        audio_dir = run_dir / "audio"
        wav = audio_dir / f"turn_{turn_idx:03d}.wav"
        mp3 = audio_dir / f"turn_{turn_idx:03d}.mp3"
        if wav.exists():
            audio_path = wav
        elif mp3.exists():
            audio_path = mp3

    try:
        if use_audio and audio_path and audio_path.exists():
            send_echo_audio(interactions_url, api_key, audio_path, conv_id)
            log(f"sent AUDIO for turn {turn_idx} -> {audio_path.name}{' (startup)' if startup else ''}")
        elif reply:
            send_echo_text(interactions_url, api_key, reply, conv_id)
            log(f"sent TEXT  for turn {turn_idx}: {reply[:60]}{'…' if len(reply) > 60 else ''}{' (startup)' if startup else ''}")
        else:
            log(f"no text/audio available for turn {turn_idx}; skipping")
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        log(f"HTTP {e.response.status_code}: {body}")
    except Exception as e:
        log(f"send error: {e}")


def tail_queue(queue_path: Path,
               interactions_url: str,
               api_key: str,
               run_dir: Path,
               use_audio: bool,
               convo_id_hint: str | None,
               backfill: int):
    """
    Simple file tailer: remembers byte offset across iterations.
    If backfill > 0, will replay the last N lines on startup before live tailing.
    """
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.touch(exist_ok=True)

    # optional backfill of last N lines
    if backfill > 0:
        try:
            lines = queue_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-backfill:]:
                _process_line(line, interactions_url, api_key, run_dir, use_audio, convo_id_hint, startup=True)
        except Exception as e:
            log(f"backfill error: {e}")

    # live tail
    last_size = queue_path.stat().st_size
    log(f"watching {queue_path.resolve()}")
    log(f"dest: {interactions_url}")
    if use_audio:
        log("mode: AUDIO (base64)")
    else:
        log("mode: TEXT")

    while True:
        try:
            size = queue_path.stat().st_size
            if size < last_size:
                # file truncated/rotated; restart
                last_size = 0
            if size > last_size:
                with queue_path.open("r", encoding="utf-8") as f:
                    f.seek(last_size)
                    for line in f:
                        _process_line(line, interactions_url, api_key, run_dir, use_audio, convo_id_hint)
                last_size = size
        except FileNotFoundError:
            # queue might not exist yet; recreate and retry
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.touch(exist_ok=True)
            last_size = 0
        time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser(description="Push runner replies to Tavus Echo in near-real-time")
    ap.add_argument("--run_id", required=True, help="Run/session id (matches runner.py)")
    ap.add_argument("--outbox_dir", default="outbox", help="Base outbox directory")
    ap.add_argument("--interactions_url", required=True,
                    help="Tavus Interactions endpoint: https://tavusapi.com/v2/conversations/<conversation_id>/interactions")
    ap.add_argument("--token", default=os.getenv("TAVUS_API_KEY", ""), help="x-api-key token")
    ap.add_argument("--use_audio", action="store_true",
                    help="Send ElevenLabs audio (if present) instead of text")
    ap.add_argument("--convo_id", default=None,
                    help="Optional conversation_id to include in payload")
    ap.add_argument("--backfill", type=int, default=0,
                    help="On startup, replay last N queued turns before tailing")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("Missing token. Pass --token or set TAVUS_API_KEY")

    run_dir = Path(args.outbox_dir) / args.run_id
    queue_path = run_dir / "tavus" / "queue.jsonl"

    tail_queue(queue_path, args.interactions_url, args.token, run_dir, bool(args.use_audio), args.convo_id, args.backfill)


if __name__ == "__main__":
    main()
