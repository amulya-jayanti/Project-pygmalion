#!/usr/bin/env python3
"""
Concussion ER Patient Simulator (Core)
Mic/File -> Whisper (STT) -> Mem0 (retrieve) -> OpenAI LLM -> Mem0 (save) -> print reply
Adds:
  - Persona seeding (adult male/female, child, elder)
  - Light evaluation signals per turn (distress trend + concussion coverage)
  - Optional transcript export on exit

Env (.env):
  OPENAI_API_KEY=...
  MEM0_API_KEY=...
Optional:
  MODEL_NAME=gpt-4o-mini
  WHISPER_MODEL=base

System deps:
  - ffmpeg (for Whisper)
  - PortAudio (for --mic)
"""

import os, sys, time, json, argparse
from typing import Dict, Any, List, Optional
import httpx, subprocess

# --- Optional .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- STT: Whisper ---
import whisper

# --- Mic recording (optional) ---
try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wavwrite
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

# --- OpenAI LLM ---
from openai import OpenAI

# --- Mem0 ---
from mem0 import MemoryClient

import re

APP_ID   = "er-sim"
AGENT_ID = "patient-avatar-concussion"

SYSTEM_RULES = """You are a distressed ER patient with a suspected concussion. Stay in character.
• Begin highly distressed (short, breathy, scared). Calm gradually if the clinician reassures and asks focused questions.
• Keep replies ≤10 seconds. Speak in short, natural phrases. No medical jargon or diagnoses.
• Be consistent with previously stated facts (mechanism, time since injury, symptoms, meds/allergies).
• If multiple questions arrive, answer the most urgent briefly.
• Do NOT invent vitals/meds. Describe how you feel (headache, nausea, dizziness, vision, neck pain, memory).
• If the clinician uses empathy or guides breathing, reduce distress slightly on the next turn.
"""

# ---------- Personas ----------
PERSONAS = {
  "adult_male": {
    "age": 28, "sex": "male",
    "mechanism": "fell off skateboard hitting head on pavement",
    "time_since": "30 minutes",
    "symptoms": ["frontal pounding headache","nausea","photophobia","dizziness"],
    "loc": "brief loss of consciousness a few seconds",
    "neck_pain": False,
    "vomiting": 1,
    "meds": [],
    "anticoagulants": False,
    "allergies": ["penicillin"],
    "distress_start": 0.85, "distress_min": 0.25
  },
  "adult_female": {
    "age": 34, "sex": "female",
    "mechanism": "car door hit right temple",
    "time_since": "1 hour",
    "symptoms": ["throbbing right-sided headache","dizziness","nausea","blurry vision at times"],
    "loc": "no loss of consciousness",
    "neck_pain": False,
    "vomiting": 0,
    "meds": ["combined birth control pill"],
    "anticoagulants": False,
    "allergies": [],
    "distress_start": 0.8, "distress_min": 0.25
  },
  "child_8yo": {
    "age": 8, "sex": "male",
    "mechanism": "struck by soccer ball in the head",
    "time_since": "20 minutes",
    "symptoms": ["headache","nausea","sleepy","teary"],
    "loc": "no loss of consciousness",
    "neck_pain": False,
    "vomiting": 0,
    "meds": [],
    "anticoagulants": False,
    "allergies": [],
    "distress_start": 0.9, "distress_min": 0.2
  },
  "elder_72f": {
    "age": 72, "sex": "female",
    "mechanism": "ground-level fall in kitchen",
    "time_since": "45 minutes",
    "symptoms": ["worsening headache","unsteady","nausea"],
    "loc": "uncertain loss of consciousness",
    "neck_pain": True,
    "vomiting": 0,
    "meds": ["warfarin"],
    "anticoagulants": True,
    "allergies": [],
    "distress_start": 0.8, "distress_min": 0.25
  }
}

# ---------- Coverage keywords for concussion ----------
KEYWORDS = {
  "LOC": ["lose consciousness","pass out","black out","knocked out","did you faint"],
  "AMNESIA": ["remember","memory","before","after"],
  "VOMIT": ["vomit","throw up","emesis","nausea"],
  "HEADACHE": ["headache","how bad","severe pain"],
  "VISION": ["blurry","double","vision","bright light","photophobia"],
  "NECK": ["neck pain","sore neck","stiff neck"],
  "SEIZURE": ["seizure","shaking","convulsion"],
  "ANTICOAG": ["blood thinner","warfarin","eliquis","xarelto","plavix"],
  "MECH": ["what happened","how did it happen","mechanism","hit your head","fall"],
  "TIME": ["when did it start","how long","time since","minutes","hours"]
}

REASSURE = [
    "you're safe","you are safe","breathe with me","slow breaths","i'm here",
    "i am here","with you","one step at a time","we'll go slow","we will go slow"
]

JARGON = ["emesis","photophobia","post-ictal","gcs","loc","anticoagulants"]
DISMISSIVE = ["probably fine","it's mild","you'll be fine","you will be fine"]
LEADING_PATTERNS = [r"\bso you (did|didn'?t)\b", r"\byou didn'?t\b"]

def _normalize(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    s = re.sub(r"\s+"," ",s).strip().lower()
    return s

def _features(utterance: str) -> dict:
    """Extract quality features of the clinician line."""
    t = _normalize(utterance)
    qmarks = t.count("?")
    commas = t.count(",")
    words = len(t.split())
    stacked = (qmarks >= 2) or (commas >= 2) or (words > 25)  # rough heuristic
    jargon = any(w in t for w in JARGON)
    dismissive = any(p in t for p in DISMISSIVE)
    leading = any(re.search(p, t) for p in LEADING_PATTERNS)
    reassurance = any(p in t for p in REASSURE)
    focused = (not stacked) and (not jargon) and (not leading) and (not dismissive)
    return {
        "reassurance": reassurance,
        "stacked": stacked,
        "jargon": jargon,
        "dismissive": dismissive,
        "leading": leading,
        "focused": focused
    }
# ---------- Mem0 helpers ----------
def mem0_search(mem, *, query, user_id, agent_id, app_id, run_id, filters=None, limit=8):
    filters = filters or {}
    query = "" if query is None else str(query)
    if hasattr(mem, "memories") and hasattr(mem.memories, "search"):
        return mem.memories.search(query=query, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id, filters=filters, limit=limit)
    if hasattr(mem, "search"):
        return mem.search(query=query, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id, filters=filters, limit=limit)
    raise AttributeError("Mem0 client missing search()")

def mem0_add(mem, *, messages, user_id, agent_id, app_id, run_id, metadata=None):
    metadata = metadata or {}
    if hasattr(mem, "memories") and hasattr(mem.memories, "add"):
        return mem.memories.add(messages=messages, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id, metadata=metadata)
    if hasattr(mem, "add"):
        return mem.add(messages=messages, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id, metadata=metadata)
    raise AttributeError("Mem0 client missing add()")

def mem0_get_all(mem, *, user_id, agent_id=None, app_id=None, run_id=None, limit=400):
    """
    Normalize Mem0 get_all across SDK variants and optional kwargs.
    Tries the most specific signature first, then backs off if unsupported.
    Returns a Python list of memory items.
    """
    # Prefer namespaced client if available
    target = None
    if hasattr(mem, "memories") and hasattr(mem.memories, "get_all"):
        target = mem.memories.get_all
    elif hasattr(mem, "get_all"):
        target = mem.get_all
    else:
        raise AttributeError("Mem0 client missing get_all()")

    # Try the most complete call, then back off on TypeError
    tries = [
        dict(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id, limit=limit),
        dict(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id),
        dict(user_id=user_id, agent_id=agent_id, app_id=app_id),
        dict(user_id=user_id, agent_id=agent_id),
        dict(user_id=user_id),
    ]
    for kwargs in tries:
        try:
            res = target(**{k:v for k,v in kwargs.items() if v is not None})
            return res if isinstance(res, list) else (res.get("data") or [])
        except TypeError:
            continue  # argument not supported in this SDK, try a simpler call
    return []

def _mem0_extract_messages(item):
    if not isinstance(item, dict):
        return []
    if isinstance(item.get("messages"), list):
        return item["messages"]
    mem = item.get("memory")
    if isinstance(mem, dict) and isinstance(mem.get("messages"), list):
        return mem["messages"]
    return []

def _mem0_extract_metadata(item):
    if not isinstance(item, dict):
        return {}
    meta = {}
    if isinstance(item.get("memory"), dict):
        meta = item["memory"].get("metadata") or {}
    return item.get("metadata", meta) or {}


# ---------- Init / clients ----------
def ensure_env():
    miss=[]
    if not os.getenv("OPENAI_API_KEY"): miss.append("OPENAI_API_KEY")
    if not os.getenv("MEM0_API_KEY"): miss.append("MEM0_API_KEY")
    if miss: raise RuntimeError("Missing env: " + ", ".join(miss))

def load_clients():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

def load_whisper(name:str): return whisper.load_model(name)

# ---------- Persona seeding & state ----------
def seed_persona(mem, scope:Dict[str,str], persona_key:str) -> Dict[str,Any]:
    p = PERSONAS[persona_key]
    info = {
      "persona": persona_key,
      "age": p["age"], "sex": p["sex"],
      "mechanism": p["mechanism"], "time_since": p["time_since"],
      "symptoms": p["symptoms"], "loc": p["loc"],
      "neck_pain": p["neck_pain"], "vomiting": p["vomiting"],
      "meds": p["meds"], "anticoagulants": p["anticoagulants"],
    }
    mem0_add(mem,
      messages=[
        {"role":"system","content":f"SESSION SETUP: {json.dumps(info)}"},
        {"role":"assistant","content":"(internal) Persona seeded for consistency."}
      ],
      user_id=scope["user_id"], agent_id=scope["agent_id"], app_id=scope["app_id"], run_id=scope["run_id"],
      metadata={"setup":True, "ts":int(time.time())}
    )
    return {
        "distress": p["distress_start"],
        "distress_min": p["distress_min"],
        "coverage": {k: False for k in KEYWORDS.keys()},
        "coverage_first_ts": {k: None for k in KEYWORDS.keys()},
        "turns": 0,
        "started_at": time.time(),
        "persona": persona_key 
    }   

# def update_eval_signals(clinician_text: str, state: Dict[str, Any], persona_key: str):
#     txt = _normalize(clinician_text)
#     state["turns"] += 1

#     now = time.time()
#     # coverage: flip to True when first detected and stamp first_ts
#     for k, words in KEYWORDS.items():
#         if not state["coverage"][k]:
#             if any(_normalize(w) in txt for w in words):
#                 state["coverage"][k] = True
#                 if state["coverage_first_ts"][k] is None:
#                     state["coverage_first_ts"][k] = now

#     # reassurance → reduce distress
#     if any(_normalize(w) in txt for w in REASSURE) or sum(state["coverage"].values()) >= 3:
#         state["distress"] = max(state["distress_min"], state["distress"] - 0.1)

#     return state

def update_eval_signals(clinician_text: str, state: Dict[str, Any], persona_key: str):
    # coverage / timing
    txt = _normalize(clinician_text)
    state["turns"] = state.get("turns", 0) + 1
    now = time.time()

    # Flip coverage flags & first_ts when detected
    for k, words in KEYWORDS.items():
        if not state["coverage"][k]:
            if any(_normalize(w) in txt for w in words):
                state["coverage"][k] = True
                if state.get("coverage_first_ts") is not None:
                    state["coverage_first_ts"][k] = now

    # Quality features
    f = _features(clinician_text)

    # Early gating: within first 3 turns, allow calm only if they asked MECH or LOC or TIME
    early_ok = state["turns"] > 3 or any(state["coverage"][k] for k in ("MECH","LOC","TIME"))

    # Distress update
    delta = 0.0
    if f["dismissive"] or f["leading"] or f["stacked"] or f["jargon"]:
        # Poor technique → distress rises a bit
        delta += +0.08
    elif f["reassurance"] and f["focused"] and early_ok:
        # Good reassurance + focused question → calm
        delta += -0.12
    else:
        # Neutral: small drift toward current level (no change)
        delta += 0.0

    # Apply and clamp
    new_distress = min(1.0, max(state["distress_min"], state["distress"] + delta))
    state["distress"] = new_distress
    # (Optional) keep a reason for debugging/printing
    state["last_reason"] = {
        "features": f,
        "delta": round(delta, 3),
        "early_ok": early_ok
    }
    return state

def compute_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    covered = [k for k, v in state["coverage"].items() if v]
    coverage_pct = round(100.0 * len(covered) / max(1, len(state["coverage"])), 1)

    # Example “critical” checks for concussion
    critical_keys = ["LOC", "MECH", "TIME", "HEADACHE", "VOMIT", "VISION", "NECK", "ANTICOAG"]
    critical_covered = sum(1 for k in critical_keys if state["coverage"].get(k))
    critical_pct = round(100.0 * critical_covered / len(critical_keys), 1)

    # Time to first LOC/mechanism question (seconds since session start)
    def first_delta(k):
        ts = state["coverage_first_ts"].get(k)
        return round(ts - state["started_at"], 1) if ts else None

    summary = {
        "turns": state["turns"],
        "final_distress": round(state["distress"], 3),
        "coverage_percent_all": coverage_pct,
        "coverage_percent_critical": critical_pct,
        "first_time_to_LOC_sec": first_delta("LOC"),
        "first_time_to_MECH_sec": first_delta("MECH"),
        "first_time_to_TIME_sec": first_delta("TIME"),
        "covered_keys": covered,
        "missed_keys": [k for k in state["coverage"] if not state["coverage"][k]],
    }

    # naive overall score (weights are hackathon-simple)
    score = (
        critical_pct * 0.6 +             # focus on critical coverage
        coverage_pct * 0.2 +             # breadth
        (1.0 - state["distress"]) * 100 * 0.2  # calmer is better
    )
    summary["overall_score"] = round(score, 1)
    return summary

import httpx
import shutil

def elevenlabs_tts(text: str, voice_id: str, xi_api_key: str, model: str = "eleven_turbo_v2", fmt: str = "mp3") -> bytes:
    """
    Returns raw audio bytes (mp3 or wav) for the given text using ElevenLabs TTS.
    """
    if not text:
        return b""
    if not voice_id:
        raise ValueError("Missing voice_id for ElevenLabs TTS")
    if not xi_api_key:
        raise ValueError("Missing ELEVENLABS_API_KEY")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": xi_api_key, "accept": "audio/mpeg" if fmt=="mp3" else "audio/wav", "content-type": "application/json"}
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            # sane defaults; your UE team can tweak later
            "stability": 0.4,
            "similarity_boost": 0.7,
            "style": 0.2,
            "use_speaker_boost": True
        }
    }
    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.content

# ---------- Core steps ----------
def transcribe(model, audio_path:str)->str:
    r = model.transcribe(audio_path)
    t = (r or {}).get("text","")
    return t.strip() if isinstance(t,str) else ""

# def retrieve_context(mem, user_text:str, scope)->str:
#     if not user_text: return ""
#     try:
#         res = mem0_search(mem, query=user_text, user_id=scope["user_id"], agent_id=scope["agent_id"], app_id=scope["app_id"], run_id=scope["run_id"], limit=8)
#     except Exception as e:
#         print(f"[mem0] search warn: {e}"); return ""
#     lines=[]
#     print(res)
#     for item in (res.get("data") or []):
#         for m in ((item.get("memory") or {}).get("messages") or []):
#             role, content = m.get("role"), (m.get("content") or "").strip()
#             if content: lines.append(f"{role}: {content}")
#     return "\n".join(lines)[-3500:]

def retrieve_context(mem, user_text: str, scope) -> str:
    if not user_text:
        return ""
    try:
        res = mem0_search(
            mem,
            query=user_text,
            user_id=scope["user_id"],
            agent_id=scope["agent_id"],
            app_id=scope["app_id"],
            run_id=scope["run_id"],
            limit=8
        )
    except Exception as e:
        print(f"[mem0] search warn: {e}")
        return ""

    lines = []
    # res is a list, so iterate directly
    for item in res:
        # Messages may be under item['messages'] or item['memory']['messages']
        msgs = []
        if isinstance(item, dict):
            if isinstance(item.get("messages"), list):
                msgs = item["messages"]
            elif isinstance(item.get("memory"), dict) and isinstance(item["memory"].get("messages"), list):
                msgs = item["memory"]["messages"]

        for m in msgs:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")

    # Keep context bounded
    return "\n".join(lines)[-3500:]


def call_llm(client, system_rules:str, context:str, user_text:str, model:str)->str:
    prompt = f"""[SESSION CONTEXT]
{context}

[CLINICIAN SAID]
{user_text}

[RESPONSE STYLE]
≤10s, distressed but coherent. No invented vitals/meds. Be consistent. Brief, effortful speech."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system_rules},{"role":"user","content":prompt}],
        temperature=0.5, max_tokens=200
    )
    return resp.choices[0].message.content.strip()

def save_turn(mem, user_text:str, assistant_text:str, scope, state:Dict[str,Any]):
    meta = {
      "ts": int(time.time()),
      "distress": state["distress"],
      "coverage": state["coverage"]
    }
    mem0_add(mem,
      messages=[{"role":"user","content":user_text},{"role":"assistant","content":assistant_text}],
      user_id=scope["user_id"], agent_id=scope["agent_id"], app_id=scope["app_id"], run_id=scope["run_id"],
      metadata=meta
    )
# def write_outbound_files(
#     reply_text: str,
#     scope: Dict[str, str],
#     state: Dict[str, Any],
#     outbox_dir: str,
#     *,
#     elevenlabs: bool = False,
#     voice_id: Optional[str] = None,
#     xi_api_key: Optional[str] = None,
#     tts_model: str = "eleven_turbo_v2",
#     tts_format: str = "mp3",
# ):
#     state["turn_index"] = state.get("turn_index", 0) + 1
#     idx = state["turn_index"]

#     run_dir = os.path.join(outbox_dir, scope["run_id"])
#     os.makedirs(run_dir, exist_ok=True)

#     # base files
#     base = f"turn_{idx:03d}"
#     txt_path = os.path.join(run_dir, f"{base}.txt")
#     json_path = os.path.join(run_dir, f"{base}.json")
#     q_path = os.path.join(run_dir, "queue.jsonl")

#     with open(txt_path, "w") as f:
#         f.write(reply_text or "")

#     payload = {
#         "ts": int(time.time()),
#         "run_id": scope["run_id"],
#         "user_id": scope["user_id"],
#         "agent_id": scope["agent_id"],
#         "app_id": scope["app_id"],
#         "turn_index": idx,
#         "persona": state.get("persona"),
#         "reply_text": reply_text or "",
#         "distress": state.get("distress"),
#         "coverage": state.get("coverage"),
#     }
#     with open(json_path, "w") as f:
#         json.dump(payload, f, indent=2)
#     with open(q_path, "a") as f:
#         f.write(json.dumps(payload) + "\n")

#     audio_path = None
#     if elevenlabs:
#         try:
#             audio_bytes = elevenlabs_tts(
#                 reply_text or "",
#                 voice_id=voice_id,
#                 xi_api_key=xi_api_key or os.getenv("ELEVENLABS_API_KEY",""),
#                 model=tts_model,
#                 fmt=tts_format
#             )
#             audio_dir = os.path.join(run_dir, "audio")
#             os.makedirs(audio_dir, exist_ok=True)
#             ext = "mp3" if tts_format == "mp3" else "wav"
#             audio_path = os.path.join(audio_dir, f"{base}.{ext}")
#             with open(audio_path, "wb") as af:
#                 af.write(audio_bytes)
#         except Exception as e:
#             print(f"[warn] ElevenLabs TTS failed: {e}")

#     return {"txt": txt_path, "json": json_path, "queue": q_path, "audio": audio_path}

def write_outbound_files(
    reply_text: str,
    scope: Dict[str, str],
    state: Dict[str, Any],
    outbox_dir: str,
    *,
    elevenlabs: bool = False,
    voice_id: Optional[str] = None,
    xi_api_key: Optional[str] = None,
    tts_model: str = "eleven_turbo_v2",
    tts_format: str = "mp3",
):
    state["turn_index"] = state.get("turn_index", 0) + 1
    idx = state["turn_index"]

    run_dir = os.path.join(outbox_dir, scope["run_id"])
    os.makedirs(run_dir, exist_ok=True)

    base = f"turn_{idx:03d}"
    txt_path  = os.path.join(run_dir, f"{base}.txt")
    json_path = os.path.join(run_dir, f"{base}.json")
    q_path    = os.path.join(run_dir, "queue.jsonl")

    with open(txt_path, "w") as f:
        f.write(reply_text or "")

    payload = {
        "ts": int(time.time()),
        "run_id": scope["run_id"],
        "user_id": scope["user_id"],
        "agent_id": scope["agent_id"],
        "app_id": scope["app_id"],
        "turn_index": idx,
        "persona": state.get("persona"),
        "reply_text": reply_text or "",
        "distress": state.get("distress"),
        "coverage": state.get("coverage"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(q_path, "a") as f:
        f.write(json.dumps(payload) + "\n")

    audio_path = None
    if elevenlabs:
        # Always prepare audio dir + paths first
        audio_dir = os.path.join(run_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        ext      = "mp3" if tts_format == "mp3" else "wav"
        audio_path = os.path.join(audio_dir, f"{base}.{ext}")
        err_path   = os.path.join(audio_dir, f"{base}.err")

        try:
            # Debug print so you can see what it's trying:
            print(f"[tts] ElevenLabs: voice_id={voice_id!r}, fmt={ext}, model={tts_model}")
            audio_bytes = elevenlabs_tts(
                reply_text or "",
                voice_id=voice_id,
                xi_api_key=xi_api_key or os.getenv("ELEVENLABS_API_KEY",""),
                model=tts_model,
                fmt=tts_format
            )
            with open(audio_path, "wb") as af:
                af.write(audio_bytes)
        except Exception as e:
            with open(err_path, "w") as ef:
                ef.write(str(e))
            print(f"[warn] ElevenLabs TTS failed for turn {idx}: {e}")
            audio_path = None  # indicate failure to caller

    return {"txt": txt_path, "json": json_path, "queue": q_path, "audio": audio_path}

import httpx

def elevenlabs_list_voices(xi_api_key: str) -> list[dict]:
    """Return list of voices available to this API key."""
    if not xi_api_key:
        raise ValueError("Missing ELEVENLABS_API_KEY")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": xi_api_key}
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json() or {}
        voices = data.get("voices") or []
        return voices

def resolve_voice_id(voice_id: Optional[str], voice_name: Optional[str], xi_api_key: str) -> str:
    """
    If voice_id provided, verify it exists for this key; otherwise resolve by name.
    Raises with a helpful message if no match.
    """
    voices = elevenlabs_list_voices(xi_api_key)
    # quick index
    by_id = {v.get("voice_id"): v for v in voices}
    by_name_lower = { (v.get("name") or "").strip().lower(): v for v in voices }

    if voice_id:
        if voice_id in by_id:
            return voice_id
        # maybe user pasted a library/share id; try to find same id fragment
        matches = [vid for vid in by_id if voice_id.lower() in (vid or "").lower()]
        if matches:
            return matches[0]
        raise RuntimeError(
            f"ElevenLabs voice_id not found for this API key: {voice_id}\n"
            f"Available voices: {', '.join([v.get('name','?')+'('+v.get('voice_id','')+')' for v in voices])}"
        )

    if voice_name:
        v = by_name_lower.get(voice_name.strip().lower())
        if v and v.get("voice_id"):
            return v["voice_id"]
        raise RuntimeError(
            f"ElevenLabs voice name not found: {voice_name}\n"
            f"Available voices: {', '.join([v.get('name','?') for v in voices])}"
        )

    # Nothing provided: pick the first voice to be helpful
    if voices:
        print(f"[tts] No voice given; defaulting to first: {voices[0].get('name')} ({voices[0].get('voice_id')})")
        return voices[0].get("voice_id")

    raise RuntimeError("No ElevenLabs voices available on this API key.")

def _tavus_headers():
    key = os.getenv("TAVUS_API_KEY","")
    if not key:
        raise RuntimeError("Missing TAVUS_API_KEY (set in .env or export)")
    return {"Authorization": f"Bearer {key}"}

def tavus_submit_text(text: str, tavus_id: str) -> dict:
    """
    Create a Tavus video from text. Supports either persona_id or replica_id,
    since accounts may label it differently.
    Returns JSON with at least a video id (e.g., {'id': '...'}).
    """
    if not text:
        raise ValueError("tavus_submit_text: empty text")
    if not tavus_id:
        raise ValueError("tavus_submit_text: missing tavus_id")

    url = "https://api.tavus.io/v2/videos"
    headers = _tavus_headers()

    # Try persona_id first; if API complains, retry as replica_id
    for key_name in ("persona_id", "replica_id"):
        payload = {key_name: tavus_id, "input": text}
        try:
            with httpx.Client(timeout=60) as client:
                r = client.post(url, headers=headers, json=payload)
                if r.status_code == 400 and "persona" in r.text.lower() and key_name == "persona_id":
                    continue  # retry with replica_id
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            # If first attempt fails due to key name, loop continues; otherwise raise
            if key_name == "replica_id":
                raise
            continue
    raise RuntimeError("Tavus submit failed for both persona_id and replica_id payloads.")

def tavus_check(video_id: str) -> dict:
    url = f"https://api.tavus.io/v2/videos/{video_id}"
    headers = _tavus_headers()
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()

def tavus_poll_until_ready(video_id: str, timeout_s: int = 180, interval_s: float = 3.0) -> dict:
    import time
    start = time.time()
    last_status = None
    while time.time() - start < timeout_s:
        meta = tavus_check(video_id)
        status = (meta.get("status") or "").lower()
        if status in ("done", "completed", "complete", "finished", "success"):
            return meta
        if status != last_status:
            print(f"[tavus] status: {status or 'unknown'}")
            last_status = status
        time.sleep(interval_s)
    raise TimeoutError("Tavus video not ready before timeout")

def tavus_download(meta: dict, out_path: str) -> str:
    """
    Downloads the generated video (mp4) to out_path.
    Tavus responses usually include 'download_url' or 'video_url'.
    """
    url = meta.get("download_url") or meta.get("video_url") or meta.get("url")
    if not url:
        raise RuntimeError(f"No downloadable URL in Tavus response: {meta}")
    with httpx.Client(timeout=120) as client:
        r = client.get(url)
        r.raise_for_status()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(r.content)
    return out_path

def tavus_list_all_min():
    headers = {"Authorization": f"Bearer {os.getenv('TAVUS_API_KEY','')}"}
    items = []
    # Try personas
    try:
        r = httpx.get("https://api.tavus.io/v2/personas", headers=headers, timeout=30)
        if r.status_code < 400:
            data = r.json() or {}
            for x in (data.get("data") or data.get("personas") or []):
                items.append(("persona", x.get("id") or x.get("persona_id"), x.get("name") or x.get("title") or "(unnamed)"))
    except Exception:
        pass
    # Try replicas
    try:
        r = httpx.get("https://api.tavus.io/v2/replicas", headers=headers, timeout=30)
        if r.status_code < 400:
            data = r.json() or {}
            for x in (data.get("data") or data.get("replicas") or []):
                items.append(("replica", x.get("id") or x.get("replica_id"), x.get("name") or x.get("title") or "(unnamed)"))
    except Exception:
        pass
    return items

def open_file_mac(path: str):
    """Play/open a file on macOS.

    Prefer `afplay` (direct audio playback) when available for WAV/MP3.
    Fall back to `open` for other types or when afplay isn't present.
    """
    try:
        if shutil.which("afplay"):
            # afplay is a lightweight local audio player on macOS
            subprocess.run(["afplay", path], check=False)
            return
    except Exception:
        pass
    try:
        subprocess.run(["open", path], check=False)
    except Exception:
        pass
# ---------- IO helpers ----------
def record_from_mic(seconds=8.0, samplerate=16000, channels=1, out_path="clinician_temp.wav")->str:
    if not HAS_SOUNDDEVICE: raise RuntimeError("Mic mode requires sounddevice + scipy.")
    print(f"[mic] Recording {seconds:.1f}s...")
    import numpy as np
    audio = sd.rec(int(seconds*samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    wavwrite(out_path, samplerate, audio)
    print(f"[mic] Saved {out_path}")
    return out_path

# def export_transcript(mem, scope, out_path=None)->str:
#     try:
#         res = mem0_search(mem, query="*", user_id=scope["user_id"], agent_id=scope["agent_id"], app_id=scope["app_id"], run_id=scope["run_id"], limit=200)
#     except Exception as e:
#         print(f"[mem0] export warn: {e}"); res={"data":[]}
#     items=res.get("data") or []
#     convo=[]
#     for it in items:
#         meta=(it.get("memory") or {}).get("metadata") or {}
#         ts=meta.get("ts",0)
#         for m in (it.get("memory") or {}).get("messages") or []:
#             convo.append((ts, m.get("role",""), (m.get("content") or "").strip()))
#     convo.sort(key=lambda x:(x[0], 0 if x[1]=="user" else 1))
#     lines=[]
#     for ts, role, content in convo:
#         tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
#         lines.append(f"[{tstr}] {role.upper()}: {content}")
#     text="\n".join(lines)
#     out_path = out_path or f"transcript_{scope['run_id']}.txt"
#     with open(out_path,"w") as f: f.write(text)
#     return out_path


# def export_transcript(mem, scope, out_path=None) -> str:
#     try:
#         res = mem0_search(
#             mem,
#             query="",
#             user_id=scope["user_id"],
#             agent_id=scope["agent_id"],
#             app_id=scope["app_id"],
#             run_id=scope["run_id"],
#             limit=200
#         )
#     except Exception as e:
#         print(f"[mem0] export warn: {e}")
#         res = []

#     convo = []
#     print(f"Result is {res}")
#     # ✅ res is a list
#     for it in res:
#         meta = {}
#         if isinstance(it, dict):
#             # try nested metadata first
#             if isinstance(it.get("memory"), dict):
#                 meta = it["memory"].get("metadata") or {}
#             meta = it.get("metadata", meta) or {}

#         ts = meta.get("ts", 0)

#         # messages may be at top-level or nested under "memory"
#         msgs = []
#         if isinstance(it.get("messages"), list):
#             msgs = it["messages"]
#         elif isinstance(it.get("memory"), dict) and isinstance(it["memory"].get("messages"), list):
#             msgs = it["memory"]["messages"]

#         for m in msgs:
#             role = m.get("role", "")
#             content = (m.get("content") or "").strip()
#             if content:
#                 convo.append((ts, role, content))

#     # sort by timestamp then user→assistant
#     convo.sort(key=lambda x: (x[0], 0 if x[1] == "user" else 1))

#     lines = []
#     for ts, role, content in convo:
#         tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
#         lines.append(f"[{tstr}] {role.upper()}: {content}")

#     text = "\n".join(lines)
#     out_path = out_path or f"transcript_{scope['run_id']}.txt"
#     with open(out_path, "w") as f:
#         f.write(text)

#     return out_path

# def export_transcript(mem, scope, out_path=None, local_log=None) -> str:
#     try:
#         items = mem0_get_all(
#             mem,
#             user_id=scope["user_id"],
#             agent_id=scope.get("agent_id"),
#             app_id=scope.get("app_id"),
#             run_id=scope.get("run_id"),
#             limit=400
#         )
#         print(mem.get_all(user_id=scope["user_id"], agent_id=scope.get("agent_id"), app_id=scope.get("app_id"), run_id=scope.get("run_id"), limit=5))
#     except Exception as e:
#         print(f"[mem0] get_all warn: {e}")
#         items = []
#     print(f"[mem0] get_all returned {items}")
#     lines = []
#     if items:
#         convo = []
#         for it in items:
#             meta = _mem0_extract_metadata(it)
#             ts = meta.get("ts", 0)
#             msgs = _mem0_extract_messages(it)
#             for m in msgs:
#                 role = m.get("role", "")
#                 content = (m.get("content") or "").strip()
#                 if content:
#                     convo.append((ts, role, content))
#         # sort: timestamp then user→assistant to keep pairs together
#         convo.sort(key=lambda x: (x[0], 0 if x[1] == "user" else 1))
#         for ts, role, content in convo:
#             tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
#             lines.append(f"[{tstr}] {role.upper()}: {content}")

#     # Optional local fallback so your demo never ends blank
#     if not lines and local_log:
#         local_sorted = sorted(local_log, key=lambda x: (x[0], 0 if x[1] == "user" else 1))
#         for ts, role, content in local_sorted:
#             tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
#             lines.append(f"[{tstr}] {role.upper()}: {content}")

#     if not lines:
#         lines = ["[no turns found via Mem0 get_all and no local log provided]"]

#     text = "\n".join(lines)
#     out_path = out_path or f"transcript_{scope['run_id']}.txt"
#     with open(out_path, "w") as f:
#         f.write(text)
#     return out_path

# ---------- CLI ----------
def parse_args():
    p=argparse.ArgumentParser(description="Concussion ER Patient Simulator (Core)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mic", action="store_true")
    g.add_argument("--in", dest="in_wav")
    p.add_argument("--user_id", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--persona", choices=list(PERSONAS.keys()), default="adult_male")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--samplerate", type=int, default=16000)
    p.add_argument("--llm_model", default=os.getenv("MODEL_NAME","gpt-4o-mini"))
    p.add_argument("--whisper_model", default=os.getenv("WHISPER_MODEL","base"))
    # p.add_argument("--export_transcript", action="store_true")
    p.add_argument("--save_last", default="last_reply.txt")
    p.add_argument("--outbox_dir", default="outbox", help="Directory to write outbound turn files")
    p.add_argument("--elevenlabs", action="store_true", help="Generate audio via ElevenLabs for each reply")
    p.add_argument("--voice_id", default=os.getenv("ELEVENLABS_VOICE_ID"), help="ElevenLabs voice id")
    p.add_argument("--tts_model", default=os.getenv("ELEVENLABS_MODEL","eleven_turbo_v2"), help="ElevenLabs model")
    p.add_argument("--tts_format", default="mp3", choices=["mp3","wav"], help="Audio format to save")
    p.add_argument("--voice_name", help="ElevenLabs voice name (we will resolve to an ID)")
    p.add_argument("--list_voices", action="store_true", help="List available ElevenLabs voices and exit")
    p.add_argument("--autoplay", action="store_true", help="Auto-play WAV audio each turn (requires --tts_format wav)")
    p.add_argument("--tavus", action="store_true", help="Send each reply to Tavus to generate a talking-head clip")
    p.add_argument("--tavus_id", default=os.getenv("TAVUS_ID"), help="Tavus persona/replica ID from the dashboard")
    p.add_argument("--tavus_poll", action="store_true", help="Poll Tavus until the clip is ready, then download it")
    p.add_argument("--tavus_open", action="store_true", help="Open the downloaded Tavus clip after saving (macOS 'open')")
    p.add_argument("--tavus_list", action="store_true", help="List Tavus personas/replicas visible to this API key and exit")

    return p.parse_args()

def main():
    ensure_env()
    args = parse_args()
    # ElevenLabs API key and optional voice resolution/listing
    xi_key = os.getenv("ELEVENLABS_API_KEY", "")

    if args.tavus_list:
        items = tavus_list_all_min()
        if not items:
            print("No Tavus personas/replicas visible to this API key.")
        else:
            print("\nTavus entities for this API key:")
            for kind, _id, name in items:
                print(f"- {name} ({kind}:{_id})")
        return


    # If user just wants to list voices, do it and exit.
    if args.list_voices:
        try:
            voices = elevenlabs_list_voices(xi_key)
            print("\nYour ElevenLabs voices:")
            for v in voices:
                print(f"- {v.get('name')}  (id: {v.get('voice_id')})")
            sys.exit(0)
        except Exception as e:
            print(f"[tts] List voices failed: {e}")
            sys.exit(1)

    # Resolve/verify the voice ID once; store it back into args.voice_id
    if args.elevenlabs:
        try:
            args.voice_id = resolve_voice_id(args.voice_id, args.voice_name, xi_key)
            print(f"[tts] Using ElevenLabs voice_id: {args.voice_id}")
        except Exception as e:
            print(f"[tts] Voice resolution failed: {e}")
            sys.exit(1)
    openai_client, mem = load_clients()
    whisper_model = load_whisper(args.whisper_model)

    scope = {"app_id":APP_ID, "agent_id":AGENT_ID, "user_id":args.user_id, "run_id":args.run_id}

    # seed persona once per run
    state = seed_persona(mem, scope, args.persona)

    def run_one(audio_path:str):
        user_text = transcribe(whisper_model, audio_path)
        if not user_text:
            return {"error":"Empty transcription"}
        # eval → update distress/coverage
        update_eval_signals(user_text, state, args.persona)
        ctx = retrieve_context(mem, user_text, scope)
        reply = call_llm(openai_client, SYSTEM_RULES, ctx, user_text, model=args.llm_model)
        save_turn(mem, user_text, reply, scope, state)
        # keep last reply file as-is
        if args.save_last:
            with open(args.save_last,"w") as f: f.write(reply)

        # write structured outbound files (text, json, optional audio) to outbox
        try:
            paths = write_outbound_files(
                reply,
                scope,
                state,
                args.outbox_dir,
                elevenlabs=args.elevenlabs,
                voice_id=args.voice_id,
                xi_api_key=os.getenv("ELEVENLABS_API_KEY",""),
                tts_model=args.tts_model,
                tts_format=args.tts_format
            )
        
            print(f"[outbox] wrote {paths['txt']} and {paths['json']}" + (f" and {paths['audio']}" if paths.get("audio") else ""))
        except Exception as e:
            print(f"[outbox] write failed: {e}")
        # Auto-play WAV audio each turn if requested and available
        try:
            if args.autoplay and paths.get("audio") and os.path.exists(paths.get("audio")):
                # only autoplay WAV on macOS as configured
                if args.tts_format == "wav":
                    open_file_mac(paths.get("audio"))
        except Exception as e:
            print(f"[autoplay] failed: {e}")
        # Optional: submit reply text to Tavus to generate a talking-head clip
        if args.tavus:
            tavus_dir = os.path.join(args.outbox_dir, scope["run_id"], "tavus")
            os.makedirs(tavus_dir, exist_ok=True)
            # Attempt to submit to Tavus if an ID is provided; but always queue a line
            video_id = None
            submit_ok = False
            if args.tavus_id:
                try:
                    submit = tavus_submit_text(reply, args.tavus_id)
                    video_id = submit.get("id") or submit.get("video_id") or submit.get("uuid")
                    pend_path = os.path.join(tavus_dir, f"turn_{state['turn_index']:03d}.pending.json")
                    with open(pend_path, "w") as f:
                        json.dump({"video_id": video_id, "submitted_at": int(time.time())}, f, indent=2)
                    print(f"[tavus] submitted video_id={video_id}")
                    submit_ok = True
                except Exception as e:
                    err_path = os.path.join(tavus_dir, f"turn_{state['turn_index']:03d}.err")
                    with open(err_path, "w") as f:
                        f.write(str(e))
                    print(f"[tavus] submit ERROR: {e}")

            # Always append a line to the tavus queue so echo_push can pick it up and send audio
            try:
                tavus_queue = os.path.join(tavus_dir, "queue.jsonl")
                payload = {
                    "ts": int(time.time()),
                    "run_id": scope["run_id"],
                    "user_id": scope["user_id"],
                    "agent_id": scope["agent_id"],
                    "app_id": scope["app_id"],
                    "turn_index": state.get("turn_index"),
                    "persona": state.get("persona"),
                    "reply_text": reply or "",
                    "distress": state.get("distress"),
                    "coverage": state.get("coverage"),
                    "video_id": video_id,
                }
                with open(tavus_queue, "a") as qf:
                    qf.write(json.dumps(payload) + "\n")
                print(f"[tavus] queued for echo_push: {tavus_queue}")
            except Exception as e:
                print(f"[tavus] failed to write tavus queue line: {e}")

            # If user requested polling and we have a video id, poll & download
            if args.tavus_poll and video_id:
                try:
                    meta = tavus_poll_until_ready(video_id, timeout_s=300, interval_s=3.0)
                    mp4_path = os.path.join(tavus_dir, f"turn_{state['turn_index']:03d}.mp4")
                    tavus_download(meta, mp4_path)
                    print(f"[tavus] saved {mp4_path}")
                    if args.tavus_open:
                        open_file_mac(mp4_path)
                except Exception as e:
                    err_path = os.path.join(tavus_dir, f"turn_{state['turn_index']:03d}.err")
                    with open(err_path, "w") as f:
                        f.write(str(e))
                    print(f"[tavus] poll/download ERROR: {e}")
        return {"user_text": user_text, "assistant_text": reply, "distress": state["distress"], "coverage": state["coverage"]}


    if args.in_wav:
        out = run_one(args.in_wav)
        print(json.dumps(out, indent=2))
        # if args.export_transcript:
        #     path=export_transcript(mem, scope); print(f"[exported] {path}")
        return

    if not HAS_SOUNDDEVICE:
        raise RuntimeError("Mic mode requires sounddevice + scipy.")
    print("Mic mode. Press ENTER to record; Ctrl+C to quit.")
    try:
        while True:
            input("\nPress ENTER to record...")
            p = record_from_mic(seconds=args.seconds, samplerate=args.samplerate)
            out = run_one(p)
            print("\n--- TRANSCRIPTION ---\n" + out.get("user_text",""))
            print("\n--- PATIENT REPLY ---\n" + out.get("assistant_text",""))
            print("\n--- STATE ---")
            print(json.dumps({"distress": out.get("distress"), "coverage": out.get("coverage")}, indent=2))
    except KeyboardInterrupt:
        print("\nExiting.")
        # if args.export_transcript:
        #     path=export_transcript(mem, scope); print(f"[exported] {path}")

if __name__ == "__main__":
    main()
