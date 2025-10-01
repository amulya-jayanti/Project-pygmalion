# AI-ER-Patient-for-Healthcare-Professionals

## Overview
**AI-ER-Patient** is an AI-driven emergency patient simulator designed to train healthcare professionals in high-stress ER scenarios.  
It uses **conversational AI, avatars, and adaptive memory** to simulate realistic patient behavior and provide structured performance evaluations.

---

## Key Features
- **Dynamic Scenarios**: Heart attacks, burns, gunshot wounds, head trauma, overdoses, accidents, and more.  
- **Realistic Interactions**: Patients express pain, panic, and evolving symptoms in real time.  
- **Communication Training**: Focus on empathy, reassurance, and clarity.  
- **Adaptive Feedback**: Logs responses, evaluates decisions, and generates reports.  

---

## How It Works
1. **Speech-to-Text**: Whisper transcribes clinician input.  
2. **AI Response**: LLM generates realistic patient replies.  
3. **Avatar Output**: ElevenLabs + TAVUS/Audio2Face provide expressive voice and visuals.  
4. **Memory (Mem0)**: Tracks vitals, symptoms, and history for realistic continuity.  
5. **Evaluation**: Produces metrics on coverage, safety, and distress reduction.  

**Realtime Loop:**
```text
Mic → Whisper → Mem0.retrieve → LLM → Mem0.add → TTS → Avatar → Report
```
## Evaluation Metrics
- **Critical Coverage (60%)**: Key lifesaving questions asked.  
- **General Coverage (20%)**: Supporting questions included.  
- **Distress Reduction (20%)**: Ability to calm the patient.  

## Implementation Roadmap
- **Phase 0**: CLI + logging setup.  
- **Phase 1**: Core loop (Mic → Whisper → LLM).  
- **Phase 2**: Persona presets and distress tracking.  
- **Phase 3**: TTS + Avatar integration.  
- **Phase 4**: Metrics summary and reporting.  

## Tech Stack
- **STT**: Whisper  
- **Memory**: Mem0  
- **LLM**: OpenAI (gpt-4o-mini for low latency)  
- **TTS**: ElevenLabs / pyttsx3 (fallback)  
- **Avatar**: NVIDIA Audio2Face or TAVUS  
- **Evaluation**: Python scoring modules  

## Getting Started
1. **Clone repo**
   ```bash
   git clone https://github.com/your-username/AI-ER-Patient-for-Healthcare-Professionals.git
   cd AI-ER-Patient-for-Healthcare-Professionals
   ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
3. Add API keys in .env
    ```text
    OpenAI
    ElevenLabs
    TAVUS
    ```
4. Run simulation
    ```bash
    python main.py --mic --user_id=<id> --run_id=<id>


