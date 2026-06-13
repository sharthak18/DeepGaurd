# DeepGuard 🛡️

> **Open-source deepfake & AI-generated media detector.**
> No GPU required — all AI inference is delegated to free external APIs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🎯 What it does

DeepGuard analyses images and videos and tells you whether they are **real** or
**AI-generated / deepfaked** using a **4-layer forensic pipeline**:

| Layer | Method | What it catches |
|-------|--------|----------------|
| 🔐 **C2PA Metadata** | Cryptographic signature scan | AI tools (DALL-E, Midjourney, Photoshop AI) that embed provenance data |
| 🔬 **ELA Forensics** | Error Level Analysis (local, no API) | Manual splicing, face-swaps, Photoshop edits |
| 🤖 **AI Ensemble** | Two HuggingFace open-source models | Modern AI-generated images & deepfake faces |
| ⚡ **Sightengine** | Enterprise API (optional) | Final high-precision tie-breaker; prevents false positives on real photos |

Your hardware runs only a lightweight Python script — no GPU, no downloaded model weights.

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/your-org/deepguard.git
cd deepguard
```

### 2. Create a virtual environment & install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Get your free API keys

**HuggingFace (Required)**
1. Sign up at [huggingface.co](https://huggingface.co/) — free
2. Go to **Settings → Access Tokens → New token (Read)**
3. Copy the token

**Sightengine (Strongly recommended — prevents false positives)**
1. Sign up at [sightengine.com](https://sightengine.com/) — free, 500 ops/month
2. Dashboard → copy **API User** and **API Secret**

### 4. Configure

```bash
cp .env.example .env
# Then edit .env and paste your keys:
```

```env
HF_TOKEN=hf_...

SIGHTENGINE_API_USER=...
SIGHTENGINE_API_SECRET=...
```

### 5. Install FFmpeg (required for video)

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## 🖥️ Web Interface (Streamlit)

The easiest way to use DeepGuard is through the drag-and-drop web UI:

```bash
~/.local/bin/streamlit run app.py
# Or if streamlit is on your PATH:
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

**Features:**
- Drag & drop image or video upload
- Live progress bar showing each forensic layer
- Colour-coded FAKE / REAL / UNCERTAIN verdict
- Per-model AI score breakdown with animated probability bars
- C2PA and ELA forensic layer status
- Per-frame breakdown for videos
- File metadata display

---

## ⌨️ Command-Line Interface

```bash
# Auto-detect file type and analyse
deepguard detect photo.jpg
deepguard detect clip.mp4

# Force a specific mode
deepguard detect --image  photo.jpg
deepguard detect --video  clip.mp4

# Verbose / debug — shows each forensic layer
deepguard detect photo.jpg --verbose

# Machine-readable JSON output
deepguard detect photo.jpg --json

# Save JSON report to file
deepguard detect photo.jpg --output report.json

# Show API status and model registry
deepguard info
```

### Example output

```
╔══════════════════════════════════════════════════════════════╗
║  🚨  DeepGuard Detection Report  🚨                          ║
╚══════════════════════════════════════════════════════════════╝

  File:    suspicious_face.jpg
  Type:    Image
  Verdict: FAKE
  Fake probability:    ████████████████░░░░  82.4%
  Ensemble confidence: 79.1%

  Model                              Label   Confidence
  ─────────────────────────────────────────────────────
  AI-vs-Deepfake-vs-Real-Siglip2     FAKE    99.5%
  Deep-Fake-Detector-v2-Model        FAKE    72.2%
  Sightengine (ensemble)             FAKE    88.0%

  Analysed at: 2026-05-14T17:13:18+00:00
```

---

## 📦 Project Structure

```
deepguard/
├── app.py                  # Streamlit web interface
├── deepguard/
│   ├── cli.py              # `deepguard` CLI entry point
│   ├── config.py           # API key loading (.env)
│   ├── ensemble.py         # Weighted score aggregation
│   ├── apis/
│   │   ├── huggingface.py  # HuggingFace Inference API client
│   │   └── sightengine.py  # Optional Sightengine client
│   ├── detectors/
│   │   ├── image_detector.py   # C2PA → ELA → AI ensemble pipeline
│   │   ├── video_detector.py   # FFmpeg frame extraction → image pipeline
│   │   └── audio_detector.py   # Disabled (free APIs unsupported)
│   └── utils/
│       ├── file_utils.py   # File validation & type detection
│       ├── video_utils.py  # FFmpeg helpers
│       └── report.py       # Rich terminal output & JSON export
├── tests/                  # Pytest unit tests (all API calls mocked)
├── .env.example
├── requirements.txt
├── pyproject.toml
├── REPORT.md               # Full project report & methodology
└── README.md
```

---

## 🤖 Models Used

| Model | Modality | License |
|-------|----------|---------|
| [`prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2`](https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2) | Image | Apache-2.0 |
| [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | Image | Apache-2.0 |
| [Sightengine Deepfake API](https://sightengine.com/docs/detect-ai-generated-images) | Image / Video | Commercial (free tier) |

---

## ⚙️ Configuration

All settings live in `.env`:

```env
# Required
HF_TOKEN=hf_...

# Optional but strongly recommended (prevents false positives on real media)
SIGHTENGINE_API_USER=...
SIGHTENGINE_API_SECRET=...

# Tuning
DEEPGUARD_VIDEO_MAX_FRAMES=10   # max frames to sample per video
DEEPGUARD_LOG_LEVEL=WARNING
```

> ⚠️ **Restart Streamlit or the CLI whenever you change `.env`** — the environment
> is loaded once at startup and cached for the session.

---

## 🚦 CLI Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| `0` | REAL — file appears genuine |
| `1` | FAKE — deepfake / AI-generated detected |
| `2` | UNCERTAIN — low confidence |

```bash
deepguard detect photo.jpg
if [ $? -eq 1 ]; then echo "Deepfake detected!"; fi
```

---

## ⚠️ Limitations

- Detection is **probabilistic**, not guaranteed — always apply human judgment
- Without Sightengine, open-source models have a **high false-positive rate on real photos** — strongly recommended to enable it
- HuggingFace free tier has **rate limits** — first call may take 20–30s (model cold-start)
- Video analysis samples **up to 10 frames** by default (set `DEEPGUARD_VIDEO_MAX_FRAMES`)
- **Audio deepfake detection is currently disabled** — HuggingFace free tier does not load audio classification models; Sightengine audio API is in development

---

## 🤝 Contributing

Contributions welcome!

- 🐛 **Bug reports** → [Issues](https://github.com/your-org/deepguard/issues)
- 💡 **Feature requests** → [Discussions](https://github.com/your-org/deepguard/discussions)
- 🔍 **Adding a new model** → edit `deepguard/config.py` and open a PR

---

## 📄 License

MIT © DeepGuard Contributors

This project is free and open-source. All AI models are open-source (Apache-2.0 / MIT).
Sightengine is an optional external service with its own terms of service.
