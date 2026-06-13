"""
DeepGuard — Streamlit Web Interface
app.py

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# ── Ensure project root and its .venv are on PYTHONPATH ──────────────────────
# Streamlit may run from the system Python; inserting the project venv's
# site-packages ensures dotenv and all other project deps are importable.
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Automatically find and inject the venv site-packages
for _sp in sorted(_PROJECT_ROOT.glob(".venv/lib/python*/site-packages")):
    if str(_sp) not in sys.path:
        sys.path.insert(1, str(_sp))
        break

import streamlit as st

# ── Page config — MUST be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="DeepGuard · AI Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS — premium dark-glass UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: radial-gradient(ellipse at 20% 0%, #1a0e3d 0%, #0D0D1A 55%, #0a0a1a 100%);
    }

    /* ── Hero header ── */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #6C63FF 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #8b8bcc;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.2rem;
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(108,99,255,0.5);
    }

    /* ── Verdict badges ── */
    .verdict-fake {
        display: inline-block;
        background: linear-gradient(135deg, #ff4b4b, #c0392b);
        color: white;
        font-size: 2rem;
        font-weight: 800;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        letter-spacing: 0.12em;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 0 20px rgba(255,75,75,0.5);
        box-shadow: 0 0 30px rgba(255,75,75,0.25);
    }
    .verdict-real {
        display: inline-block;
        background: linear-gradient(135deg, #00d084, #006644);
        color: white;
        font-size: 2rem;
        font-weight: 800;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        letter-spacing: 0.12em;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 0 20px rgba(0,208,132,0.5);
        box-shadow: 0 0 30px rgba(0,208,132,0.25);
    }
    .verdict-uncertain {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b, #b45309);
        color: white;
        font-size: 2rem;
        font-weight: 800;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        letter-spacing: 0.12em;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 0 20px rgba(245,158,11,0.5);
        box-shadow: 0 0 30px rgba(245,158,11,0.25);
    }

    /* ── Probability bar ── */
    .prob-bar-track {
        background: rgba(255,255,255,0.07);
        border-radius: 100px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        margin: 0.4rem 0;
    }
    .prob-bar-fill-fake {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #ff4b4b, #ff8c42);
        transition: width 1s ease;
        box-shadow: 0 0 10px rgba(255,75,75,0.5);
    }
    .prob-bar-fill-real {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #00d084, #38bdf8);
        transition: width 1s ease;
        box-shadow: 0 0 10px rgba(0,208,132,0.5);
    }
    .prob-bar-fill-uncertain {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
        transition: width 1s ease;
    }

    /* ── Layer badges ── */
    .layer-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(108,99,255,0.15);
        border: 1px solid rgba(108,99,255,0.35);
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-size: 0.82rem;
        color: #a78bfa;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }

    /* ── Model row ── */
    .model-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.7rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .model-row:last-child { border-bottom: none; }
    .model-name { color: #a0a0cc; font-size: 0.9rem; }
    .model-label-fake { color: #ff4b4b; font-weight: 600; font-size: 0.9rem; }
    .model-label-real { color: #00d084; font-weight: 600; font-size: 0.9rem; }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #c4b5fd;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(108,99,255,0.4) !important;
        border-radius: 16px !important;
        background: rgba(108,99,255,0.05) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(108,99,255,0.8) !important;
        background: rgba(108,99,255,0.1) !important;
    }

    /* ── Streamlit override cleanups ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #38bdf8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(108,99,255,0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(108,99,255,0.5);
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper renderers
# ══════════════════════════════════════════════════════════════════════════════

def _prob_bar(prob: float, verdict: str) -> str:
    pct = int(prob * 100)
    css_class = {
        "FAKE": "prob-bar-fill-fake",
        "REAL": "prob-bar-fill-real",
    }.get(verdict, "prob-bar-fill-uncertain")
    return (
        f'<div class="prob-bar-track">'
        f'<div class="{css_class}" style="width:{pct}%"></div>'
        f'</div>'
    )


def _verdict_badge(verdict: str) -> str:
    icons = {"FAKE": "🚨", "REAL": "✅", "UNCERTAIN": "⚠️"}
    css = {"FAKE": "verdict-fake", "REAL": "verdict-real", "UNCERTAIN": "verdict-uncertain"}
    icon = icons.get(verdict, "❓")
    css_cls = css.get(verdict, "verdict-uncertain")
    return f'<span class="{css_cls}">{icon} {verdict}</span>'


def _render_image_result(result, original_filename: str = "") -> None:
    """Render a full forensic report card for an image DetectionResult."""
    v = result.verdict
    prob_pct = result.fake_probability * 100
    conf_pct = result.confidence * 100

    # ── Verdict header ────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;padding:1rem 0;">'
        f'{_verdict_badge(v)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**🎯 Fake Probability**")
        st.markdown(_prob_bar(result.fake_probability, v), unsafe_allow_html=True)
        color = "#ff4b4b" if v == "FAKE" else "#00d084" if v == "REAL" else "#f59e0b"
        st.markdown(
            f'<p style="font-size:2rem;font-weight:800;color:{color};margin:0">'
            f'{prob_pct:.1f}%</p>',
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(f"**📊 Ensemble Confidence**")
        st.markdown(_prob_bar(result.confidence, "REAL"), unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:2rem;font-weight:800;color:#a78bfa;margin:0">'
            f'{conf_pct:.1f}%</p>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Forensic Layers ───────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Forensic Layer Breakdown</div>', unsafe_allow_html=True)

    meta = result.metadata or {}

    # C2PA
    c2pa_result = meta.get("c2pa_result", "not_checked")
    c2pa_icon = "🔐" if c2pa_result == "clean" else ("🚨" if c2pa_result == "ai_signed" else "➖")
    c2pa_label = {
        "clean": "No AI signatures found",
        "ai_signed": "AI-generated signature detected!",
        "unavailable": "C2PA library unavailable",
        "not_checked": "Not checked",
    }.get(c2pa_result, c2pa_result)
    st.markdown(
        f'<span class="layer-badge">{c2pa_icon} C2PA Metadata: <strong style="margin-left:4px">{c2pa_label}</strong></span><br/>',
        unsafe_allow_html=True,
    )

    # ELA
    ela_score = meta.get("ela_probability", None)
    if ela_score is not None:
        ela_pct = ela_score * 100
        ela_icon = "🟥" if ela_score > 0.6 else ("🟨" if ela_score > 0.35 else "🟩")
        st.markdown(
            f'<span class="layer-badge">{ela_icon} Error Level Analysis (ELA): <strong style="margin-left:4px">{ela_pct:.1f}% manipulation signal</strong></span><br/>',
            unsafe_allow_html=True,
        )

    # Sightengine
    if result.sightengine_score is not None:
        se_icon = "🚨" if result.sightengine_score > 0.5 else "✅"
        st.markdown(
            f'<span class="layer-badge">{se_icon} Sightengine Enterprise: <strong style="margin-left:4px">{result.sightengine_score*100:.1f}% fake probability</strong></span>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-model scores ──────────────────────────────────────────────────
    if result.model_scores:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 AI Model Ensemble Scores</div>', unsafe_allow_html=True)
        for ms in result.model_scores:
            label_cls = "model-label-fake" if ms.label == "fake" else "model-label-real"
            short_id = ms.model_id.split("/")[-1]
            conf_bar = _prob_bar(ms.confidence, "FAKE" if ms.label == "fake" else "REAL")
            st.markdown(
                f'<div class="model-row">'
                f'<span class="model-name">🔷 {short_id}</span>'
                f'<span class="{label_cls}">{ms.label.upper()} &nbsp;{ms.confidence*100:.1f}%</span>'
                f'</div>{conf_bar}',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── File info ─────────────────────────────────────────────────────────
    with st.expander("📂 File Metadata"):
        size_bytes = meta.get("file_size_bytes", 0)
        size_kb = size_bytes / 1024
        display_name = original_filename or meta.get('filename', 'N/A')
        st.markdown(f"- **Filename:** `{display_name}`")
        st.markdown(f"- **File size:** `{size_kb:.1f} KB`")
        st.markdown(f"- **Analysed at:** `{result.timestamp}`")
        if "error" in meta:
            st.error(f"Note: {meta['error']}")


def _render_video_result(result, original_filename: str = "") -> None:
    """Render a full forensic report card for a VideoDetectionResult."""
    v = result.verdict
    prob_pct = result.fake_probability * 100

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;padding:1rem 0;">'
        f'{_verdict_badge(v)}'
        f'</div>',
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**🎯 Fake Probability**")
        st.markdown(_prob_bar(result.fake_probability, v), unsafe_allow_html=True)
        color = "#ff4b4b" if v == "FAKE" else "#00d084" if v == "REAL" else "#f59e0b"
        st.markdown(
            f'<p style="font-size:2rem;font-weight:800;color:{color};margin:0">{prob_pct:.1f}%</p>',
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown("**🎞️ Frames Analysed**")
        st.markdown(
            f'<p style="font-size:2rem;font-weight:800;color:#a78bfa;margin:0">{result.frames_analysed}</p>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-frame breakdown ───────────────────────────────────────────────
    if result.frame_results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎞️ Per-Frame Breakdown</div>', unsafe_allow_html=True)
        for fr in result.frame_results:
            r = fr.result
            fc = "#ff4b4b" if r.verdict == "FAKE" else "#00d084" if r.verdict == "REAL" else "#f59e0b"
            bar = _prob_bar(r.fake_probability, r.verdict)
            st.markdown(
                f'<div class="model-row">'
                f'<span class="model-name">Frame {fr.frame_index}</span>'
                f'<span style="color:{fc};font-weight:600">{r.verdict} &nbsp;{r.fake_probability*100:.1f}%</span>'
                f'</div>{bar}',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📂 File Metadata"):
        meta = result.metadata or {}
        display_name = original_filename or meta.get('filename', 'N/A')
        st.markdown(f"- **Filename:** `{display_name}`")
        dur = meta.get('duration_seconds')
        if dur:
            st.markdown(f"- **Duration:** `{dur:.1f}s`")
        size_bytes = meta.get('file_size_bytes', 0)
        st.markdown(f"- **File size:** `{size_bytes/1024/1024:.2f} MB`")
        st.markdown(f"- **Analysed at:** `{result.timestamp}`")


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Hero Section ──────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="hero-title">🛡️ DeepGuard</h1>'
        '<p class="hero-subtitle">Open-source deepfake & AI-generated media detector &nbsp;·&nbsp; '
        'Images &amp; Videos &nbsp;·&nbsp; 4-Layer Forensic Pipeline</p>',
        unsafe_allow_html=True,
    )

    # ── How it works pills ────────────────────────────────────────────────
    cols = st.columns(4)
    pills = [
        ("🔐", "C2PA Metadata", "Cryptographic AI signature check"),
        ("🔬", "ELA Forensics", "Pixel-level manipulation analysis"),
        ("🤖", "AI Ensemble", "Open-source HuggingFace models"),
        ("⚡", "Sightengine", "Enterprise-grade AI verification"),
    ]
    for col, (icon, title, desc) in zip(cols, pills):
        col.markdown(
            f'<div class="glass-card" style="text-align:center;padding:1rem">'
            f'<div style="font-size:2rem">{icon}</div>'
            f'<div style="font-weight:700;color:#c4b5fd;margin:0.4rem 0">{title}</div>'
            f'<div style="font-size:0.8rem;color:#6b6b9e">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Media to Analyse")
    uploaded = st.file_uploader(
        "Drag & drop an image or video file here, or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp", "gif",
              "mp4", "avi", "mov", "mkv", "webm", "flv"],
        key="main_uploader",
        label_visibility="visible",
    )

    if uploaded is None:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3rem;color:#4a4a7a">'
            '<div style="font-size:3.5rem">🔍</div>'
            '<div style="font-size:1.1rem;margin-top:0.8rem">No file uploaded yet</div>'
            '<div style="font-size:0.85rem;margin-top:0.4rem">Supports JPG, PNG, WEBP, MP4, AVI, MKV and more</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Detect media type ─────────────────────────────────────────────────
    fname = uploaded.name.lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    suffix = Path(fname).suffix
    is_video = suffix in video_exts

    # ── Preview + Analyse button ──────────────────────────────────────────
    col_preview, col_controls = st.columns([1, 1])
    with col_preview:
        if not is_video:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
        else:
            st.video(uploaded)

    with col_controls:
        st.markdown(
            f'<div class="glass-card">'
            f'<div style="font-size:1.1rem;font-weight:600;color:#e0e0ff">📄 {uploaded.name}</div>'
            f'<div style="font-size:0.85rem;color:#6b6b9e;margin-top:0.3rem">'
            f'{"Video" if is_video else "Image"} · {uploaded.size / 1024:.1f} KB</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        analyse_btn = st.button("🚀 Run DeepGuard Analysis", use_container_width=True)

    # ── Run analysis ──────────────────────────────────────────────────────
    if analyse_btn:
        # Save upload to a temp file
        suffix_str = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_str) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        result_placeholder = st.empty()

        with st.spinner("🔬 Running forensic pipeline — this may take 30–60s…"):
            progress = st.progress(0, text="Initialising…")
            try:
                # Lazy imports so errors surface cleanly in the UI
                from deepguard import config  # noqa — triggers validation

                if not is_video:
                    progress.progress(10, text="🔐 Checking cryptographic metadata (C2PA)…")
                    time.sleep(0.3)
                    from deepguard.detectors import image_detector
                    progress.progress(30, text="🔬 Running Error Level Analysis (ELA)…")
                    time.sleep(0.3)
                    progress.progress(55, text="🤖 Querying AI model ensemble…")
                    result = image_detector.detect(tmp_path)
                    progress.progress(90, text="⚡ Aggregating scores…")
                    time.sleep(0.3)
                else:
                    progress.progress(10, text="🎞️ Extracting video frames with FFmpeg…")
                    time.sleep(0.3)
                    from deepguard.detectors import video_detector
                    progress.progress(30, text="🤖 Running frame-by-frame AI analysis…")
                    result = video_detector.detect(tmp_path)
                    progress.progress(90, text="📊 Aggregating frame verdicts…")
                    time.sleep(0.3)

                progress.progress(100, text="✅ Analysis complete!")
                time.sleep(0.5)
                progress.empty()

                st.markdown("---")
                st.markdown("## 🧾 Forensic Report")

                if not is_video:
                    _render_image_result(result, original_filename=uploaded.name)
                else:
                    _render_video_result(result, original_filename=uploaded.name)

            except Exception as exc:
                progress.empty()
                st.error(f"❌ Analysis failed: {exc}")
                with st.expander("Show technical details"):
                    import traceback
                    st.code(traceback.format_exc())
            finally:
                tmp_path.unlink(missing_ok=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#3a3a6a;font-size:0.8rem;padding:1rem 0">'
        '🛡️ DeepGuard &nbsp;·&nbsp; Open-source deepfake detection &nbsp;·&nbsp; '
        'Powered by HuggingFace, Sightengine, C2PA & ELA forensics'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
