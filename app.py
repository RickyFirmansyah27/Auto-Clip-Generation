import streamlit as st
import queue
import threading
import os
import gc
import shutil
import signal
import sys
from logic import VideoProcessor

# === Graceful Exit Handler ===
def signal_handler(sig, frame):
    print("\n🛑 Shutting down immediately...")
    # Force exit without waiting for threads
    os._exit(0)

try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except ValueError:
    # Streamlit runs in a separate thread, so signal handlers might fail
    pass

# === Cleanup ===
def cleanup_folders():
    for folder in ['temp', 'hasil_shorts']:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except:
                pass
    os.makedirs('hasil_shorts', exist_ok=True)

# === Config ===
st.set_page_config(page_title="ClipGenAI", page_icon="✂️", layout="wide")

# === Theme ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {font-family: 'Inter', sans-serif; background: #0a0a0a;}
    #MainMenu, footer, header, .stDeployButton {display: none !important;}
    
    .title {text-align:center; font-size:1.6rem; font-weight:700; color:#00ff88; padding:0.8rem 0 0.3rem;}
    .subtitle {text-align:center; color:#555; font-size:0.8rem; margin-bottom:1rem;}
    
    .stTextInput > div > div > input {
        background: #111 !important;
        border: 1px solid #222 !important;
        border-radius: 6px !important;
        color: #fff !important;
    }
    .stTextInput > div > div > input:focus {border-color: #00ff88 !important;}
    .stSelectbox > div > div {background: #111 !important; border: 1px solid #222 !important;}
    
    .stTabs [data-baseweb="tab-list"] {background: transparent; border-bottom: 1px solid #222; border-radius: 0; padding: 0;}
    .stTabs [data-baseweb="tab"] {color: #666; font-size: 0.8rem; background: transparent !important; border:None !important;}
    .stTabs [aria-selected="true"] {color: #00ff88 !important; border-bottom: 2px solid #00ff88 !important; border-radius: 0 !important; background: transparent !important;}
    
    .stButton > button[kind="primary"] {
        background: #00ff88 !important;
        color: #000 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        color: #888 !important;
    }
    
    .stProgress > div > div > div {background: #00ff88 !important;}
    .stSlider > div > div > div > div {background: #00ff88 !important;}
    
    .log-box {
        background: #0d0d0d;
        border: 1px solid #1a1a1a;
        border-radius: 6px;
        padding: 0.5rem;
        font-family: monospace;
        font-size: 0.7rem;
        height: 150px;
        overflow-y: auto;
    }
    
    hr {border-color: #1a1a1a !important; margin: 0.75rem 0 !important;}
    
    .stTextInput > label, .stSelectbox > label, .stFileUploader > label, .stNumberInput > label {
        font-size: 0.7rem !important; color: #444 !important;
    }
</style>
""", unsafe_allow_html=True)

# === State ===
if 'logs' not in st.session_state: st.session_state.logs = []
if 'processing' not in st.session_state: st.session_state.processing = False
if 'completed_videos' not in st.session_state: st.session_state.completed_videos = []

output_dir = 'hasil_shorts'
os.makedirs(output_dir, exist_ok=True)

# === Header ===
st.markdown('<div class="title">✂️ Clipping GenAI</div><div class="subtitle">AI Shorts Generator</div>', unsafe_allow_html=True)

# === Layout ===
left, right = st.columns([1, 1], gap="medium")

with left:
    # API
    api_key = st.text_input("🔑 API Key", type="password", placeholder="Groq API Key")
    if not api_key: api_key = os.getenv("GROQ_API_KEY", "")
    
    st.markdown("---")
    
    # Source
    t1, t2 = st.tabs(["📹 YouTube", "📁 Upload"])
    with t1:
        youtube_url = st.text_input("URL", placeholder="https://youtube.com/watch?v=...")
        source_type = "YouTube"
    with t2:
        local_file = st.file_uploader("Video", type=['mp4', 'mov'])
        if local_file:
            source_type = "File"
    
    st.markdown("---")
    
    # Settings
    clip_count = st.slider("📊 Number of Clips", 1, 10, 5)
    
    with st.expander("🎨 Subtitles"):
        enable_subs = st.checkbox("Enable", value=True)
        if enable_subs:
            c1, c2 = st.columns(2)
            with c1:
                font_name = st.selectbox("Font", ["Impact", "Arial"])
                font_size = st.slider("Size", 50, 100, 70)
            with c2:
                font_color = st.color_picker("Color", "#00FF88")
                stroke_color = st.color_picker("Outline", "#000000")
            stroke_width = st.slider("Outline", 1, 6, 4)
            text_pos = st.slider("Position", 0.6, 0.85, 0.75)
        else:
            font_name, font_size, font_color = "Impact", 70, "#00FF88"
            stroke_color, stroke_width, text_pos = "#000000", 4, 0.75
    
st.markdown("---")

# Buttons
c1, c2 = st.columns([3, 1])
with c1:
    def enable_processing():
        st.session_state.processing = True

    st.button("⚡ Generate", type="primary", use_container_width=True, 
              disabled=st.session_state.processing, on_click=enable_processing)
with c2:
    def reset_state():
        st.session_state.processing = False
        cleanup_folders()
        st.session_state.logs = []
        gc.collect()

    st.button("🔄", use_container_width=True, help="Reset", on_click=reset_state)

# Right Column - Status & Logs
with right:
    st.markdown("**📊 Status & Logs**")
    progress = st.progress(0)
    status = st.empty()
    
    # Log Box (tall, scrollable, show all logs)
    log_placeholder = st.empty()
    def render_logs():
        # Taller log box (600px) with scroll, showing ALL logs without trimming
        html = '<div class="log-box" style="max-height: 600px; overflow-y: auto; min-height: 300px;">'
        # Show ALL logs, no limit
        for lvl, msg in st.session_state.logs:
            c = {"SUCCESS":"#00ff88","ERROR":"#ff4444","WARNING":"#ffaa00","INFO":"#00aaff"}.get(lvl,"#888")
            # Show FULL message, no character limit
            html += f'<div style="color:{c}; margin-bottom: 0.3rem; font-size: 0.85rem; line-height: 1.3;">• {msg}</div>'
        if not st.session_state.logs:
            html += '<div style="color:#555;text-align:center;padding:3rem;">🎬 Ready</div>'
        html += '</div>'
        log_placeholder.markdown(html, unsafe_allow_html=True)
    
    render_logs()



# === Video Gallery (Bottom) ===
st.markdown("---")
st.markdown("### 🎬 Generated Videos")
video_gallery_placeholder = st.empty()

def render_gallery():
    with video_gallery_placeholder.container():
        if os.path.exists(output_dir):
            files = sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
            if files:
                # Use chunks of 4 for better grid layout
                cols = st.columns(4)
                for i, f in enumerate(files):
                    file_path = os.path.join(output_dir, f)
                    with cols[i % 4]:
                        st.video(file_path)

# Initial render
render_gallery()

# === Process ===
if st.session_state.processing:
    # Validation
    valid = True
    if not api_key:
        st.toast("❌ Enter API Key")
        valid = False
    elif source_type == "YouTube" and not youtube_url:
        st.toast("❌ Enter URL")
        valid = False
    elif source_type == "File" and not local_file:
        st.toast("❌ Upload video")
        valid = False
    
    if not valid:
        st.session_state.processing = False
        st.rerun()
    
    else:
        if not st.session_state.logs: # Simple heuristic: if logs empty, probably fresh start
             cleanup_folders()
        
        # Live Video Callback - just notifies that a video is ready
        def on_video(path):
            # With parallel processing, videos are saved and will appear in gallery
            # Gallery is rendered automatically by checking the folder
            pass  # No action needed - render_gallery checks folder directly

        config = {
            'api_key': api_key,
            'source_type': 'youtube' if source_type == "YouTube" else 'file',
            'youtube_url': youtube_url,
            'local_file': "",
            'clip_count': clip_count,
            'auto_clip': False,
            'enable_subtitle': enable_subs,
            'font_name': font_name,
            'font_size': font_size,
            'font_color': font_color,
            'font_color_alt': "#FFFFFF",
            'stroke_color': stroke_color,
            'stroke_width': stroke_width,
            'text_position': text_pos,
            'output_dir': output_dir,
            'video_callback': on_video
        }
        
        if source_type == "File" and local_file:
            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/{local_file.name}"
            # Only write if not exists or similar? Or overwrite.
            with open(temp_path, "wb") as f:
                f.write(local_file.getbuffer())
            config['local_file'] = temp_path
        
        # Thread-safe queues for callbacks from worker threads
        log_queue = queue.Queue()
        progress_queue = queue.Queue()
        
        def log_cb(lvl, msg):
            # Safe to call from worker threads - just adds to queue
            log_queue.put((lvl, msg))
        
        def prog_cb(v, t):
            # Safe to call from worker threads - just adds to queue
            progress_queue.put((v, t))
        
        def process_queues():
            """Process all queued updates in the main thread"""
            # Process all pending logs
            while not log_queue.empty():
                try:
                    lvl, msg = log_queue.get_nowait()
                    st.session_state.logs.append((lvl, msg))
                except queue.Empty:
                    break
            render_logs()
            
            # Process latest progress update
            latest_progress = None
            while not progress_queue.empty():
                try:
                    latest_progress = progress_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_progress:
                v, t = latest_progress
                progress.progress(v)
                status.text(t)
        
        processor = VideoProcessor(log_callback=log_cb)
        
        # Start processing in background thread
        process_complete = threading.Event()
        process_error = [None]
        
        def process_wrapper():
            try:
                processor.process_video(config, progress_callback=prog_cb)
            except Exception as e:
                process_error[0] = e
            finally:
                process_complete.set()
        
        process_thread = threading.Thread(target=process_wrapper, daemon=True)
        process_thread.start()
        
        # Keep UI responsive and process queue updates
        import time
        while not process_complete.is_set():
            process_queues()  # Update UI from queue (safe in main thread)
            render_gallery()  # Update gallery
            time.sleep(0.5)  # Check twice per second
            
            # Allow early exit if user stops or refreshes page
            if not st.session_state.processing:
                break
        
        # Wait for thread to complete with timeout
        process_thread.join(timeout=2.0)
        
        # Final processing
        process_queues()
        render_gallery()
        
        try:
            if process_error[0]:
                raise process_error[0]
            
            st.balloons()
            st.success("🎉 All clips finished!")
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            st.session_state.processing = False
            st.rerun()
