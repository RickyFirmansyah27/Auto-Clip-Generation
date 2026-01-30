import streamlit as st
import queue
import threading
import os
import gc
import shutil
import signal
import sys
from dotenv import load_dotenv

load_dotenv()

from logic import VideoProcessor

def signal_handler(sig, frame):
    print("\n🛑 Shutting down immediately...")
    os._exit(0)

try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except ValueError:
    pass

def cleanup_folders():
    for folder in ['temp', 'hasil_shorts']:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except:
                pass
    os.makedirs('hasil_shorts', exist_ok=True)

st.set_page_config(page_title="ClipGenAI", page_icon="✂️", layout="wide")

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
    
    /* Cancel button - red style */
    div[data-testid="column"]:last-child .stButton > button[kind="primary"] {
        background: #ff4444 !important;
        color: #fff !important;
    }
    div[data-testid="column"]:last-child .stButton > button[kind="primary"]:hover {
        background: #ff6666 !important;
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

if 'logs' not in st.session_state: st.session_state.logs = []
if 'processing' not in st.session_state: st.session_state.processing = False
if 'completed_videos' not in st.session_state: st.session_state.completed_videos = []

output_dir = 'hasil_shorts'
os.makedirs(output_dir, exist_ok=True)

st.markdown('<div class="title">✂️ Clipping GenAI</div><div class="subtitle">AI Shorts Generator</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="medium")

with left:
    api_key = st.text_input("🔑 API Key", type="password", placeholder="Groq API Key")
    if not api_key: api_key = os.getenv("GROQ_API_KEY", "")
    
    st.markdown("---")
    
    t1, t2 = st.tabs(["📹 YouTube", "📁 Upload"])
    with t1:
        youtube_url = st.text_input("URL", placeholder="https://youtube.com/watch?v=...")
        source_type = "YouTube"
    with t2:
        local_file = st.file_uploader("Video", type=['mp4', 'mov'])
        if local_file:
            source_type = "File"
    
    st.markdown("---")
    
    clip_count = st.slider("📊 Number of Clips", 1, 10, 5)
    
    with st.expander("🎨 Subtitles"):
        enable_subs = st.checkbox("Enable", value=True)
        if enable_subs:
            font_dir = "fonts"
            if not os.path.exists(font_dir):
                os.makedirs(font_dir)
            
            local_fonts = [f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf'))]
            default_fonts = ["Impact", "Arial", "Verdana", "Comic Sans MS"]
            available_fonts = local_fonts + default_fonts
            
            default_idx = 0
            for i, f in enumerate(available_fonts):
                if "bold" in f.lower() or "montserrat" in f.lower():
                    default_idx = i
                    break
            
            c1, c2 = st.columns(2)
            with c1:
                font_name = st.selectbox("Font", available_fonts, index=default_idx)
                font_size = st.slider("Size", 40, 120, 70)
            with c2:
                font_color = st.color_picker("Color", "#FFFF00")
                stroke_color = st.color_picker("Outline", "#000000")
            stroke_width = st.slider("Outline", 1, 6, 4)
            text_pos = st.slider("Position", 0.6, 0.85, 0.75)
            
            if font_name in local_fonts:
                font_name = os.path.join(font_dir, font_name)
        else:
            font_name, font_size, font_color = "Impact", 70, "#FFFF00"
            stroke_color, stroke_width, text_pos = "#000000", 4, 0.75
    
    st.markdown("---")
    
    if 'cancelling' not in st.session_state:
        st.session_state.cancelling = False
    
    c1, c2 = st.columns([3, 1])
    with c1:
        def enable_processing():
            if 'processor' in st.session_state and st.session_state.processor:
                st.session_state.processor.stop_processing()
            
            if 'worker_thread' in st.session_state:
                st.session_state.worker_thread = None
            if 'processor' in st.session_state:
                st.session_state.processor = None
            if 'process_complete' in st.session_state:
                st.session_state.process_complete.clear()
            if 'processing_done' in st.session_state:
                del st.session_state.processing_done
            if 'process_error' in st.session_state:
                st.session_state.process_error = [None]
            
            st.session_state.logs = []
            st.session_state.completed_videos = []
            
            for folder in ['temp']:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                    except:
                        pass
            os.makedirs('temp', exist_ok=True)
            os.makedirs('hasil_shorts', exist_ok=True)
            
            gc.collect()
            
            st.session_state.processing = True
            st.session_state.cancelling = False

        st.button("⚡ Generate", type="primary", use_container_width=True, 
                  disabled=st.session_state.processing, on_click=enable_processing)
    with c2:
        def cancel_processing():
            st.session_state.cancelling = True
            if 'processor' in st.session_state and st.session_state.processor:
                st.session_state.processor.stop_processing()
            
            st.session_state.processing = False
            st.session_state.logs.append(("WARNING", "🛑 Cancelling all processes..."))
            
            if 'worker_thread' in st.session_state:
                st.session_state.worker_thread = None
            if 'processor' in st.session_state:
                st.session_state.processor = None
            if 'process_complete' in st.session_state:
                st.session_state.process_complete.set()
            
            gc.collect()
            st.session_state.cancelling = False
        
        def reset_all():
            st.session_state.processing = False
            st.session_state.cancelling = False
            st.session_state.logs = []
            st.session_state.completed_videos = []
            
            if 'worker_thread' in st.session_state:
                st.session_state.worker_thread = None
            if 'processor' in st.session_state:
                st.session_state.processor = None
            if 'process_complete' in st.session_state:
                st.session_state.process_complete.clear()
            if 'processing_done' in st.session_state:
                del st.session_state.processing_done
            
            for folder in ['temp', 'hasil_shorts']:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                    except Exception as e:
                        print(f"Error cleaning {folder}: {e}")
            os.makedirs('hasil_shorts', exist_ok=True)
            
            gc.collect()

        if st.session_state.processing:
            st.button("❌ Cancel", use_container_width=True, on_click=cancel_processing, 
                      help="Stop all processing", type="primary")
        else:
            st.button("🔄 Reset", use_container_width=True, on_click=reset_all,
                      help="Reset all settings and clear files")

with right:
    st.markdown("**📊 Status & Logs**")
    progress = st.progress(0)
    status = st.empty()
    
    log_placeholder = st.empty()
    def render_logs():
        html = '<div class="log-box" style="max-height: 450px; overflow-y: auto; min-height: 400px;">'
        for lvl, msg in st.session_state.logs:
            c = {"SUCCESS":"#00ff88","ERROR":"#ff4444","WARNING":"#ffaa00","INFO":"#00aaff"}.get(lvl,"#888")
            html += f'<div style="color:{c}; margin-bottom: 0.3rem; font-size: 0.85rem; line-height: 1.3;">• {msg}</div>'
        if not st.session_state.logs:
            html += '<div style="color:#555;text-align:center;padding:3rem;">🎬 Ready</div>'
        html += '</div>'
        log_placeholder.markdown(html, unsafe_allow_html=True)
    
    render_logs()

st.markdown("---")
st.markdown("### 🎬 Generated Videos")
video_gallery_placeholder = st.empty()

def render_gallery():
    with video_gallery_placeholder.container():
        if os.path.exists(output_dir):
            files = sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
            if files:
                cols = st.columns(4)
                for i, f in enumerate(files):
                    file_path = os.path.join(output_dir, f)
                    with cols[i % 4]:
                        st.video(file_path)

render_gallery()

if st.session_state.processing:
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
        # Initialize session state variables for the worker if they don't exist
        if 'worker_thread' not in st.session_state:
            st.session_state.worker_thread = None
        if 'processor' not in st.session_state:
            st.session_state.processor = None
        if 'log_queue' not in st.session_state:
            st.session_state.log_queue = queue.Queue()
        if 'progress_queue' not in st.session_state:
            st.session_state.progress_queue = queue.Queue()
        if 'process_complete' not in st.session_state:
            st.session_state.process_complete = threading.Event()
        if 'process_error' not in st.session_state:
            st.session_state.process_error = [None]  # Use list to be mutable

        # Clear previous run data only if starting fresh (thread is None)
        if st.session_state.worker_thread is None:
             if not st.session_state.logs:
                 cleanup_folders()
             # Reset events and queues for new run
             st.session_state.process_complete.clear()
             st.session_state.process_error[0] = None
             # Re-create queues to ensure they are fresh
             st.session_state.log_queue = queue.Queue()
             st.session_state.progress_queue = queue.Queue()

        def on_video(path):
            pass

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
            # Only save file if not already running to avoid overwriting/locking
            if st.session_state.worker_thread is None:
                temp_path = f"temp/{local_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(local_file.getbuffer())
                config['local_file'] = temp_path
            else:
                 # Reconstruct path for existing run
                 config['local_file'] = f"temp/{local_file.name}"
        
        
        # Capture session state objects to local variables for thread safety
        # The thread cannot access st.session_state directly efficiently without context
        log_queue = st.session_state.log_queue
        progress_queue = st.session_state.progress_queue
        process_complete = st.session_state.process_complete
        process_error = st.session_state.process_error

        # Callbacks that put data into the queues (using local references)
        def log_cb(lvl, msg):
            log_queue.put((lvl, msg))
        
        def prog_cb(v, t):
            progress_queue.put((v, t))
        
        def process_queues():
            # Process Log Queue
            while not st.session_state.log_queue.empty():
                try:
                    lvl, msg = st.session_state.log_queue.get_nowait()
                    st.session_state.logs.append((lvl, msg))
                except queue.Empty:
                    break
            render_logs()
            
            # Process Progress Queue
            latest_progress = None
            while not st.session_state.progress_queue.empty():
                try:
                    latest_progress = st.session_state.progress_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_progress:
                v, t = latest_progress
                progress.progress(v)
                status.text(t)
        
        # Only start a new thread if one isn't already running
        if st.session_state.worker_thread is None or not st.session_state.worker_thread.is_alive():
            # Double check completion event to avoid restarting a finished job if state lingered
            if not st.session_state.process_complete.is_set():
                processor = VideoProcessor(log_callback=log_cb)
                st.session_state.processor = processor # Store for cancellation
                
                def process_wrapper():
                    try:
                        processor.process_video(config, progress_callback=prog_cb)
                    except Exception as e:
                        process_error[0] = e
                    finally:
                        process_complete.set()
                
                t = threading.Thread(target=process_wrapper, daemon=True)
                st.session_state.worker_thread = t
                t.start()
        
        import time
        # Monitor Loop
        # We loop here to update the UI while the thread runs.
        # Streamlit will re-run the script if user interaction happens, returning us here.
        # If we just let it fall through, the UI might stale until a re-run.
        # But `st.timer` or `st.empty` loops are common patterns.
        # We'll use a loop with sleep but break if thread finishes.
        
        while not st.session_state.process_complete.is_set():
            process_queues()
            render_gallery()
            time.sleep(0.5)
            # Rerun explicitly to keep UI fresh if needed, but sleep prevents busy wait.
            # However, a hard rerun inside loop resets the script. 
            # We want to loop and update placeholders.
            
            if not st.session_state.processing: # User cancelled via reset
                 break
        
        # Final update after loop break
        process_queues()
        render_gallery()
        
        # Check if done
        if st.session_state.process_complete.is_set():
            try:
                if st.session_state.process_error[0]:
                    raise st.session_state.process_error[0]
                
                # Only show success once
                if "processing_done" not in st.session_state:
                     st.balloons()
                     st.success("🎉 All clips finished!")
                     st.session_state.processing_done = True # Prevent repeat animations
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                st.session_state.processing = False
                st.session_state.worker_thread = None
                st.session_state.processor = None
                st.rerun()

