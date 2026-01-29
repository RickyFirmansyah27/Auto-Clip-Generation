"""
AI Auto Shorts - GUI Application
Modern GUI untuk membuat short-form content dari video YouTube
"""

import os
import sys
import json
import threading
import queue
import customtkinter as ctk
from tkinter import filedialog, colorchooser
import cv2
import numpy as np
import mediapipe as mp
import whisper
import yt_dlp
import torch
import shutil
from groq import Groq
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class LogRedirector:
    """Redirect print output to GUI log"""
    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, text):
        if text.strip():
            self.log_queue.put(("INFO", text.strip()))

    def flush(self):
        pass

class AIAutoShortsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🎬 AI Auto Shorts")
        self.geometry("900x750")
        self.minsize(800, 700)

        # Queue untuk log messages
        self.log_queue = queue.Queue()
        self.is_processing = False
        self.cancel_flag = False

        # Default values
        self.default_config = {
            'font_size': 70,
            'font_color': '#FFD700',
            'font_color_alt': '#FFFFFF',
            'stroke_color': '#000000',
            'stroke_width': 3,
            'text_position': 0.75,
            'output_dir': os.path.join(os.getcwd(), 'hasil_shorts')
        }

        self.create_widgets()
        self.check_log_queue()

    def create_widgets(self):
        # Main container with scrollable frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # === HEADER ===
        header = ctk.CTkLabel(
            self.main_frame,
            text="🎬 AI Auto Shorts Generator",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        header.pack(pady=(10, 5))

        subtitle = ctk.CTkLabel(
            self.main_frame,
            text="Convert YouTube videos to viral TikTok/Shorts clips",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle.pack(pady=(0, 15))

        # === INPUT SECTION ===
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.pack(fill="x", padx=10, pady=5)

        input_label = ctk.CTkLabel(
            input_frame,
            text="📝 Input Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        input_label.pack(anchor="w", padx=10, pady=(10, 5))

        # API Key
        api_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        api_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(api_frame, text="Groq API Key:", width=120, anchor="w").pack(side="left")
        self.api_key_var = ctk.StringVar()
        self.api_key_entry = ctk.CTkEntry(api_frame, textvariable=self.api_key_var, show="*", width=400)
        self.api_key_entry.pack(side="left", padx=5, fill="x", expand=True)

        self.show_key_btn = ctk.CTkButton(api_frame, text="👁", width=40, command=self.toggle_api_key)
        self.show_key_btn.pack(side="left", padx=5)

        # Load from .env if exists
        env_key = os.getenv("GROQ_API_KEY", "")
        if env_key:
            self.api_key_var.set(env_key)

        # Video Source Selection
        source_label = ctk.CTkLabel(
            input_frame,
            text="📹 Video Source",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        source_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Source type radio buttons
        source_type_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        source_type_frame.pack(fill="x", padx=10, pady=5)

        self.source_type_var = ctk.StringVar(value="youtube")

        self.youtube_radio = ctk.CTkRadioButton(
            source_type_frame,
            text="YouTube URL",
            variable=self.source_type_var,
            value="youtube",
            command=self.toggle_source_type
        )
        self.youtube_radio.pack(side="left", padx=(0, 20))

        self.file_radio = ctk.CTkRadioButton(
            source_type_frame,
            text="Local File",
            variable=self.source_type_var,
            value="file",
            command=self.toggle_source_type
        )
        self.file_radio.pack(side="left")

        # YouTube URL frame
        self.url_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        self.url_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.url_frame, text="URL:", width=80, anchor="w").pack(side="left")
        self.url_var = ctk.StringVar()
        self.url_entry = ctk.CTkEntry(self.url_frame, textvariable=self.url_var, width=400, placeholder_text="https://www.youtube.com/watch?v=...")
        self.url_entry.pack(side="left", padx=5, fill="x", expand=True)

        # Local file frame (initially hidden)
        self.file_frame = ctk.CTkFrame(input_frame, fg_color="transparent")

        ctk.CTkLabel(self.file_frame, text="File:", width=80, anchor="w").pack(side="left")
        self.file_path_var = ctk.StringVar()
        self.file_entry = ctk.CTkEntry(self.file_frame, textvariable=self.file_path_var, width=350, placeholder_text="Select video file...")
        self.file_entry.pack(side="left", padx=5, fill="x", expand=True)

        self.browse_file_btn = ctk.CTkButton(self.file_frame, text="📂 Browse", width=100, command=self.browse_video_file)
        self.browse_file_btn.pack(side="left", padx=5)

        # Clip Count
        clip_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        clip_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(clip_frame, text="Jumlah Klip:", width=120, anchor="w").pack(side="left")

        # Auto clip checkbox
        self.auto_clip_var = ctk.BooleanVar(value=False)
        self.auto_clip_checkbox = ctk.CTkCheckBox(
            clip_frame,
            text="Auto",
            variable=self.auto_clip_var,
            command=self.toggle_auto_clip,
            width=70
        )
        self.auto_clip_checkbox.pack(side="left", padx=(0, 10))

        # Slider frame (can be hidden)
        self.clip_slider_frame = ctk.CTkFrame(clip_frame, fg_color="transparent")
        self.clip_slider_frame.pack(side="left", fill="x", expand=True)

        self.clip_count_var = ctk.IntVar(value=5)
        self.clip_slider = ctk.CTkSlider(
            self.clip_slider_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            variable=self.clip_count_var,
            width=250,
            command=self.update_clip_label
        )
        self.clip_slider.pack(side="left", padx=5)
        self.clip_count_label = ctk.CTkLabel(self.clip_slider_frame, text="5 klip", width=60)
        self.clip_count_label.pack(side="left")

        # Auto info label
        self.auto_clip_info = ctk.CTkLabel(
            clip_frame,
            text="(1 klip per 5 menit video)",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        # Initially hidden, shown when auto is checked

        # === SUBTITLE CONFIGURATION ===
        subtitle_frame = ctk.CTkFrame(self.main_frame)
        subtitle_frame.pack(fill="x", padx=10, pady=5)

        subtitle_label = ctk.CTkLabel(
            subtitle_frame,
            text="🔤 Subtitle Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        subtitle_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Enable/Disable subtitle checkbox
        enable_sub_frame = ctk.CTkFrame(subtitle_frame, fg_color="transparent")
        enable_sub_frame.pack(fill="x", padx=10, pady=5)

        self.enable_subtitle_var = ctk.BooleanVar(value=True)
        self.enable_subtitle_cb = ctk.CTkCheckBox(
            enable_sub_frame,
            text="Enable Subtitles",
            variable=self.enable_subtitle_var,
            command=self.toggle_subtitle_options
        )
        self.enable_subtitle_cb.pack(side="left")

        # Container for subtitle options (to show/hide)
        self.subtitle_options_frame = ctk.CTkFrame(subtitle_frame, fg_color="transparent")

        # Font Size
        font_size_frame = ctk.CTkFrame(subtitle_frame, fg_color="transparent")
        font_size_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(font_size_frame, text="Font Size:", width=120, anchor="w").pack(side="left")
        self.font_size_var = ctk.IntVar(value=70)
        self.font_size_slider = ctk.CTkSlider(font_size_frame, from_=30, to=120, number_of_steps=90, variable=self.font_size_var, width=200, command=self.update_font_size_label)
        self.font_size_slider.pack(side="left", padx=5)
        self.font_size_label = ctk.CTkLabel(font_size_frame, text="70px", width=60)
        self.font_size_label.pack(side="left")

        # Font Colors (in one row)
        colors_frame = ctk.CTkFrame(subtitle_frame, fg_color="transparent")
        colors_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(colors_frame, text="Main Color:", width=80, anchor="w").pack(side="left")
        self.font_color_var = ctk.StringVar(value=self.default_config['font_color'])
        self.font_color_btn = ctk.CTkButton(colors_frame, text="", width=40, height=25, fg_color=self.default_config['font_color'], command=lambda: self.pick_color('font'))
        self.font_color_btn.pack(side="left", padx=5)

        ctk.CTkLabel(colors_frame, text="Alt Color:", width=70, anchor="w").pack(side="left", padx=(20, 0))
        self.font_color_alt_var = ctk.StringVar(value=self.default_config['font_color_alt'])
        self.font_color_alt_btn = ctk.CTkButton(colors_frame, text="", width=40, height=25, fg_color=self.default_config['font_color_alt'], command=lambda: self.pick_color('font_alt'))
        self.font_color_alt_btn.pack(side="left", padx=5)

        ctk.CTkLabel(colors_frame, text="Stroke:", width=60, anchor="w").pack(side="left", padx=(20, 0))
        self.stroke_color_var = ctk.StringVar(value=self.default_config['stroke_color'])
        self.stroke_color_btn = ctk.CTkButton(colors_frame, text="", width=40, height=25, fg_color=self.default_config['stroke_color'], command=lambda: self.pick_color('stroke'))
        self.stroke_color_btn.pack(side="left", padx=5)

        # Stroke Width & Position
        pos_frame = ctk.CTkFrame(subtitle_frame, fg_color="transparent")
        pos_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(pos_frame, text="Stroke Width:", width=100, anchor="w").pack(side="left")
        self.stroke_width_var = ctk.IntVar(value=3)
        self.stroke_width_slider = ctk.CTkSlider(pos_frame, from_=1, to=10, number_of_steps=9, variable=self.stroke_width_var, width=100, command=self.update_stroke_label)
        self.stroke_width_slider.pack(side="left", padx=5)
        self.stroke_width_label = ctk.CTkLabel(pos_frame, text="3px", width=40)
        self.stroke_width_label.pack(side="left")

        ctk.CTkLabel(pos_frame, text="Text Position:", width=100, anchor="w").pack(side="left", padx=(30, 0))
        self.text_pos_var = ctk.DoubleVar(value=0.75)
        self.text_pos_slider = ctk.CTkSlider(pos_frame, from_=0.5, to=0.9, variable=self.text_pos_var, width=100, command=self.update_pos_label)
        self.text_pos_slider.pack(side="left", padx=5)
        self.text_pos_label = ctk.CTkLabel(pos_frame, text="75%", width=40)
        self.text_pos_label.pack(side="left")

        # === OUTPUT SECTION ===
        output_frame = ctk.CTkFrame(self.main_frame)
        output_frame.pack(fill="x", padx=10, pady=5)

        output_label = ctk.CTkLabel(
            output_frame,
            text="📁 Output Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        output_label.pack(anchor="w", padx=10, pady=(10, 5))

        folder_frame = ctk.CTkFrame(output_frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(folder_frame, text="Save to:", width=120, anchor="w").pack(side="left")
        self.output_dir_var = ctk.StringVar(value=self.default_config['output_dir'])
        self.output_dir_entry = ctk.CTkEntry(folder_frame, textvariable=self.output_dir_var, width=400)
        self.output_dir_entry.pack(side="left", padx=5, fill="x", expand=True)

        self.browse_btn = ctk.CTkButton(folder_frame, text="📂 Browse", width=100, command=self.browse_folder)
        self.browse_btn.pack(side="left", padx=5)

        # === CONTROL BUTTONS ===
        control_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        control_frame.pack(fill="x", padx=10, pady=10)

        self.start_btn = ctk.CTkButton(
            control_frame,
            text="▶️ Start Processing",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            fg_color="#28a745",
            hover_color="#218838",
            command=self.start_processing
        )
        self.start_btn.pack(side="left", padx=5, expand=True, fill="x")

        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="⏹ Stop",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            fg_color="#dc3545",
            hover_color="#c82333",
            state="disabled",
            command=self.stop_processing
        )
        self.stop_btn.pack(side="left", padx=5, expand=True, fill="x")

        self.open_folder_btn = ctk.CTkButton(
            control_frame,
            text="📂 Open Output",
            font=ctk.CTkFont(size=16),
            height=45,
            command=self.open_output_folder
        )
        self.open_folder_btn.pack(side="left", padx=5, expand=True, fill="x")

        # === PROGRESS BAR ===
        self.progress_var = ctk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, variable=self.progress_var, height=15)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(self.main_frame, text="Ready", text_color="gray")
        self.progress_label.pack()

        # === LOG DISPLAY ===
        log_frame = ctk.CTkFrame(self.main_frame)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        log_label = ctk.CTkLabel(
            log_frame,
            text="📋 Processing Log",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        log_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.log_text = ctk.CTkTextbox(log_frame, height=150, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def update_clip_label(self, value):
        self.clip_count_label.configure(text=f"{int(value)} klip")

    def toggle_auto_clip(self):
        """Toggle auto clip mode - hide/show slider"""
        if self.auto_clip_var.get():
            # Hide slider, show auto info
            self.clip_slider_frame.pack_forget()
            self.auto_clip_info.pack(side="left", padx=10)
        else:
            # Show slider, hide auto info
            self.auto_clip_info.pack_forget()
            self.clip_slider_frame.pack(side="left", fill="x", expand=True)

    def update_font_size_label(self, value):
        self.font_size_label.configure(text=f"{int(value)}px")

    def update_stroke_label(self, value):
        self.stroke_width_label.configure(text=f"{int(value)}px")

    def update_pos_label(self, value):
        self.text_pos_label.configure(text=f"{int(value*100)}%")

    def toggle_api_key(self):
        if self.api_key_entry.cget("show") == "*":
            self.api_key_entry.configure(show="")
            self.show_key_btn.configure(text="🙈")
        else:
            self.api_key_entry.configure(show="*")
            self.show_key_btn.configure(text="👁")

    def pick_color(self, color_type):
        color = colorchooser.askcolor(title=f"Choose {color_type} color")[1]
        if color:
            if color_type == 'font':
                self.font_color_var.set(color)
                self.font_color_btn.configure(fg_color=color)
            elif color_type == 'font_alt':
                self.font_color_alt_var.set(color)
                self.font_color_alt_btn.configure(fg_color=color)
            elif color_type == 'stroke':
                self.stroke_color_var.set(color)
                self.stroke_color_btn.configure(fg_color=color)

    def browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if folder:
            self.output_dir_var.set(folder)

    def toggle_source_type(self):
        """Toggle between YouTube URL and local file input"""
        if self.source_type_var.get() == "youtube":
            self.file_frame.pack_forget()
            self.url_frame.pack(fill="x", padx=10, pady=5, after=self.youtube_radio.master)
        else:
            self.url_frame.pack_forget()
            self.file_frame.pack(fill="x", padx=10, pady=5, after=self.youtube_radio.master)

    def browse_video_file(self):
        """Browse for local video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.file_path_var.set(file_path)

    def toggle_subtitle_options(self):
        """Toggle subtitle options visibility based on checkbox"""
        # Just for visual feedback - actual logic handled in config
        pass

    def open_output_folder(self):
        folder = self.output_dir_var.get()
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            self.log("ERROR", f"Folder tidak ditemukan: {folder}")

    def log(self, level, message):
        self.log_queue.put((level, message))

    def check_log_queue(self):
        """Check log queue and update GUI"""
        try:
            while True:
                level, message = self.log_queue.get_nowait()

                # Color coding
                if level == "SUCCESS":
                    tag = "✅ "
                elif level == "ERROR":
                    tag = "❌ "
                elif level == "WARNING":
                    tag = "⚠️ "
                else:
                    tag = "ℹ️ "

                self.log_text.insert("end", f"{tag}{message}\n")
                self.log_text.see("end")
        except queue.Empty:
            pass

        self.after(100, self.check_log_queue)

    def start_processing(self):
        # Validation
        if not self.api_key_var.get().strip():
            self.log("ERROR", "Groq API Key tidak boleh kosong!")
            return

        # Check source type
        source_type = self.source_type_var.get()
        if source_type == "youtube":
            if not self.url_var.get().strip():
                self.log("ERROR", "YouTube URL tidak boleh kosong!")
                return
        else:
            if not self.file_path_var.get().strip():
                self.log("ERROR", "Pilih file video terlebih dahulu!")
                return
            if not os.path.exists(self.file_path_var.get()):
                self.log("ERROR", "File tidak ditemukan!")
                return

        # Update UI
        self.is_processing = True
        self.cancel_flag = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting...")

        # Clear log
        self.log_text.delete("1.0", "end")

        # Get config
        config = {
            'api_key': self.api_key_var.get().strip(),
            'source_type': source_type,
            'youtube_url': self.url_var.get().strip() if source_type == "youtube" else "",
            'local_file': self.file_path_var.get().strip() if source_type == "file" else "",
            'clip_count': self.clip_count_var.get(),
            'auto_clip': self.auto_clip_var.get(),
            'enable_subtitle': self.enable_subtitle_var.get(),
            'font_size': self.font_size_var.get(),
            'font_color': self.font_color_var.get(),
            'font_color_alt': self.font_color_alt_var.get(),
            'stroke_color': self.stroke_color_var.get(),
            'stroke_width': self.stroke_width_var.get(),
            'text_position': self.text_pos_var.get(),
            'output_dir': self.output_dir_var.get()
        }

        # Start processing in thread
        thread = threading.Thread(target=self.process_video, args=(config,), daemon=True)
        thread.start()

    def stop_processing(self):
        self.cancel_flag = True
        self.log("WARNING", "Stopping... Please wait for current operation to finish.")
        self.progress_label.configure(text="Stopping...")

    def update_progress(self, value, text):
        self.progress_bar.set(value)
        self.progress_label.configure(text=text)

    def processing_finished(self):
        self.is_processing = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def process_video(self, config):
        """Main processing logic - runs in separate thread"""
        temp_dir = "temp"
        output_dir = config['output_dir']

        # Create directories
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # 1. Get Source Video
            source_path = None

            if config['source_type'] == 'youtube':
                # Download from YouTube
                self.after(0, lambda: self.update_progress(0.1, "📥 Downloading video..."))
                self.log("INFO", f"Downloading: {config['youtube_url']}")

                source_path = self.download_video(config['youtube_url'], temp_dir)
                if not source_path or self.cancel_flag:
                    if self.cancel_flag:
                        self.log("WARNING", "Cancelled by user")
                    self.after(0, self.processing_finished)
                    return
            else:
                # Use local file
                self.after(0, lambda: self.update_progress(0.1, "📂 Loading local file..."))
                self.log("INFO", f"Using local file: {config['local_file']}")

                # Copy to temp dir for processing
                source_path = f"{temp_dir}/source_video.mp4"
                shutil.copy2(config['local_file'], source_path)
                self.log("SUCCESS", "Local file loaded!")

            self.log("SUCCESS", "Video downloaded successfully!")

            # 2. Extract Audio
            self.after(0, lambda: self.update_progress(0.2, "🎵 Extracting audio..."))
            self.log("INFO", "Extracting audio for transcription...")

            video = VideoFileClip(source_path)
            video_duration = video.duration
            audio_path = f"{temp_dir}/source_audio.wav"

            # Check if video has audio
            if video.audio is None:
                self.log("ERROR", "Video tidak memiliki audio track! Coba video lain.")
                video.close()
                self.after(0, self.processing_finished)
                return

            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()

            # Use auto clip count or manual setting
            if config['auto_clip']:
                # Auto mode: let user decide via slider, use that value
                self.log("INFO", f"Auto mode: menggunakan {config['clip_count']} klip")

            # No limit on clip count - user decides

            self.log("SUCCESS", f"Audio extracted! Duration: {int(video_duration/60)} minutes")

            if self.cancel_flag:
                self.after(0, self.processing_finished)
                return

            # 3. Transcribe
            self.after(0, lambda: self.update_progress(0.3, "🎤 Transcribing audio..."))
            self.log("INFO", "Transcribing with Whisper (this may take a while)...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log("INFO", f"Using device: {device.upper()}")

            model = whisper.load_model("base", device=device)
            whisper_result = model.transcribe(audio_path, language='id', task='transcribe', fp16=False, word_timestamps=True)

            if not whisper_result:
                self.log("ERROR", "Transcription failed!")
                self.after(0, self.processing_finished)
                return

            full_text = ""
            for seg in whisper_result['segments']:
                full_text += f"[{seg['start']:.1f}] {seg['text']}\n"
            all_words = [w for seg in whisper_result['segments'] for w in seg['words']]

            self.log("SUCCESS", f"Transcription complete! {len(all_words)} words detected")

            if self.cancel_flag:
                self.after(0, self.processing_finished)
                return

            # 4. Analyze with AI
            self.after(0, lambda: self.update_progress(0.4, "🤖 AI analyzing hooks..."))
            self.log("INFO", f"Sending to Groq AI for analysis...")

            clips_data = self.analyze_hooks_with_groq(config['api_key'], full_text, config['clip_count'])

            if not clips_data:
                self.log("ERROR", "AI could not find any clips!")
                self.after(0, self.processing_finished)
                return

            self.log("SUCCESS", f"Found {len(clips_data)} viral segments!")

            if self.cancel_flag:
                self.after(0, self.processing_finished)
                return

            # 5. Process each clip
            total_clips = len(clips_data)
            for i, data in enumerate(clips_data):
                if self.cancel_flag:
                    break

                progress = 0.5 + (0.5 * i / total_clips)
                clip_name = data.get('title', f'Clip_{i+1}')
                self.after(0, lambda p=progress, n=clip_name: self.update_progress(p, f"🎬 Processing: {n}"))
                self.log("INFO", f"Processing clip {i+1}/{total_clips}: {clip_name}")

                self.process_single_clip(
                    source_path,
                    float(data['start']),
                    float(data['end']),
                    f"Short_{i+1}_{clip_name}",
                    all_words,
                    config,
                    temp_dir,
                    output_dir
                )

            if self.cancel_flag:
                self.log("WARNING", "Processing cancelled by user")
            else:
                self.log("SUCCESS", f"🎉 All done! Check folder: {output_dir}")
                self.after(0, lambda: self.update_progress(1.0, "✅ Complete!"))

        except Exception as e:
            self.log("ERROR", f"Error: {str(e)}")
        finally:
            self.after(0, self.processing_finished)

    def download_video(self, url, temp_dir):
        """Download YouTube video"""
        output_path = f"{temp_dir}/source_video.mp4"
        if os.path.exists(output_path):
            os.remove(output_path)

        ydl_opts = {
            # Prioritize 1080p, then 720p, with audio merged
            'format': 'bestvideo[height>=720][height<=1080]+bestaudio/bestvideo+bestaudio/best',
            'outtmpl': f"{temp_dir}/raw_video.%(ext)s",
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 5,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            for file in os.listdir(temp_dir):
                if file.startswith("raw_video"):
                    src = os.path.join(temp_dir, file)
                    if file.endswith(".mp4"):
                        shutil.move(src, output_path)
                        return output_path
                    elif os.path.exists(src):
                        # Convert to mp4 if needed
                        shutil.move(src, output_path)
                        return output_path
            return None
        except Exception as e:
            self.log("ERROR", f"Download failed: {str(e)}")
            return None

    def analyze_hooks_with_groq(self, api_key, transcript_text, num_clips):
        """Analyze transcript with Groq AI"""
        client = Groq(api_key=api_key)
        safe_text = transcript_text[:25000]

        prompt = f"""
        You are a professional Video Editor. Analyze this transcript.
        Find exactly {num_clips} viral segments for TikTok/YouTube Shorts.

        STRICT DURATION RULES:
        - MINIMUM duration: 30 seconds
        - MAXIMUM duration: 60 seconds
        - Each clip MUST be between 30-60 seconds
        - Do NOT create clips shorter than 30 seconds

        CRITERIA:
        1. Must have a strong hook in the first 5 seconds.
        2. Must be self-contained context.
        3. Each segment duration = end - start must be >= 30 and <= 60 seconds.

        TRANSCRIPT:
        {safe_text} ... (truncated)

        OUTPUT STRICT JSON ONLY (ensure end - start is between 30 and 60 for each):
        [
          {{ "start": 120.0, "end": 165.0, "title": "Judul_Klip_1" }},
          {{ "start": 300.5, "end": 350.0, "title": "Judul_Klip_2" }}
        ]
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON. Each clip MUST be 30-60 seconds long."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.6,
                response_format={"type": "json_object"},
            )
            result_content = chat_completion.choices[0].message.content
            data = json.loads(result_content)

            clips = []
            if isinstance(data, list):
                clips = data
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        clips = v
                        break

            # Validate and filter clips - enforce 30-60 second rule
            valid_clips = []
            for clip in clips:
                try:
                    start = float(clip.get('start', 0))
                    end = float(clip.get('end', 0))
                    duration = end - start

                    if duration < 30:
                        # Extend clip to 30 seconds if too short
                        self.log("WARNING", f"Clip '{clip.get('title', 'Unknown')}' too short ({duration:.1f}s), extending to 30s")
                        clip['end'] = start + 30
                    elif duration > 60:
                        # Trim clip to 60 seconds if too long
                        self.log("WARNING", f"Clip '{clip.get('title', 'Unknown')}' too long ({duration:.1f}s), trimming to 60s")
                        clip['end'] = start + 60

                    valid_clips.append(clip)
                except:
                    continue

            return valid_clips
        except Exception as e:
            self.log("ERROR", f"Groq API Error: {str(e)}")
            return []

    def process_single_clip(self, source_video, start_t, end_t, clip_name, segment_words, config, temp_dir, output_dir):
        """Process a single clip with face tracking and subtitles"""
        try:
            full_clip = VideoFileClip(source_video)
            if end_t > full_clip.duration:
                end_t = full_clip.duration
            clip = full_clip.subclip(start_t, end_t)

            # Save temp file for face detection
            temp_sub = f"{temp_dir}/temp_{clip_name}.mp4"
            clip.write_videofile(temp_sub, codec='libx264', audio_codec='aac', logger=None)

            # Face tracking with OpenCV Haar Cascade (more reliable than mediapipe)
            cap = cv2.VideoCapture(temp_sub)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            centers = []

            try:
                # Use OpenCV's built-in Haar Cascade face detector
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                frame_skip = 2  # Process every 2nd frame for better accuracy
                frame_idx = 0
                last_x_c = width // 2
                face_found_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip == 0:
                        # Resize for faster detection (keeping more resolution for accuracy)
                        scale = 0.4
                        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

                        # Use lower scaleFactor for more accurate detection
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.05,  # More accurate
                            minNeighbors=4,    # More sensitive
                            minSize=(20, 20)   # Detect smaller faces
                        )

                        if len(faces) > 0:
                            # Get the largest face
                            largest_face = max(faces, key=lambda f: f[2] * f[3])
                            x, y, w, h = largest_face
                            # Scale back to original size and get face center
                            face_center_x = int((x + w/2) / scale)

                            # Smooth transition - don't jump too fast
                            max_jump = width * 0.05  # Max 5% of width per detection
                            diff = face_center_x - last_x_c
                            if abs(diff) > max_jump:
                                face_center_x = last_x_c + (max_jump if diff > 0 else -max_jump)

                            last_x_c = int(face_center_x)
                            face_found_count += 1

                    centers.append(last_x_c)
                    frame_idx += 1

                self.log("INFO", f"Face tracking: analyzed {len(centers)} frames, {face_found_count} faces detected")

            except Exception as e:
                # Fallback: use center crop if face detection fails
                self.log("WARNING", f"Face tracking failed, using center crop: {str(e)[:50]}")
                cap.release()
                cap = cv2.VideoCapture(temp_sub)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                centers = [width // 2] * max(1, frame_count)

            cap.release()

            if not centers:
                centers = [width//2]

            # Apply stronger smoothing to prevent jerky movement
            window = 30  # Larger window for smoother tracking
            if len(centers) > window:
                centers = np.convolve(centers, np.ones(window)/window, mode='same')

            # Crop function for 9:16 portrait
            def crop_fn(get_frame, t):
                idx = int(t * fps)
                safe_idx = min(idx, len(centers)-1)
                cx = centers[safe_idx]
                img = get_frame(t)
                h, w = img.shape[:2]
                target_width = int(h * 9/16)
                # Ensure target_width is even (required by H.264 encoder)
                target_width = target_width - (target_width % 2)
                x1 = int(cx - target_width/2)
                x1 = max(0, min(w - target_width, x1))
                return img[:, x1:x1+target_width]

            # Resize to 1080x1920 (even dimensions for H.264)
            final_clip = clip.fl(crop_fn, apply_to=['mask']).resize((1080, 1920))

            # Add subtitles (if enabled)
            subs = []
            if config.get('enable_subtitle', True):
                vid_w, vid_h = final_clip.w, final_clip.h
                valid_words = [w for w in segment_words if w['start'] >= start_t and w['end'] <= end_t]

                self.log("INFO", f"Adding subtitles: {len(valid_words)} words found")

                # Try to load font
                try:
                    font_path = "C:/Windows/Fonts/impact.ttf"
                    if not os.path.exists(font_path):
                        font_path = "C:/Windows/Fonts/arial.ttf"
                    font = ImageFont.truetype(font_path, config['font_size'])
                except:
                    font = ImageFont.load_default()

                for w in valid_words:
                    try:
                        raw_text = w.get('word', w.get('text', '')).strip()
                        if not raw_text:
                            continue

                        text = raw_text.upper()
                        color = config['font_color_alt'] if len(text) <= 3 else config['font_color']
                        pos_y = int(vid_h * config['text_position'])

                        # Create text image with PIL
                        # Calculate text size
                        dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
                        dummy_draw = ImageDraw.Draw(dummy_img)
                        bbox = dummy_draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0] + 20
                        text_h = bbox[3] - bbox[1] + 20

                        # Create transparent image with text
                        txt_img = Image.new('RGBA', (text_w, text_h), (0, 0, 0, 0))
                        draw = ImageDraw.Draw(txt_img)

                        # Convert hex color to RGB
                        stroke_c = config['stroke_color'].lstrip('#')
                        stroke_rgb = tuple(int(stroke_c[i:i+2], 16) for i in (0, 2, 4))
                        font_c = color.lstrip('#')
                        font_rgb = tuple(int(font_c[i:i+2], 16) for i in (0, 2, 4))

                        # Draw text with stroke
                        x, y = 10, 10
                        stroke_w = config['stroke_width']
                        for dx in range(-stroke_w, stroke_w+1):
                            for dy in range(-stroke_w, stroke_w+1):
                                draw.text((x+dx, y+dy), text, font=font, fill=stroke_rgb)
                        draw.text((x, y), text, font=font, fill=font_rgb)

                        # Convert to numpy array and create ImageClip
                        txt_array = np.array(txt_img)
                        txt_clip = (ImageClip(txt_array, ismask=False)
                            .set_position(('center', pos_y))
                            .set_start(w['start'] - start_t)
                            .set_end(w['end'] - start_t)
                            .set_duration(w['end'] - w['start']))

                        subs.append(txt_clip)
                    except Exception as e:
                        self.log("WARNING", f"Subtitle error: {str(e)[:50]}")
                        continue

                self.log("INFO", f"Created {len(subs)} subtitle clips")
            else:
                self.log("INFO", "Subtitles disabled")

            final = CompositeVideoClip([final_clip] + subs)

            # Output
            safe_name = "".join([c for c in clip_name if c.isalnum() or c == '_'])
            output_filename = f"{output_dir}/{safe_name}.mp4"

            final.write_videofile(
                output_filename,
                codec='libx264',
                audio_codec='aac',
                fps=24,
                preset='fast',
                threads=4,
                logger=None,
                ffmpeg_params=['-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level', '3.0']
            )

            full_clip.close()
            final.close()

            if os.path.exists(temp_sub):
                os.remove(temp_sub)

            self.log("SUCCESS", f"Saved: {output_filename}")

        except Exception as e:
            self.log("ERROR", f"Failed to process {clip_name}: {str(e)}")


if __name__ == "__main__":
    # Load env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass

    app = AIAutoShortsApp()
    app.mainloop()
