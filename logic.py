import os
import json
import cv2
import numpy as np
import yt_dlp
import torch
import shutil
import threading
import subprocess
from groq import Groq
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
from moviepy.config import change_settings

import config as cfg

_model_download_lock = threading.Lock()
_mediapipe_model_path = None

change_settings({"IMAGEMAGICK_BINARY": cfg.IMAGEMAGICK_PATH})

class VideoProcessor:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.cancel_flag = False
        self.task_manager = None

    def log(self, level, message):
        if self.log_callback:
            self.log_callback(level, message)
        else:
            print(f"[{level}] {message}")

    def stop_processing(self):
        self.cancel_flag = True
        self.log("WARNING", "Stop signal received...")
        if self.task_manager:
            self.task_manager.cancel()

    def process_video(self, config, progress_callback=None):
        """Main processing logic"""
        temp_dir = cfg.TEMP_DIR
        output_dir = config['output_dir']

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        try:
            source_path = None
            
            if config['source_type'] == 'youtube':
                if progress_callback: progress_callback(0.1, "📥 Downloading video...")
                self.log("INFO", f"Downloading: {config['youtube_url']}")

                source_path = self.download_video(config['youtube_url'], temp_dir)
                if not source_path or self.cancel_flag:
                    if self.cancel_flag:
                        self.log("WARNING", "Cancelled by user")
                    return
            else:
                if progress_callback: progress_callback(0.1, "📂 Loading local file...")
                self.log("INFO", f"Using local file: {config['local_file']}")

                source_path = f"{temp_dir}/source_video.mp4"
                shutil.copy2(config['local_file'], source_path)
                self.log("SUCCESS", "Local file loaded!")

            self.log("SUCCESS", "Video source ready!")

            if progress_callback: progress_callback(0.2, "🎵 Extracting audio...")
            self.log("INFO", "Extracting audio for transcription...")

            video = VideoFileClip(source_path)
            video_duration = video.duration
            audio_path = f"{temp_dir}/source_audio.wav"

            if video.audio is None:
                self.log("ERROR", "Video has no audio track!")
                video.close()
                return

            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()

            if self.cancel_flag: return

            if progress_callback: progress_callback(0.3, "🎤 Transcribing audio...")
            
            whisper_result = self.transcribe_audio(audio_path, config['api_key'])
            
            if not whisper_result:
                self.log("ERROR", "Transcription failed!")
                return

            full_text = ""
            for seg in whisper_result['segments']:
                full_text += f"[{seg['start']:.1f}] {seg['text']}\n"
            all_words = [w for seg in whisper_result['segments'] for w in seg['words']]

            self.log("SUCCESS", f"Transcription complete! {len(all_words)} words detected")

            if self.cancel_flag: return

            if progress_callback: progress_callback(0.4, "🤖 AI analyzing hooks...")
            self.log("INFO", f"Sending to Groq AI for analysis...")

            clips_data = self.analyze_hooks_with_groq(config['api_key'], full_text, config['clip_count'])

            if not clips_data:
                self.log("ERROR", "AI could not find any clips!")
                return

            self.log("SUCCESS", f"Found {len(clips_data)} viral segments!")

            if self.cancel_flag: return

            from clip_manager import ClipTaskManager
            
            total_clips = len(clips_data)
            max_workers = min(cfg.MAX_WORKERS, total_clips)
            
            if progress_callback: progress_callback(0.5, f"🎬 Processing {total_clips} clips in parallel...")
            self.log("INFO", f"🚀 Starting parallel processing: {total_clips} clips with {max_workers} worker(s)")
            
            self.task_manager = ClipTaskManager(
                max_workers=max_workers,
                log_callback=self.log_callback,
                progress_callback=progress_callback,
                video_callback=config.get('video_callback')
            )
            
            completed = self.task_manager.process_clips_parallel(
                processor=self,
                clips_data=clips_data,
                source_path=source_path,
                all_words=all_words,
                config=config,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

            if self.cancel_flag or self.task_manager.cancel_flag.is_set():
                self.log("WARNING", "Processing cancelled by user")
            else:
                self.log("SUCCESS", f"🎉 All done! Completed {completed}/{total_clips} clips")
                if progress_callback: progress_callback(1.0, "✅ Complete!")
            
            self.task_manager = None

        except Exception as e:
            self.log("ERROR", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def download_video(self, url, temp_dir):
        """Download YouTube video"""
        output_path = f"{temp_dir}/source_video.mp4"
        if os.path.exists(output_path):
            os.remove(output_path)

        ydl_opts = {
            'format': 'bestvideo[height>=1080]+bestaudio/bestvideo[height>=720]+bestaudio/bestvideo+bestaudio/best',
            'outtmpl': f"{temp_dir}/raw_video.%(ext)s",
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
            'socket_timeout': 60,
            'retries': 10,
            'fragment_retries': 10,
            'nocheckcertificate': True,
            'prefer_ffmpeg': True,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        
        self.log("INFO", "🎬 Downloading HD video (1080p/720p)...")

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
                        shutil.move(src, output_path)
                        return output_path
            return None
        except Exception as e:
            self.log("ERROR", f"Download failed: {str(e)}")
            return None

    def transcribe_audio(self, audio_path, api_key):    
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)  
        try:
            if file_size_mb > cfg.AUDIO_MAX_SIZE_MB:
                self.log("INFO", f"Audio file is {file_size_mb:.1f}MB, preprocessing to reduce size...")
                audio_path = self._preprocess_audio(audio_path)
                new_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                self.log("SUCCESS", f"Preprocessed: {file_size_mb:.1f}MB → {new_size_mb:.1f}MB")
            
            self.log("INFO", "🚀 Attempting Groq Whisper API (ultra-fast)...")
            client = Groq(api_key=api_key)
            
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(audio_path, audio_file.read()),
                    model=cfg.WHISPER_MODEL,
                    response_format="verbose_json",
                    language=cfg.WHISPER_LANGUAGE,
                    timestamp_granularities=["word"]
                )
            
            words_list = None
            if isinstance(transcription, dict):
                words_list = transcription.get('words', [])
            elif hasattr(transcription, 'words'):
                words_list = transcription.words
            
            if words_list:
                segments = []
                current_segment = {"start": 0, "end": 0, "text": "", "words": []}
                
                for word in words_list:
                    if isinstance(word, dict):
                        word_data = {
                            "word": word.get('word', ''),
                            "start": word.get('start', 0),
                            "end": word.get('end', 0)
                        }
                    else:
                        word_data = {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end
                        }
                    
                    current_segment["words"].append(word_data)
                    current_segment["text"] += word_data["word"] + " "
                    
                    if len(current_segment["words"]) >= 10 or word_data["word"].strip().endswith(('.', '?', '!')):
                        current_segment["start"] = current_segment["words"][0]["start"]
                        current_segment["end"] = current_segment["words"][-1]["end"]
                        segments.append(current_segment)
                        current_segment = {"start": 0, "end": 0, "text": "", "words": []}
                
                if current_segment["words"]:
                    current_segment["start"] = current_segment["words"][0]["start"]
                    current_segment["end"] = current_segment["words"][-1]["end"]
                    segments.append(current_segment)
                
                total_words = sum(len(s["words"]) for s in segments)
                self.log("SUCCESS", f"✅ Groq Whisper API: {total_words} words in {len(segments)} segments")
                return {"segments": segments}
            else:
                raise Exception("No words in Groq response")
                
        except Exception as e:
            self.log("WARNING", f"Groq Whisper API failed: {str(e)[:100]}")
            self.log("INFO", "Falling back to faster-whisper...")
        
        try:
            from faster_whisper import WhisperModel
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            self.log("INFO", f"⚡ Using faster-whisper on {device.upper()} ({compute_type})")
            
            model = WhisperModel("base", device=device, compute_type=compute_type)
            segments_iter, info = model.transcribe(
                audio_path,
                language="id",
                word_timestamps=True,
                vad_filter=True
            )
            
            segments = []
            for segment in segments_iter:
                words = []
                for word in segment.words:
                    words.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    })
                
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": words
                })
            
            total_words = sum(len(s["words"]) for s in segments)
            self.log("SUCCESS", f"✅ faster-whisper: {total_words} words in {len(segments)} segments")
            return {"segments": segments}
            
        except ImportError:
            self.log("WARNING", "faster-whisper not installed, falling back to standard whisper")
        except Exception as e:
            self.log("WARNING", f"faster-whisper failed: {str(e)[:60]}")
        
        try:
            import whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log("INFO", f"🐌 Using standard Whisper on {device.upper()} (this may take a while)...")
            
            model = whisper.load_model("base", device=device)
            result = model.transcribe(audio_path, language='id', task='transcribe', fp16=False, word_timestamps=True)
            
            total_words = sum(len(s.get("words", [])) for s in result["segments"])
            self.log("SUCCESS", f"✅ Standard Whisper: {total_words} words")
            return result
            
        except Exception as e:
            self.log("ERROR", f"All transcription methods failed: {str(e)}")
            return None
    
    def _preprocess_audio(self, audio_path):
        """Preprocess audio to reduce file size (downsample to 16kHz mono FLAC)"""
        import subprocess
        
        temp_dir = os.path.dirname(audio_path)
        preprocessed_path = os.path.join(temp_dir, "preprocessed_audio.flac")
        
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ar', '16000',
            '-ac', '1',
            '-map', '0:a',
            '-c:a', 'flac',
            preprocessed_path
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(preprocessed_path):
                return preprocessed_path
            else:
                self.log("WARNING", "Audio preprocessing failed, using original file")
                return audio_path
                
        except Exception as e:
            self.log("WARNING", f"Audio preprocessing error: {str(e)[:60]}")
            return audio_path


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
        - Each clip MUST be between 40-120 seconds
        - Do NOT create clips shorter than 40 seconds

        CRITERIA:
        1. Must have a strong hook in the first 5 seconds.
        2. Must be self-contained context.
        3. Each segment duration = end - start must be >= 30 and <= 60 seconds.

        TRANSCRIPT:
        {safe_text} ... (truncated)

        OUTPUT STRICT JSON ONLY (ensure end - start is between 30 and 60 for each):
        [
          {{ "start": 120.0, "end": 165.0, "title": "Klip_1" }},
          {{ "start": 300.5, "end": 350.0, "title": "Klip_2" }}
        ]
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON. Each clip MUST be 30-60 seconds long."},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-20b",
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

            valid_clips = []
            for clip in clips:
                try:
                    start = float(clip.get('start', 0))
                    end = float(clip.get('end', 0))
                    duration = end - start

                    if duration < 30:
                        self.log("WARNING", f"Clip '{clip.get('title', 'Unknown')}' too short ({duration:.1f}s), extending to 30s")
                        clip['end'] = start + 30
                    elif duration > 60:
                        self.log("WARNING", f"Clip '{clip.get('title', 'Unknown')}' too long ({duration:.1f}s), trimming to 60s")
                        clip['end'] = start + 60

                    valid_clips.append(clip)
                except:
                    continue

            return valid_clips
        except Exception as e:
            self.log("ERROR", f"Groq API Error: {str(e)}")
            return []

    def process_single_clip(self, source_video, start_t, end_t, clip_name, segment_words, config, temp_dir, output_dir, video_callback=None):
        """Process a single clip with face tracking and subtitles - OPTIMIZED"""
        import time
        perf_start = time.perf_counter()
        
        try:
            safe_clip_name = ''.join(c if c.isalnum() else '_' for c in clip_name)[:50]
            
            cap = cv2.VideoCapture(source_video)
            total_fps = cap.get(cv2.CAP_PROP_FPS)
            total_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            total_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.log("INFO", f"📹 Source video: {total_width}x{total_height} @ {total_fps:.1f}fps")
            
            if total_height < 1080:
                self.log("WARNING", f"⚠️ Source video is NOT HD! ({total_height}p) - Quality may be limited")
            
            start_frame = int(start_t * total_fps)
            end_frame = int(end_t * total_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            width = total_width
            height = total_height
            fps = total_fps

            centers = []

            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                import urllib.request
                
                global _mediapipe_model_path
                
                with _model_download_lock:
                    if _mediapipe_model_path is None or not os.path.exists(_mediapipe_model_path):
                        model_path = os.path.join(temp_dir, cfg.MEDIAPIPE_MODEL_NAME)
                        if not os.path.exists(model_path):
                            self.log("INFO", "Downloading MediaPipe face detection model...")
                            urllib.request.urlretrieve(cfg.MEDIAPIPE_MODEL_URL, model_path)
                            self.log("SUCCESS", "Model downloaded!")
                        _mediapipe_model_path = model_path
                
                use_gpu = torch.cuda.is_available()
                if use_gpu:
                    try:
                        delegate = python.BaseOptions.Delegate.GPU
                        base_options = python.BaseOptions(
                            model_asset_path=_mediapipe_model_path,
                            delegate=delegate
                        )
                        self.log("INFO", "🚀 Face detection using GPU delegate")
                    except Exception:
                        base_options = python.BaseOptions(model_asset_path=_mediapipe_model_path)
                        self.log("INFO", "Face detection using CPU (GPU delegate failed)")
                else:
                    base_options = python.BaseOptions(model_asset_path=_mediapipe_model_path)
                
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=cfg.FACE_DETECTION_CONFIDENCE
                )
                
                with vision.FaceDetector.create_from_options(options) as face_detector:
                    frame_idx = 0
                    center_x_default = width // 2
                    last_x_c = center_x_default
                    smoothed_x = float(center_x_default)
                    face_found_count = 0
                    frames_to_process = end_frame - start_frame
                    no_face_count = 0

                    while frame_idx < frames_to_process:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        face_detected_this_frame = False
                        
                        if frame_idx % cfg.FACE_SKIP_FRAMES == 0:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            detection_result = face_detector.detect(mp_image)
                            
                            if detection_result.detections:
                                best_detection = None
                                max_area = 0
                                min_area = width * height * cfg.FACE_MIN_AREA_RATIO
                                
                                for detection in detection_result.detections:
                                    bbox = detection.bounding_box
                                    area = bbox.width * bbox.height
                                    if area > max_area and area > min_area:
                                        max_area = area
                                        best_detection = bbox
                                
                                if best_detection:
                                    raw_center_x = best_detection.origin_x + best_detection.width // 2
                                    
                                    max_jump = width * cfg.FACE_MAX_JUMP_RATIO
                                    diff = raw_center_x - last_x_c
                                    if abs(diff) > max_jump:
                                        target_x = last_x_c + (max_jump if diff > 0 else -max_jump)
                                    else:
                                        target_x = raw_center_x
                                    
                                    smoothed_x = cfg.FACE_EMA_ALPHA * target_x + (1 - cfg.FACE_EMA_ALPHA) * smoothed_x
                                    last_x_c = int(smoothed_x)
                                    face_found_count += 1
                                    face_detected_this_frame = True
                                    no_face_count = 0
                        
                        if not face_detected_this_frame:
                            no_face_count += 1
                            if no_face_count > cfg.FACE_NO_DETECT_THRESHOLD:
                                pull_to_center = (center_x_default - smoothed_x) * cfg.FACE_CENTER_PULL_STRENGTH
                                smoothed_x += pull_to_center
                                last_x_c = int(smoothed_x)
                        
                        centers.append(int(smoothed_x))
                        frame_idx += 1

                total_tracked = len(centers)
                detection_rate = (face_found_count / (total_tracked / cfg.FACE_SKIP_FRAMES)) * 100 if total_tracked > 0 else 0
                self.log("INFO", f"Face tracking: {total_tracked} frames, {face_found_count} detections ({detection_rate:.0f}%)")

            except ImportError:
                self.log("ERROR", "MediaPipe not installed! Run `pip install mediapipe`")
                frame_count = end_frame - start_frame
                centers = [width // 2] * max(1, frame_count)

            except Exception as e:
                self.log("WARNING", f"Face tracking failed: {str(e)[:100]}")
                frame_count = end_frame - start_frame
                centers = [width // 2] * max(1, frame_count)

            cap.release()

            if not centers:
                centers = [width//2]

            from scipy.ndimage import gaussian_filter1d
            sigma = min(cfg.GAUSSIAN_SIGMA_MAX, len(centers) // 4) if len(centers) > 30 else cfg.GAUSSIAN_SIGMA_MIN
            centers = gaussian_filter1d(centers, sigma=sigma)

            def crop_fn(get_frame, t):
                idx = int(t * fps)
                safe_idx = min(idx, len(centers)-1)
                cx = centers[safe_idx]
                img = get_frame(t)
                h, w = img.shape[:2]
                target_width = int(h * 9/16)
                target_width = target_width - (target_width % 2)
                x1 = int(cx - target_width/2)
                x1 = max(0, min(w - target_width, x1))
                return img[:, x1:x1+target_width]

            full_clip = VideoFileClip(source_video)
            if end_t > full_clip.duration:
                end_t = full_clip.duration
            clip = full_clip.subclip(start_t, end_t)
            
            cropped_clip = clip.fl(crop_fn, apply_to=['mask'])
            
            safe_name = "".join([c for c in clip_name if c.isalnum() or c == '_'])
            output_filename = f"{output_dir}/{safe_name}.mp4"
            temp_video_cropped = f"{temp_dir}/temp_cropped_{safe_name}.mp4"
            temp_video_noSub = f"{temp_dir}/temp_nosub_{safe_name}.mp4"
            
            cropped_clip.write_videofile(
                temp_video_cropped,
                codec=cfg.CPU_VIDEO_CODEC,
                audio_codec='aac',
                fps=cfg.OUTPUT_FPS,
                preset='ultrafast',
                threads=4,
                logger=None,
                bitrate='50M'
            )
            
            full_clip.close()
            cropped_clip.close()
            
            self.log("INFO", "🔄 Re-encoding with high quality (no scaling)...")
            
            encode_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_cropped,
                '-c:v', cfg.CPU_VIDEO_CODEC,
                '-preset', cfg.get_encode_preset(),
                '-crf', cfg.VIDEO_CRF,
                '-b:v', cfg.VIDEO_BITRATE,
                '-maxrate', cfg.VIDEO_BITRATE_MAX,
                '-bufsize', cfg.VIDEO_BUFSIZE,
                '-profile:v', 'high',
                '-level', '4.2',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', cfg.AUDIO_BITRATE,
                temp_video_noSub
            ]
            
            encode_result = subprocess.run(encode_cmd, capture_output=True, text=True)
            
            if encode_result.returncode != 0:
                self.log("WARNING", f"FFmpeg encode failed: {encode_result.stderr[:200]}")
                shutil.copy(temp_video_cropped, temp_video_noSub)
            
            if os.path.exists(temp_video_cropped):
                os.remove(temp_video_cropped)
            
            face_track_time = time.perf_counter() - perf_start

            if config.get('enable_subtitle', True):
                valid_words = [w for w in segment_words if w['start'] >= start_t and w['end'] <= end_t]
                self.log("INFO", f"Creating ASS subtitles: {len(valid_words)} words")
                
                ass_path = f"{temp_dir}/subs_{safe_name}.ass"
                
                font_name = config.get('font_name', 'Impact')
                font_size = config['font_size']
                stroke_width = config['stroke_width']
                
                def hex_to_ass(hex_color):
                    hex_color = hex_color.lstrip('#')
                    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                    return f"&H00{b:02X}{g:02X}{r:02X}"
                
                primary_color = hex_to_ass(config['font_color'])
                secondary_color = hex_to_ass(config['font_color_alt'])
                outline_color = hex_to_ass(config['stroke_color'])
                
                margin_v = int(1920 * (1 - config['text_position']))
                
                ass_content = f"""[Script Info]
Title: Kliperr Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Highlight,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{stroke_width},3,2,50,50,{margin_v},1
Style: Normal,{font_name},{font_size},{secondary_color},&H000000FF,{outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{stroke_width},3,2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
                
                def time_to_ass(seconds):
                    seconds = max(0, seconds - start_t)
                    h = int(seconds // 3600)
                    m = int((seconds % 3600) // 60)
                    s = int(seconds % 60)
                    cs = int((seconds % 1) * 100)
                    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
                
                for w in valid_words:
                    try:
                        raw_text = w.get('word', w.get('text', '')).strip()
                        if not raw_text:
                            continue
                        
                        text = raw_text.upper()
                        start_time = time_to_ass(w['start'])
                        end_time = time_to_ass(w['end'])
                        
                        style = "Highlight" if len(text) > 3 else "Normal"
                        
                        ass_content += f"Dialogue: 0,{start_time},{end_time},{style},,0,0,0,,{text}\n"
                    except:
                        continue
                
                with open(ass_path, 'w', encoding='utf-8') as f:
                    f.write(ass_content)
                
                self.log("INFO", "Burning subtitles with FFmpeg...")
                
                video_codec = cfg.get_video_codec()
                preset = cfg.get_ffmpeg_preset()
                
                if cfg.is_gpu_enabled():
                     self.log("INFO", f"🚀 Using GPU encoding ({video_codec})")
                else:
                     self.log("INFO", f"Using CPU encoding ({video_codec})")

                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video_noSub,
                    '-vf', f"ass='{ass_path.replace(chr(92), '/')}'",
                    '-c:v', video_codec,
                    '-preset', preset,
                    '-crf', cfg.VIDEO_CRF,
                    '-b:v', cfg.VIDEO_BITRATE,
                    '-maxrate', cfg.VIDEO_BITRATE_MAX,
                    '-bufsize', cfg.VIDEO_BUFSIZE,
                    '-profile:v', 'high',
                    '-level', '4.2',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', cfg.AUDIO_BITRATE,
                    '-movflags', '+faststart',
                    output_filename
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.log("WARNING", f"FFmpeg subtitle failed, using video without subs")
                    shutil.copy(temp_video_noSub, output_filename)
                
                if os.path.exists(ass_path):
                    os.remove(ass_path)
            else:
                shutil.copy(temp_video_noSub, output_filename)
            
            if os.path.exists(temp_video_noSub):
                os.remove(temp_video_noSub)

            total_time = time.perf_counter() - perf_start
            self.log("SUCCESS", f"✅ Saved: {output_filename} ({total_time:.1f}s)")
            
            if video_callback:
                video_callback(output_filename)

        except Exception as e:
            self.log("ERROR", f"Failed to process {clip_name}: {str(e)}")
