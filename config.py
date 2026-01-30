"""
Kliperr Configuration - Centralized Constants
All hardcoded values are defined here for easy maintenance.
"""
import os

TEMP_DIR = "temp"
OUTPUT_DIR = "hasil_shorts"
FONT_DIR = "fonts"

MAX_WORKERS = 6
OUTPUT_FPS = 30
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
VIDEO_BITRATE = "8M"
VIDEO_BITRATE_MAX = "10M"
VIDEO_BUFSIZE = "16M"
AUDIO_BITRATE = "256k"

FACE_SKIP_FRAMES = 3
FACE_EMA_ALPHA = 0.18
FACE_MAX_JUMP_RATIO = 0.02
FACE_CENTER_PULL_STRENGTH = 0.005
FACE_MIN_AREA_RATIO = 0.01
FACE_DETECTION_CONFIDENCE = 0.5
FACE_NO_DETECT_THRESHOLD = 10
GAUSSIAN_SIGMA_MAX = 90
GAUSSIAN_SIGMA_MIN = 10

ENCODE_PRESET_GPU = "fast"
ENCODE_PRESET_CPU = "ultrafast"
GPU_VIDEO_CODEC = "h264_nvenc"
CPU_VIDEO_CODEC = "libx264"
NVENC_PRESET = "p4"
CPU_PRESET_FINAL = "slow"

DEFAULT_FONT = "Impact"
DEFAULT_FONT_SIZE = 70
DEFAULT_FONT_COLOR = "#FFFF00"
DEFAULT_FONT_COLOR_ALT = "#FFFFFF"
DEFAULT_STROKE_COLOR = "#000000"
DEFAULT_STROKE_WIDTH = 4
DEFAULT_TEXT_POSITION = 0.75
SUBTITLE_ANIMATION_DURATION = 0.05

IMAGEMAGICK_PATH = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MEDIAPIPE_MODEL_NAME = "blaze_face_short_range.tflite"

WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"
WHISPER_LANGUAGE = "id"
AUDIO_MAX_SIZE_MB = 20

YT_DLP_SOCKET_TIMEOUT = 30
YT_DLP_RETRIES = 5
YT_DLP_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"

APP_TITLE = "ClipGenAI"
APP_ICON = "✂️"

LOG_COLORS = {
    "SUCCESS": "#00ff88",
    "ERROR": "#ff4444",
    "WARNING": "#ffaa00",
    "INFO": "#00aaff"
}

PRIMARY_COLOR = "#00ff88"
CANCEL_COLOR = "#ff4444"
BACKGROUND_COLOR = "#0a0a0a"
BORDER_COLOR = "#222"

def is_gpu_enabled():
    """Check if GPU encoding should be used based on APP_ENV"""
    return os.getenv("APP_ENV", "production").lower() == "local"

def get_encode_preset():
    """Get appropriate encoding preset based on environment"""
    return ENCODE_PRESET_GPU if is_gpu_enabled() else ENCODE_PRESET_CPU

def get_video_codec():
    """Get appropriate video codec based on environment"""
    return GPU_VIDEO_CODEC if is_gpu_enabled() else CPU_VIDEO_CODEC

def get_ffmpeg_preset():
    """Get FFmpeg preset based on environment"""
    return NVENC_PRESET if is_gpu_enabled() else CPU_PRESET_FINAL
