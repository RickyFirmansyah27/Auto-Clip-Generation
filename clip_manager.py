import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ClipTaskManager:
    """Manages parallel video clip processing using thread pool"""
    
    def __init__(self, max_workers=4, log_callback=None, progress_callback=None, video_callback=None):
        self.max_workers = max_workers
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.video_callback = video_callback
        self.cancel_flag = threading.Event()
        self.lock = threading.Lock()
        self.completed_count = 0
    
    def _safe_log(self, level, message):
        """Thread-safe logging"""
        if self.log_callback:
            with self.lock:
                self.log_callback(level, message)
    
    def _safe_progress(self, value, text):
        """Thread-safe progress update"""
        if self.progress_callback:
            with self.lock:
                self.progress_callback(value, text)
    
    def _safe_video_callback(self, path):
        """Thread-safe video callback - just notify, don't render"""
        if self.video_callback:
            # Just call the callback with the path
            # No locks needed since we're not doing Streamlit operations
            self.video_callback(path)
    
    def _update_overall_progress(self, total_clips):
        """Update overall progress based on completed count"""
        with self.lock:
            self.completed_count += 1
            progress = 0.5 + (0.5 * self.completed_count / total_clips)
            self._safe_progress(progress, f"✅ Completed {self.completed_count}/{total_clips} clips")
    
    def _process_clip_wrapper(self, processor, clip_index, clip_data, total_clips, 
                              source_path, all_words, config, temp_dir, output_dir):
        if self.cancel_flag.is_set():
            return
        
        clip_name = clip_data.get('title', f'Clip_{clip_index+1}')
        self._safe_log("INFO", f"[Thread {clip_index+1}] Starting: {clip_name}")
        
        try:
            # Process the clip
            processor.process_single_clip(
                source_path,
                float(clip_data['start']),
                float(clip_data['end']),
                f"Short_{clip_index+1}_{clip_name}",
                all_words,
                config,
                temp_dir,
                output_dir,
                video_callback=self._safe_video_callback
            )
            
            # Update overall progress
            self._update_overall_progress(total_clips)
            self._safe_log("SUCCESS", f"[Thread {clip_index+1}] Finished: {clip_name}")
            
        except Exception as e:
            self._safe_log("ERROR", f"[Thread {clip_index+1}] Failed {clip_name}: {str(e)[:60]}")
            self._update_overall_progress(total_clips)  # Still update progress even on failure
    
    def process_clips_parallel(self, processor, clips_data, source_path, all_words, 
                               config, temp_dir, output_dir):
        total_clips = len(clips_data)
        self.completed_count = 0
        self.cancel_flag.clear()
        
        self._safe_log("INFO", f"Starting parallel processing with {self.max_workers} workers")
        self._safe_progress(0.5, f"🎬 Processing {total_clips} clips in parallel...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all clip processing tasks
            futures = {}
            
            for i, data in enumerate(clips_data):
                future = executor.submit(
                    self._process_clip_wrapper,
                    processor, i, data, total_clips,
                    source_path, all_words, config, temp_dir, output_dir
                )
                futures[future] = i
            
            # Wait for all tasks to complete or cancellation
            for future in as_completed(futures):
                if self.cancel_flag.is_set():
                    self._safe_log("WARNING", "Cancellation requested, stopping remaining clips...")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                # Get result to catch any exceptions
                try:
                    future.result()
                except Exception as e:
                    clip_idx = futures[future]
                    self._safe_log("ERROR", f"Unexpected error in clip {clip_idx+1}: {str(e)[:60]}")
        
        return self.completed_count
    
    def cancel(self):
        """Signal all threads to stop processing"""
        self.cancel_flag.set()
        self._safe_log("WARNING", "Cancellation signal sent to all workers")
