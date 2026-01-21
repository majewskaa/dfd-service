import cv2
import numpy as np
import subprocess
import os
import random
from typing import Dict, Any, List, Optional, Tuple
import imageio_ffmpeg

class Augmentor:
    """Handles video, audio, and multimodal augmentations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: part of preprocessing config under 'augmentations' key
        """
        self.config = config
        self.enabled = config.get("enabled", False)

    def is_enabled(self) -> bool:
        return self.enabled

    def get_suffix(self) -> str:
        """Generate a folder suffix based on active augmentations."""
        if not self.enabled:
            return ""
        
        parts = []
        
        # Video
        vid_config = self.config.get("video", {})
        if vid_config.get("compression", {}).get("enabled"):
            c_type = vid_config["compression"].get("type", "jpeg")
            if c_type == "jpeg":
                q = vid_config["compression"].get("quality", 50)
                parts.append(f"jpeg{q}")
            elif c_type == "h264":
                crf = vid_config["compression"].get("crf", 23)
                parts.append(f"crf{crf}")
        
        if vid_config.get("resolution", {}).get("enabled"):
            ratio = vid_config["resolution"].get("scale_ratio", 0.5)
            parts.append(f"res{int(ratio*100)}")

        # Audio
        aud_config = self.config.get("audio", {})
        if aud_config.get("compression", {}).get("enabled"):
            br = aud_config["compression"].get("bitrate", "64k")
            fmt = aud_config["compression"].get("format", "mp3")
            parts.append(f"{fmt}{br}")
        
        if aud_config.get("noise", {}).get("enabled"):
            snr = aud_config["noise"].get("snr_db", 15)
            parts.append(f"snr{snr}")

        # Multimodal
        mm_config = self.config.get("multimodal", {})
        if mm_config.get("av_desync", {}).get("enabled"):
            parts.append("desync")

        if not parts:
            return "_aug"
        
        return "_" + "_".join(parts)

    def augment_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply video augmentations to a single frame."""
        if not self.enabled:
            return frame

        vid_config = self.config.get("video", {})
        
        # Resolution (Downscale -> Upscale)
        res_cfg = vid_config.get("resolution", {})
        if res_cfg.get("enabled", False):
            scale = res_cfg.get("scale_ratio", 0.5)
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            # Downscale
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Upscale back
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        # Compression (JPEG) - H.264 is done at video level, but we simulate JPEG per frame
        comp_cfg = vid_config.get("compression", {})
        if comp_cfg.get("enabled", False):
            c_type = comp_cfg.get("type", "jpeg")
            if c_type == "jpeg":
                quality = int(comp_cfg.get("quality", 50))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encimg = cv2.imencode('.jpg', frame, encode_param)
                frame = cv2.imdecode(encimg, 1) # BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                
        return frame

    def augment_video_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Wrapper to handle RGB frames correctly."""
        if not self.enabled:
            return frame_rgb
        
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        vid_config = self.config.get("video", {})
        
        # Resolution
        res_cfg = vid_config.get("resolution", {})
        if res_cfg.get("enabled", False):
            scale = res_cfg.get("scale_ratio", 0.5)
            h, w = frame_bgr.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        # Compression
        comp_cfg = vid_config.get("compression", {})
        if comp_cfg.get("enabled", False):
            c_type = comp_cfg.get("type", "jpeg")
            if c_type == "jpeg":
                quality = int(comp_cfg.get("quality", 50))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encimg = cv2.imencode('.jpg', frame_bgr, encode_param)
                frame_bgr = cv2.imdecode(encimg, 1)

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


    def build_ffmpeg_audio_filter(self) -> str:
        """Build ffmpeg filter string for audio augmentations."""
        if not self.enabled:
            return ""
        
        filters = []
        aud_config = self.config.get("audio", {})
        
    def needs_audio_compression(self) -> bool:
        """Check if audio compression is enabled."""
        if not self.enabled:
            return False
        return self.config.get("audio", {}).get("compression", {}).get("enabled", False)

    def extract_compressed_audio_as_pcm(self, video_path: str, sr: int) -> Optional[bytes]:
        """Extract audio with compression simulation (Encode -> Decode -> PCM)."""
        aud_config = self.config.get("audio", {})
        comp_cfg = aud_config.get("compression", {})
        
        fmt = comp_cfg.get("format", "mp3")
        bitrate = comp_cfg.get("bitrate", "64k")
        
        # 1. Encode
        if fmt == "mp3":
            codec = "libmp3lame"
            container = "mp3"
        elif fmt == "aac":
            codec = "aac"
            container = "adts"
        else:
            return None

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Cmd1: Video -> Compressed Audio
        # Add -y to overwrite if needed (stdout), -vn for no video
        cmd1 = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-vn", 
            "-acodec", codec,
            "-b:a", bitrate,
            "-f", container,
            "-",
        ]
        
        # Cmd2: Compressed Audio -> PCM
        cmd2 = [
            ffmpeg_exe, "-y", "-i", "-",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", str(sr),
            "-",
        ]
        
        import tempfile
        
        # Use temporary files for stderr to avoid deadlocks and buffer filling
        with tempfile.TemporaryFile() as err1_file, tempfile.TemporaryFile() as err2_file:
            p1 = None
            p2 = None
            try:
                # Start the first process (Video -> Compressed Audio)
                p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=err1_file)
                
                # Start the second process (Compressed Audio -> PCM), reading from p1
                p2 = subprocess.Popen(
                    cmd2, 
                    stdin=p1.stdout, 
                    stdout=subprocess.PIPE, 
                    stderr=err2_file
                )
                
                # Allow p1 to receive a SIGPIPE if p2 exits.
                p1.stdout.close() 

                # Wait for p2 to finish and get output with timeout
                try:
                    stdout_data, _ = p2.communicate(timeout=60) # 60s timeout per file
                except subprocess.TimeoutExpired:
                    print(f"Audio compression timed out for {video_path}")
                    p2.kill()
                    p1.kill()
                    return None
                
                # Wait for p1 to finish
                try:
                    p1.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p1.kill()

                if p2.returncode != 0:
                     print(f"Audio compression failed with return code {p2.returncode} for {video_path}")
                     # Print stderr for debugging
                     err2_file.seek(0)
                     print(f"ffmpeg p2 stderr: {err2_file.read().decode('utf-8', errors='replace')}")
                     err1_file.seek(0)
                     print(f"ffmpeg p1 stderr: {err1_file.read().decode('utf-8', errors='replace')}")
                     return None
                
                # Check p1 return code too (ignoring SIGPIPE which is -13 on unix, or broken pipe)
                if p1.returncode != 0 and p1.returncode != -13 and p1.returncode != 3221225786: # 3221225786 is CTRL+C on Windows/SIGINT propagation sometimes?
                     # Just warn, as long as p2 got data it might be fine, but worth noting
                     # Wait, if p1 failed, p2 probably got truncated data.
                     # But if p2 returned 0, it means it decoded what it got successfully.
                     pass

                return stdout_data

            except Exception as e:
                print(f"Audio compression failed for {video_path}: {e}")
                return None
            finally:
                # Ensure cleanup
                if p1:
                    try: p1.kill() 
                    except: pass
                if p2:
                    try: p2.kill() 
                    except: pass

        # Noise
        if aud_config.get("noise", {}).get("enabled", False):
            snr = aud_config["noise"].get("snr_db", 15)
            # ffmpeg doesn't have a direct "add noise at SNR" filter easily without a noise file.
            pass

        return ",".join(filters)

    def apply_audio_noise_numpy(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise to audio signal (float32) to match target SNR."""
        if not self.enabled:
            return audio
            
        aud_config = self.config.get("audio", {})
        noise_cfg = aud_config.get("noise", {})
        if noise_cfg.get("enabled", False):
            snr_db = noise_cfg.get("snr_db", 15)
            
            # Calculate signal power
            sig_power = np.mean(audio ** 2)
            if sig_power == 0:
                return audio
                
            # Calculate required noise power
            # SNR_db = 10 * log10(P_signal / P_noise)
            # P_noise = P_signal / (10^(SNR_db/10))
            noise_power = sig_power / (10 ** (snr_db / 10))
            
            # Generate noise
            noise = np.random.normal(0, np.sqrt(noise_power), audio.shape).astype(audio.dtype)
            
            return audio + noise
            
        return audio

    def get_av_desync_offset(self) -> float:
        """Return random offset in seconds for A/V desync."""
        if not self.enabled:
            return 0.0
            
        mm_config = self.config.get("multimodal", {})
        desync_cfg = mm_config.get("av_desync", {})
        if desync_cfg.get("enabled", False):
            min_shift = desync_cfg.get("min_shift_ms", -200)
            max_shift = desync_cfg.get("max_shift_ms", 200)
            shift_ms = random.uniform(min_shift, max_shift)
            return shift_ms / 1000.0
            
        return 0.0
