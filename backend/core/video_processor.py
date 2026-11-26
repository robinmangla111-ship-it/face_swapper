"""
Video Processing Module
========================
Handles video I/O operations using FFmpeg and OpenCV.
Extracts audio, reads frames, writes video, and muxes final output.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Metadata about a video file."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float          # seconds
    codec: str
    has_audio: bool
    audio_codec: Optional[str]
    file_size_mb: float

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration": round(self.duration, 2),
            "codec": self.codec,
            "has_audio": self.has_audio,
            "file_size_mb": round(self.file_size_mb, 2),
        }


class FFmpegVideoWriter:
    """Direct uncompressed rawvideo pipe to FFmpeg libx264 for visually lossless output."""
    def __init__(self, output_path: str, width: int, height: int, fps: float,
                 crf: int = 15, preset: str = "ultrafast"):
        from config import FFMPEG_PATH
        self.output_path = output_path
        self.width = width
        self.height = height
        self.cmd = [
            FFMPEG_PATH, "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def write(self, frame: np.ndarray):
        if self.proc and self.proc.stdin:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.proc.stdin.write(frame.tobytes())

    def release(self):
        if self.proc:
            if self.proc.stdin:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def create_video_writer(output_path: str, width: int, height: int, fps: float, crf: int = 15):
    """
    Creates an optimal video writer. Prefers FFmpegVideoWriter for pristine H.264 quality,
    falling back to OpenCV VideoWriter if FFmpeg is unavailable.
    """
    try:
        from config import FFMPEG_PATH
        if FFMPEG_PATH:
            writer = FFmpegVideoWriter(output_path, width, height, fps, crf=crf)
            if writer.isOpened():
                return writer
    except Exception as e:
        logger.warning(f"FFmpegVideoWriter init failed ({e}), falling back to cv2.VideoWriter")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


class VideoProcessor:
    """
    Handles video file operations using FFmpeg and OpenCV.
    """

    def __init__(self, ffmpeg_path: Optional[str] = None):
        from config import FFMPEG_PATH
        self._ffmpeg = ffmpeg_path or FFMPEG_PATH
        if not self._ffmpeg:
            raise RuntimeError(
                "FFmpeg not found. Install with: pip install imageio-ffmpeg"
            )

    def get_video_info(self, video_path: str) -> VideoInfo:
        """Extract metadata from a video file."""
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        duration = total_frames / fps if fps > 0 else 0
        file_size = Path(video_path).stat().st_size / (1024 * 1024)

        # Check audio with ffprobe
        has_audio = False
        audio_codec = None
        try:
            result = subprocess.run(
                [self._ffmpeg.replace("ffmpeg", "ffprobe") if "ffprobe" not in self._ffmpeg
                 else self._ffmpeg,
                 "-v", "quiet", "-print_format", "json",
                 "-show_streams", video_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        has_audio = True
                        audio_codec = stream.get("codec_name", "unknown")
                        break
        except Exception:
            # Fallback: try with ffmpeg
            has_audio = self._check_audio_ffmpeg(video_path)

        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            has_audio=has_audio,
            audio_codec=audio_codec,
            file_size_mb=file_size,
        )

    def _check_audio_ffmpeg(self, video_path: str) -> bool:
        """Check if video has audio using FFmpeg."""
        try:
            result = subprocess.run(
                [self._ffmpeg, "-i", video_path, "-af", "anull",
                 "-f", "null", "-", "-hide_banner"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
            )
            return "Audio:" in result.stderr
        except Exception:
            return False

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Extract audio stream from video without re-encoding.

        Args:
            video_path: Input video file path
            output_path: Output audio file path. Auto-generated if None.

        Returns:
            Path to extracted audio file, or None if no audio.
        """
        video_info = self.get_video_info(video_path)
        if not video_info.has_audio:
            logger.info("Video has no audio track")
            return None

        if output_path is None:
            from config import TEMP_DIR
            output_path = str(TEMP_DIR / f"audio_{Path(video_path).stem}.aac")

        logger.info(f"Extracting audio from {video_path}")

        cmd = [
            self._ffmpeg, "-y",
            "-i", str(video_path),
            "-vn",                   # No video
            "-acodec", "copy",       # Copy audio codec (no re-encoding)
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300
            )
            if result.returncode != 0:
                # Retry with re-encoding to AAC
                cmd[-2] = "aac"
                cmd.insert(-1, "-b:a")
                cmd.insert(-1, "192k")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300
                )

            if result.returncode == 0 and Path(output_path).exists():
                logger.info(f"Audio extracted: {output_path}")
                return output_path
            else:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return None

    def read_frames(self, video_path: str,
                    start_frame: int = 0,
                    end_frame: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields (frame_index, frame) tuples.

        Args:
            video_path: Path to video file
            start_frame: First frame to read (inclusive)
            end_frame: Last frame to read (exclusive). None = read to end.

        Yields:
            (frame_index, frame_bgr) tuples
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while True:
            if end_frame is not None and frame_idx >= end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            yield frame_idx, frame
            frame_idx += 1

        cap.release()

    def sample_frames(self, video_path: str,
                      every_n: int = 30,
                      max_samples: int = 100) -> List[Tuple[int, np.ndarray]]:
        """
        Sample evenly-spaced frames from a video.

        Args:
            video_path: Path to video file
            every_n: Take every Nth frame
            max_samples: Maximum number of frames to sample

        Returns:
            List of (frame_index, frame_bgr) tuples
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Adjust sampling rate if video is short
        if total_frames <= max_samples:
            every_n = 1

        samples = []
        frame_idx = 0

        while frame_idx < total_frames and len(samples) < max_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            samples.append((frame_idx, frame))
            frame_idx += every_n

        cap.release()
        logger.info(f"Sampled {len(samples)} frames from {total_frames} total")
        return samples

    def write_video_from_frames(self, frames_generator,
                                 output_path: str,
                                 fps: float,
                                 width: int,
                                 height: int,
                                 codec: str = "mp4v") -> str:
        """
        Write frames to a video file using OpenCV VideoWriter.

        Args:
            frames_generator: Iterable yielding BGR frames (np.ndarray)
            output_path: Output video file path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: FourCC codec string

        Returns:
            Path to written video file
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create VideoWriter: {output_path}")

        frame_count = 0
        for frame in frames_generator:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frame_count += 1

        writer.release()
        logger.info(f"Wrote {frame_count} frames to {output_path}")
        return output_path

    def merge_video_audio(self, video_path: str, audio_path: Optional[str],
                          output_path: str) -> str:
        """
        Mux video and audio streams into final output using FFmpeg.
        Re-encodes video with high quality H.264 and copies audio.

        Args:
            video_path: Input video file (processed, no audio)
            audio_path: Input audio file (extracted from original)
            output_path: Final output file path

        Returns:
            Path to output file
        """
        from config import VIDEO_CODEC, VIDEO_CRF, VIDEO_PRESET, PIXEL_FORMAT

        cmd = [self._ffmpeg, "-y"]

        # Input: processed video
        cmd.extend(["-i", str(video_path)])

        if audio_path and Path(audio_path).exists():
            # Input: original audio
            cmd.extend(["-i", str(audio_path)])
            # Map: video from first input, audio from second
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
            cmd.extend(["-c:a", "copy"])     # Copy audio (no re-encode)
        else:
            cmd.extend(["-map", "0:v:0"])

        # Video encoding settings: studio quality CRF 15 with 6M bitrate floor
        cmd.extend([
            "-c:v", VIDEO_CODEC,
            "-crf", str(VIDEO_CRF),
            "-preset", VIDEO_PRESET,
            "-pix_fmt", PIXEL_FORMAT,
            "-b:v", "5M",
            "-maxrate", "8M",
            "-bufsize", "10M",
            "-movflags", "+faststart",        # Web-friendly MP4
        ])

        cmd.append(str(output_path))

        logger.info(f"Muxing video+audio → {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg mux failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg mux failed: {result.stderr[:500]}")

        if not Path(output_path).exists():
            raise RuntimeError("Output file was not created")

        logger.info(f"Final output: {output_path} "
                     f"({Path(output_path).stat().st_size / (1024*1024):.1f} MB)")
        return output_path

    def concat_videos(self, video_paths: List[str], output_path: str) -> str:
        """
        Concatenate multiple video files using FFmpeg concat demuxer.

        Args:
            video_paths: List of video file paths in order
            output_path: Output concatenated video path

        Returns:
            Path to concatenated video
        """
        from config import TEMP_DIR

        # Create concat file list
        concat_list = TEMP_DIR / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for vp in video_paths:
                f.write(f"file '{Path(vp).resolve().as_posix()}'\n")

        cmd = [
            self._ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)

        if result.returncode != 0:
            logger.error(f"Concat failed: {result.stderr}")
            raise RuntimeError(f"Video concatenation failed: {result.stderr[:500]}")

        logger.info(f"Concatenated {len(video_paths)} chunks → {output_path}")
        return output_path

    def get_frame_at(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Get a single frame from a video by index."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
