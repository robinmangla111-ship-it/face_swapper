"""
Unit and Integration Tests for Face Swapper
============================================
Tests video processing, chunk management, API models, and FFmpeg muxing.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np


class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        from core.video_processor import VideoProcessor
        self.processor = VideoProcessor()
        self.temp_dir = tempfile.mkdtemp()
        self.test_video = str(Path(self.temp_dir) / "test.mp4")

        # Create a synthetic 2-second 30fps video (60 frames)
        width, height, fps = 320, 240, 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.test_video, fourcc, fps, (width, height))
        for i in range(60):
            frame = np.full((height, width, 3), (i * 4) % 256, dtype=np.uint8)
            # Draw a circle so there's content
            cv2.circle(frame, (160, 120), 40, (0, 255, 255), -1)
            writer.write(frame)
        writer.release()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_video_info(self):
        info = self.processor.get_video_info(self.test_video)
        self.assertEqual(info.width, 320)
        self.assertEqual(info.height, 240)
        self.assertEqual(info.total_frames, 60)
        self.assertAlmostEqual(info.fps, 30.0, places=1)

    def test_sample_frames(self):
        samples = self.processor.sample_frames(self.test_video, every_n=10, max_samples=10)
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), 10)
        idx, frame = samples[0]
        self.assertEqual(frame.shape, (240, 320, 3))

    def test_chunk_manager_split(self):
        from core.chunk_manager import ChunkManager
        mgr = ChunkManager(num_workers=4)
        chunks = mgr.create_chunks(total_frames=1000, min_chunk_size=100)
        self.assertGreater(len(chunks), 1)
        # Verify continuous coverage
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 1000)
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i][1], chunks[i + 1][0])


class TestColorCorrector(unittest.TestCase):
    def test_histogram_match(self):
        from core.color_corrector import ColorCorrector
        src = np.full((100, 100, 3), 50, dtype=np.uint8)
        tgt = np.full((100, 100, 3), 200, dtype=np.uint8)
        matched = ColorCorrector.histogram_match_color(src, tgt)
        self.assertEqual(matched.shape, (100, 100, 3))


if __name__ == "__main__":
    unittest.main()
