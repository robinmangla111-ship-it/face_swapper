---
title: Face Swapper
emoji: 🎭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Swap characters in videos with free GPU
---

# 🎭 Face Swapper

Upload a video, automatically detect all characters/faces, choose which one to replace, and swap them with any face image.

## Features
- Automatic character clustering and identification
- Preserves original video audio and timing without re-encoding
- Free dynamic GPU powered by ZeroGPU (`@spaces.GPU`)
- High-quality face blending and color correction
