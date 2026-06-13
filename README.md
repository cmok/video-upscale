# Video Upscaler CLI Tool (`upscale.py`)

A premium, easy-to-use command-line utility designed to upscale low-resolution videos (such as old Video CD `.DAT`/`.MPG` files or retro video game captures). 

The tool is optimized for **macOS Apple Silicon** (M-series chips), automatically leveraging GPU hardware acceleration (`VideoToolbox`) to deliver blazing-fast upscaling speeds.

---

## Features

*   **Metadata Probing (`info` command)**: Analyze input files to show codecs, duration, aspect ratios, and get recommended upscale resolutions.
*   **Aspect Ratio Correction**: Automatically handles aspect ratio correction for non-square pixel formats (like PAL/NTSC VCDs), scaling to standard square-pixel 4:3 or 16:9 formats to prevent stretched/squished display.
*   **Multiple Scaling Engines**:
    *   `lanczos`: Fastest high-quality mathematical scaling.
    *   `hqx` (2x, 3x, 4x): Pixel art magnification, preserving extremely sharp edges.
    *   `xbr` (2x, 3x, 4x): Vector-like curves smoothing, ideal for animations and drawings.
    *   `epx` / `super2xsai`: Classic retro-game magnification filters.
    *   `fsrcnn` (2x, 3x, 4x): Fast Super-Resolution Convolutional Neural Network (AI) model running on OpenCV DNN, restoring realistic textures and details.
*   **Snippet/Test Mode (`--trim`)**: Easily trim a short segment (e.g. 5 seconds) to check quality configurations before running a full-length upscale.
*   **Split-Screen Comparison (`--compare`)**: Generate a side-by-side video comparing the original video (scaled bilinearly) with the upscaled video, complete with custom text overlays.
*   **4-Way Grid Comparison (`compare-all` command)**: Create a grid video comparing Original, Bilinear, Lanczos, and AI FSRCNN side-by-side to choose the best configuration.
*   **Interactive Guided Setup**: Running the script without arguments starts a step-by-step visual configuration wizard.

---

## Prerequisites

The tool requires **FFmpeg** and **FFprobe** installed on your system.

### macOS Installation
```bash
brew install ffmpeg
```

---

## How to Run

You do not need to manually install any Python packages. The tool uses `uv` to manage dependencies automatically on the fly.

### 1. Probe Video Information
Inspect the streams, overall bitrate, and recommended upscale resolutions:
```bash
uv run upscale.py info input-videos/AVSEQ01.DAT
```

### 2. Guided Setup (Interactive Wizard)
Run the script without options to launch the interactive configuration wizard:
```bash
uv run upscale.py
```

### 3. Upscale a Video
Upscale using a built-in filter (e.g., `hqx` at 4x scale):
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o output_hqx_4x.mp4 --engine hqx --scale 4
```

### 4. Create a Side-by-Side Comparison Snippet
Generate a 10-second preview comparison video using the AI FSRCNN model:
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o fsrcnn_compare.mp4 --engine fsrcnn --scale 2 --trim 10 --compare
```

### 5. Generate a 4-Way Comparison Grid Video
Create a 5-second 4-way comparison video grid (Original vs. Bilinear vs. Lanczos vs. AI FSRCNN) at 2x scale:
```bash
uv run upscale.py compare-all input-videos/AVSEQ01.DAT -o grid_compare.mp4 --trim 5 --scale 2
```

---

## Upscale Parameter Reference

| Parameter | Alias | Description | Choices / Format | Default |
| :--- | :--- | :--- | :--- | :--- |
| `--output` | `-o` | Output file destination path | File path | `[input_name]_[engine]_[scale]x.mp4` |
| `--engine` | `-e` | Upscaling filter/AI backend | `lanczos`, `hqx`, `xbr`, `epx`, `super2xsai`, `fsrcnn` | `hqx` |
| `--scale` | `-s` | Scale factor | `2`, `3`, `4` | `4` |
| `--trim` | `-t` | Segment to upscale | `duration_seconds` or `start_seconds,duration_seconds` | Full Video |
| `--codec` | `-c` | Video encoding format | `h264`, `hevc` (H.265) | `h264` |
| `--aspect` | `-a` | Aspect ratio handling | `auto`, `source`, `4:3`, `16:9` | `auto` |
| `--audio` | `-d` | Audio transcode operation | `copy`, `aac`, `none` | `aac` |
| `--compare` | | Generate a side-by-side comparison | Flag | `False` |

---

## Engine Selection Guide

*   **FSRCNN (AI Model)**: Best for real-life captured footages (e.g., old family videos, movies). It resolves details and produces a natural look.
*   **hqx / xbr**: Best for pixel art, hand-drawn cartoons, anime, and computer-generated screens. It keeps borders sharp and eliminates pixelation.
*   **Lanczos**: Best for high-speed, general-purpose mathematical scaling when AI/pixel filters are not desired.
