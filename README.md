# Video Upscaler CLI Tool (`upscale.py`)

A premium, easy-to-use command-line utility designed to upscale low-resolution videos (such as old Video CD `.DAT`/`.MPG` files or retro video game captures). 

The tool is optimized for **macOS Apple Silicon** (M-series chips), automatically leveraging GPU hardware acceleration (`VideoToolbox` for encoding, and NCNN/Vulkan/Metal for Real-ESRGAN) to deliver blazing-fast upscaling speeds.

---

## Features

*   **Metadata Probing (`info` command)**: Analyze input files to show codecs, duration, aspect ratios, and get recommended upscale resolutions.
*   **Aspect Ratio Correction**: Automatically handles aspect ratio correction for non-square pixel formats (like PAL/NTSC VCDs), scaling to standard square-pixel 4:3 or 16:9 formats to prevent stretched/squished display.
*   **Multiple Scaling Engines**:
    *   `realesrgan` (2x, 3x, 4x): Premium state-of-the-art AI Super-Resolution running natively on the GPU (Vulkan/Metal).
    *   `fsrcnn` (2x, 3x, 4x): Fast Super-Resolution Convolutional Neural Network (AI) model running on OpenCV DNN, restoring realistic textures and details.
    *   `lanczos`: Fastest high-quality mathematical scaling.
    *   `hqx` (2x, 3x, 4x): Pixel art magnification, preserving extremely sharp edges.
    *   `xbr` (2x, 3x, 4x): Vector-like curves smoothing, ideal for animations and drawings.
    *   `epx` / `super2xsai`: Classic retro-game magnification filters.
*   **Snippet/Test Mode (`--trim`)**: Easily trim a short segment (e.g. 5 seconds) to check quality configurations before running a full-length upscale.
*   **Split-Screen Comparison (`--compare`)**: Generate a side-by-side video comparing the original video (scaled bilinearly) with the upscaled video.
*   **4-Way Grid Comparison (`compare-all` command)**: Create a grid video comparing Original (Bilinear), Lanczos, AI FSRCNN, and Premium Real-ESRGAN side-by-side to choose the best configuration.
*   **Interactive Guided Setup**: Running the script without arguments starts a step-by-step visual configuration wizard.

---

## Prerequisites

The tool requires **FFmpeg** and **FFprobe** installed on your system.

### macOS Installation
```bash
brew install ffmpeg
```

> [!NOTE]
> **Real-ESRGAN**: The script will automatically download the macOS universal Vulkan binary and pre-trained models (~49 MB) on its first execution and strip macOS Gatekeeper quarantine tags. No manual installation is required.

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
Upscale using the premium GPU-accelerated Real-ESRGAN engine:
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o output_realesrgan.mp4 --engine realesrgan --model realesr-animevideov3 --scale 4
```

Upscale using a lightweight filter (e.g., `hqx` at 4x scale):
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o output_hqx_4x.mp4 --engine hqx --scale 4
```

### 4. Create a Side-by-Side Comparison Snippet
Generate a 10-second preview comparison video using Real-ESRGAN:
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o realesrgan_compare.mp4 --engine realesrgan --scale 4 --trim 10 --compare
```

### 5. Generate a 4-Way Comparison Grid Video
Create a 5-second 4-way comparison video grid (Original Bilinear vs. Lanczos vs. AI FSRCNN vs. Real-ESRGAN) at 2x scale:
```bash
uv run upscale.py compare-all input-videos/AVSEQ01.DAT -o grid_compare.mp4 --trim 5 --scale 2
```

---

## Upscale Parameter Reference

| Parameter | Alias | Description | Choices / Format | Default |
| :--- | :--- | :--- | :--- | :--- |
| `--output` | `-o` | Output file destination path | File path | `[input_name]_[engine]_[scale]x.mp4` |
| `--engine` | `-e` | Upscaling filter/AI backend | `realesrgan`, `fsrcnn`, `lanczos`, `hqx`, `xbr`, `epx`, `super2xsai` | `hqx` |
| `--model` | `-m` | Real-ESRGAN model choice | `realesr-animevideov3`, `realesrgan-x4plus`, `realesrgan-x4plus-anime` | `realesr-animevideov3` |
| `--scale` | `-s` | Scale factor | `2`, `3`, `4` | `4` |
| `--trim` | `-t` | Segment to upscale | `duration_seconds` or `start_seconds,duration_seconds` | Full Video |
| `--codec` | `-c` | Video encoding format | `h264`, `hevc` (H.265) | `h264` |
| `--aspect` | `-a` | Aspect ratio handling | `auto`, `source`, `4:3`, `16:9` | `auto` |
| `--audio` | `-d` | Audio transcode operation | `copy`, `aac`, `none` | `aac` |
| `--compare` | | Generate a side-by-side comparison | Flag | `False` |
| `--quality` | `-q` | Encoder quality factor | `0-100` (HW) or `0-51` (CRF) | `65` (HW) / `18` (SW) |
| `--denoise` | `-dn` | Apply spatial/temporal denoiser | Flag | `False` |
| `--deblock` | `-db` | Apply MPEG deblocking filter | Flag | `False` |
| `--sharpen` | `-sp` | Apply unsharp mask sharpening | Flag | `False` |

---

## Quality Fine-Tuning Examples

For low-resolution Video CDs (`AVSEQ01.DAT`) with significant noise and macroblocks:

### 1. High-Quality AI Upscaling with Denoising & Sharpening
Smooth compression blocks and grain, upscale with Real-ESRGAN, and apply an unsharp mask for fine-detail restoration:
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o output_ai_fine.mp4 --engine realesrgan --scale 4 --denoise --deblock --sharpen --trim 10
```

### 2. High-Speed hqx Upscaling with Deblocking & Sharpening
Clean MPEG compression blocks, scale using C-based `hqx`, and sharpen boundaries:
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o output_hqx_fine.mp4 --engine hqx --scale 4 --deblock --sharpen --trim 10
```

### 3. Ultra-Quality HEVC / H.265 Encode
Upscale and encode using HEVC at quality factor 75 (instead of default 65) for pristine, high-fidelity results:
```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o output_hqx_hevc.mp4 --engine hqx --scale 4 --codec hevc --quality 75 --trim 10
```

---

## Engine Selection Guide

*   **Real-ESRGAN (Premium GPU AI)**: State-of-the-art AI upscaling. Best for low-resolution animations (`realesr-animevideov3` or `realesrgan-x4plus-anime`) and real-world videos (`realesrgan-x4plus`). Natively GPU accelerated on Apple Silicon.
*   **FSRCNN (AI Model)**: Lightweight, CPU/GPU-friendly AI model. Excellent quality for general real-world scenes.
*   **hqx / xbr**: Best for pixel art, hand-drawn cartoons, and computer-generated screens. It keeps borders sharp and eliminates pixelation.
*   **Lanczos**: Best for high-speed, general-purpose mathematical scaling when AI/pixel filters are not desired.

---

## Pro-Tip for Real-Life Videos

Since low-resolution real-life videos (such as PAL/NTSC VCD `.DAT` source files) often suffer from analog noise and digital MPEG compression blocking, combining **`realesrgan-x4plus`** with pre-processing deblocking and denoising yields pristine, highly restored outputs:

```bash
uv run upscale.py upscale input-videos/AVSEQ01.DAT -o output_real_clean.mp4 --engine realesrgan --model realesrgan-x4plus --scale 4 --deblock --denoise --sharpen
```

The `--deblock` and `--denoise` filters clean up macroblocks and compression artifacts *before* the frame goes into the AI Super-Resolution network, preventing the model from upscaling and amplifying source artifacts.

