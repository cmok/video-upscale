#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.1.0",
#     "rich>=13.7.0",
#     "opencv-contrib-python>=4.9.0.80",
#     "numpy>=1.26.0",
# ]
# ///

import os
import sys
import shutil
import subprocess
import json
import math
import urllib.request
import tempfile
from pathlib import Path
import click
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, ProgressColumn
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

# Initialize Rich Console
console = Console()

class FPSColumn(ProgressColumn):
    """Custom progress column to show processing speed in frames per second (fps)."""
    def render(self, task):
        speed = task.speed
        if speed is None:
            return Text("? fps", style="progress.data.speed")
        return Text(f"{speed:.1f} fps", style="progress.data.speed")

def check_requirements():
    """Checks if ffmpeg and ffprobe are available on the path."""
    for prog in ['ffmpeg', 'ffprobe']:
        if not shutil.which(prog):
            console.print(f"[bold red]Error:[/bold red] {prog} is not installed or not in your PATH.", style="red")
            console.print("Please install FFmpeg to use this tool (e.g. [cyan]brew install ffmpeg[/cyan] on macOS).")
            sys.exit(1)

def get_font_file():
    """Locates a font file on the system for ffmpeg drawtext comparison labels."""
    paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def parse_time_str(time_str):
    """Converts HH:MM:SS, MM:SS, or SS.S to float seconds."""
    if not time_str:
        return 0.0
    parts = time_str.split(':')
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
        else:
            raise ValueError()
    except ValueError:
        raise click.BadParameter(f"Invalid time format: '{time_str}'. Use HH:MM:SS, MM:SS, or seconds.")

def parse_trim(trim_str, total_duration):
    """Parses a trim string (e.g., '10' or '60,10' or '00:01:00,10') into (start_sec, duration_sec)."""
    if not trim_str:
        return 0.0, None
        
    if ',' in trim_str:
        start_part, dur_part = trim_str.split(',', 1)
        start_sec = parse_time_str(start_part.strip())
        duration_sec = parse_time_str(dur_part.strip())
    else:
        # Just a duration from the start of the video
        start_sec = 0.0
        duration_sec = parse_time_str(trim_str.strip())
        
    if start_sec >= total_duration:
        raise click.BadParameter(f"Start time ({start_sec}s) is greater than video duration ({total_duration:.1f}s).")
        
    if duration_sec is not None:
        if start_sec + duration_sec > total_duration:
            duration_sec = total_duration - start_sec
            
    return start_sec, duration_sec

def calculate_target_dims(width, height, sar, scale, aspect_mode):
    """Calculates target dimensions correcting aspect ratio if needed, keeping them even numbers."""
    if aspect_mode == 'source':
        # Simple pixel scaling
        w = int(width * scale)
        h = int(height * scale)
    elif aspect_mode == 'auto':
        # Correct if SAR is not 1:1, otherwise scale directly
        if sar and sar != "1:1":
            try:
                sar_num, sar_den = map(float, sar.split(':'))
                sar_val = sar_num / sar_den
            except Exception:
                sar_val = 1.0
            h = int(height * scale)
            w = int(h * (width / height) * sar_val)
        else:
            w = int(width * scale)
            h = int(height * scale)
    elif ':' in aspect_mode:
        # Custom aspect ratio like '4:3' or '16:9'
        try:
            dar_num, dar_den = map(float, aspect_mode.split(':'))
            dar_val = dar_num / dar_den
            h = int(height * scale)
            w = int(h * dar_val)
        except Exception:
            w = int(width * scale)
            h = int(height * scale)
    else:
        w = int(width * scale)
        h = int(height * scale)
        
    # ffmpeg requires even dimensions for H.264/HEVC encoding
    w = int(math.ceil(w / 2.0) * 2)
    h = int(math.ceil(h / 2.0) * 2)
    return w, h

def get_video_info(input_path):
    """Probes the input video using ffprobe and returns (video_stream, audio_stream, format_info)."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_format', '-show_streams',
            '-of', 'json', input_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = None
        audio_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and not video_stream:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and not audio_stream:
                audio_stream = stream
                
        format_info = data.get('format', {})
        return video_stream, audio_stream, format_info
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error probing video file with ffprobe:[/bold red]")
        console.print(e.stderr)
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error probing video:[/bold red] {e}")
        sys.exit(1)

def get_codec_params(codec_choice, is_mac, custom_quality=None):
    """Returns the appropriate encoder and quality options."""
    if codec_choice == 'h264':
        if is_mac:
            q = str(custom_quality) if custom_quality is not None else '65'
            return 'h264_videotoolbox', ['-q:v', q]
        else:
            q = str(custom_quality) if custom_quality is not None else '18'
            return 'libx264', ['-crf', q, '-preset', 'slow']
    elif codec_choice == 'hevc':
        if is_mac:
            q = str(custom_quality) if custom_quality is not None else '65'
            return 'hevc_videotoolbox', ['-q:v', q]
        else:
            q = str(custom_quality) if custom_quality is not None else '20'
            return 'libx265', ['-crf', q, '-preset', 'medium']
    q = str(custom_quality) if custom_quality is not None else '18'
    return 'libx264', ['-crf', q, '-preset', 'slow']

def download_fsrcnn_model(scale):
    """Downloads FSRCNN .pb model file and caches it locally."""
    cache_dir = Path("./.cache/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"FSRCNN_x{scale}.pb"
    model_path = cache_dir / filename
    
    if model_path.exists():
        return str(model_path)
        
    # Primary and backup URLs (Saafke is standard, thinkerleolee is backup)
    urls = [
        f"https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x{scale}.pb",
        f"https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x{scale}.pb",
        f"https://github.com/thinkerleolee/FSRCNN-OpenCV/raw/master/models/fsrcnn_x{scale}.pb"
    ]
    
    console.print(f"[yellow]Downloading FSRCNN {scale}x model...[/yellow]")
    for dl_url in urls:
        try:
            with urllib.request.urlopen(dl_url, timeout=15) as response, open(model_path, 'wb') as out_file:
                out_file.write(response.read())
            console.print(f"[green]✓ Cached model to {model_path}[/green]")
            return str(model_path)
        except Exception as e:
            console.print(f"[red]Failed to download from {dl_url}: {e}[/red]")
            
    console.print("[bold red]Error:[/bold red] Could not download the FSRCNN model file.")
    console.print("Please download it manually and place it in [cyan].cache/models/[/cyan].")
    sys.exit(1)

def get_ffmpeg_filter(engine, scale, target_width, target_height, deblock=False, denoise=False, sharpen=False):
    """Builds the FFmpeg filter string for the chosen engine, including pre/post quality enhancement filters."""
    filters = []
    if deblock:
        filters.append("spp")
    if denoise:
        filters.append("hqdn3d=1.5:1.5:6:6")
        
    engine_filter = ""
    if engine == 'lanczos':
        engine_filter = f"scale={target_width}:{target_height}:flags=lanczos"
    elif engine == 'hqx':
        engine_filter = f"hqx=n={scale},scale={target_width}:{target_height}:flags=lanczos"
    elif engine == 'xbr':
        engine_filter = f"xbr=n={scale},scale={target_width}:{target_height}:flags=lanczos"
    elif engine == 'epx':
        # epx only supports 2 and 3. For 4x, use epx=2 and scale the rest
        n = scale if scale in [2, 3] else 2
        engine_filter = f"epx=n={n},scale={target_width}:{target_height}:flags=lanczos"
    elif engine == 'super2xsai':
        engine_filter = f"super2xsai,scale={target_width}:{target_height}:flags=lanczos"
    else:
        engine_filter = f"scale={target_width}:{target_height}:flags=lanczos"
    filters.append(engine_filter)
    
    if sharpen:
        filters.append("unsharp=5:5:0.4:5:5:0.0")
        
    return ",".join(filters)

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """A powerful, easy-to-use video upscaling CLI tool."""
    check_requirements()
    if ctx.invoked_subcommand is None:
        run_interactive_wizard()

def format_bitrate(bitrate_str):
    """Formats bitrate string (bps) to Mbps or Kbps."""
    if not bitrate_str:
        return "N/A"
    try:
        val = float(bitrate_str)
        if val >= 1_000_000:
            return f"{val / 1_000_000:.2f} Mbps"
        else:
            return f"{val / 1_000:.1f} Kbps"
    except ValueError:
        return bitrate_str

def format_duration(seconds_str):
    """Formats duration in seconds to HH:MM:SS."""
    if not seconds_str:
        return "N/A"
    try:
        seconds = float(seconds_str)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{minutes:02d}:{secs:06.3f}"
    except ValueError:
        return seconds_str

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
def info(input_path):
    """Show detailed format and stream info about the input video."""
    video, audio, fmt = get_video_info(input_path)
    
    # General Info
    gen_table = Table(title=f"General Info: {Path(input_path).name}", title_style="bold magenta", show_header=False, box=None)
    gen_table.add_row("[cyan]File Size:[/cyan]", f"{int(fmt.get('size', 0)) / (1024*1024):.2f} MB")
    gen_table.add_row("[cyan]Format:[/cyan]", fmt.get('format_long_name', fmt.get('format_name', 'N/A')))
    gen_table.add_row("[cyan]Duration:[/cyan]", format_duration(fmt.get('duration')))
    gen_table.add_row("[cyan]Overall Bitrate:[/cyan]", format_bitrate(fmt.get('bit_rate')))
    
    # Video Stream Info
    v_table = Table(title="Video Stream Details", title_style="bold green", show_header=False, box=None)
    if video:
        w, h = int(video.get('width', 0)), int(video.get('height', 0))
        sar = video.get('sample_aspect_ratio', '1:1')
        dar = video.get('display_aspect_ratio', 'N/A')
        
        # Calculate true display ratio
        try:
            sar_num, sar_den = map(float, sar.split(':'))
            dar_calc = (w / h) * (sar_num / sar_den)
            dar_calc_str = f"~{dar_calc:.3f} ({dar})"
        except Exception:
            dar_calc_str = dar
            
        v_table.add_row("[green]Codec:[/green]", f"{video.get('codec_name')} ({video.get('codec_long_name')})")
        v_table.add_row("[green]Resolution:[/green]", f"{w}x{h}")
        v_table.add_row("[green]Sample Aspect Ratio (SAR):[/green]", sar)
        v_table.add_row("[green]Display Aspect Ratio (DAR):[/green]", dar_calc_str)
        
        # Frame rate parsing
        fps_str = video.get('avg_frame_rate', video.get('r_frame_rate', '0/0'))
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 0
            fps_display = f"{fps:.2f} fps ({fps_str})"
        else:
            fps_display = f"{fps_str} fps"
        v_table.add_row("[green]Frame Rate:[/green]", fps_display)
        v_table.add_row("[green]Bitrate:[/green]", format_bitrate(video.get('bit_rate')))
        v_table.add_row("[green]Color Space:[/green]", f"{video.get('pix_fmt')} / {video.get('color_range')}")
    else:
        v_table.add_row("[red]No Video Stream found![/red]")
        
    # Audio Stream Info
    a_table = Table(title="Audio Stream Details", title_style="bold yellow", show_header=False, box=None)
    if audio:
        a_table.add_row("[yellow]Codec:[/yellow]", f"{audio.get('codec_name')} ({audio.get('codec_long_name')})")
        a_table.add_row("[yellow]Channels:[/yellow]", f"{audio.get('channels')} ({audio.get('channel_layout', 'stereo')})")
        a_table.add_row("[yellow]Sample Rate:[/yellow]", f"{audio.get('sample_rate')} Hz")
        a_table.add_row("[yellow]Bitrate:[/yellow]", format_bitrate(audio.get('bit_rate')))
    else:
        a_table.add_row("[yellow]No Audio Stream found.[/yellow]")
        
    # Recommendations
    rec_table = Table(title="Upscaling Recommendations", title_style="bold cyan", show_header=False, box=None)
    if video:
        w, h = int(video.get('width', 0)), int(video.get('height', 0))
        sar = video.get('sample_aspect_ratio', '1:1')
        
        rec_table.add_row("[cyan]Ideal Engine for General/Real Video:[/cyan]", "fsrcnn (AI) or lanczos (Fast)")
        rec_table.add_row("[cyan]Ideal Engine for Pixel Art/Animation:[/cyan]", "hqx or xbr")
        
        # Suggest resolutions
        rec_table.add_row("[cyan]Scale 2x Resolution:[/cyan]", f"{calculate_target_dims(w, h, sar, 2, 'auto')[0]}x{calculate_target_dims(w, h, sar, 2, 'auto')[1]} (Corrected 4:3)")
        rec_table.add_row("[cyan]Scale 4x Resolution:[/cyan]", f"{calculate_target_dims(w, h, sar, 4, 'auto')[0]}x{calculate_target_dims(w, h, sar, 4, 'auto')[1]} (Corrected 4:3)")
        
    console.print(Panel(gen_table, border_style="magenta", expand=False))
    console.print(Panel(v_table, border_style="green", expand=False))
    console.print(Panel(a_table, border_style="yellow", expand=False))
    if video:
        console.print(Panel(rec_table, border_style="cyan", expand=False))

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', '-o', type=click.Path(), help="Output file path.")
@click.option('--engine', '-e', type=click.Choice(['lanczos', 'hqx', 'xbr', 'epx', 'super2xsai', 'fsrcnn']), default='hqx', help="Upscaling engine filter.")
@click.option('--scale', '-s', type=click.Choice(['2', '3', '4']), default='4', help="Upscaling factor.")
@click.option('--trim', '-t', help="Trim the video. Format: 'seconds' or 'start,duration' (e.g. '10' or '60,10').")
@click.option('--codec', '-c', type=click.Choice(['h264', 'hevc']), default='h264', help="Output video codec.")
@click.option('--aspect', '-a', type=click.Choice(['auto', 'source', '4:3', '16:9']), default='auto', help="Aspect ratio handling.")
@click.option('--audio', '-d', type=click.Choice(['copy', 'aac', 'none']), default='aac', help="Audio transcode action.")
@click.option('--compare', is_flag=True, help="Create a side-by-side comparison video (original vs. upscaled).")
@click.option('--quality', '-q', type=int, help="Encoder quality factor. For VideoToolbox: 0-100 (default 65). For software: CRF 0-51 (lower is better, default 18/20).")
@click.option('--denoise', '-dn', is_flag=True, help="Apply a spatial/temporal denoising filter before upscaling.")
@click.option('--deblock', '-db', is_flag=True, help="Apply a deblocking filter before upscaling.")
@click.option('--sharpen', '-sp', is_flag=True, help="Apply an unsharp mask (sharpening) filter after upscaling.")
def upscale(input_path, output, engine, scale, trim, codec, aspect, audio, compare, quality, denoise, deblock, sharpen):
    """Upscale a video using mathematical, pixel-art, or AI-based models."""
    scale = int(scale)
    is_mac = sys.platform == 'darwin'
    
    # 1. Probing the input
    console.print("[cyan]Probing input video...[/cyan]")
    video, audio_stream, fmt = get_video_info(input_path)
    if not video:
        console.print("[bold red]Error:[/bold red] Input file contains no video stream.")
        sys.exit(1)
        
    w, h = int(video.get('width', 0)), int(video.get('height', 0))
    sar = video.get('sample_aspect_ratio', '1:1')
    dar = video.get('display_aspect_ratio', 'N/A')
    total_duration = float(fmt.get('duration', 0))
    
    fps_str = video.get('avg_frame_rate', video.get('r_frame_rate', '25/1'))
    if '/' in fps_str:
        num, den = map(float, fps_str.split('/'))
        fps = num / den if den != 0 else 25.0
    else:
        fps = float(fps_str)
        
    # Parse trim
    start_sec, duration_sec = parse_trim(trim, total_duration)
    duration_to_use = duration_sec if duration_sec is not None else (total_duration - start_sec)
    total_frames = int(duration_to_use * fps)
    
    # Target dimensions
    target_w, target_h = calculate_target_dims(w, h, sar, scale, aspect)
    
    # Set default output file if not specified
    if not output:
        name_suffix = f"_{engine}_{scale}x"
        if compare:
            name_suffix += "_compare"
        output = str(Path(input_path).parent / f"{Path(input_path).stem}{name_suffix}.mp4")
        
    # Check if FSRCNN model needs to be downloaded
    model_path = None
    if engine == 'fsrcnn':
        model_path = download_fsrcnn_model(scale)
        
    # Print configuration summary
    summary = Table(title="Upscale Configuration Summary", title_style="bold magenta", show_header=False, box=None)
    summary.add_row("[cyan]Input Video:[/cyan]", input_path)
    summary.add_row("[cyan]Output Video:[/cyan]", output)
    summary.add_row("[cyan]Engine:[/cyan]", f"{engine.upper()} ({scale}x)")
    summary.add_row("[cyan]Resolution Change:[/cyan]", f"{w}x{h} -> {target_w}x{target_h}")
    summary.add_row("[cyan]FPS:[/cyan]", f"{fps:.2f}")
    if trim:
        summary.add_row("[cyan]Trim Segment:[/cyan]", f"Start: {start_sec:.2f}s, Duration: {duration_to_use:.2f}s ({total_frames} frames)")
    else:
        summary.add_row("[cyan]Trim Segment:[/cyan]", "Full Video")
    summary.add_row("[cyan]Codec:[/cyan]", f"{codec.upper()} ({'Hardware Accelerated' if is_mac else 'Software'})")
    summary.add_row("[cyan]Encoder Quality:[/cyan]", str(quality) if quality is not None else "Default (65 for hardware, 18/20 for software)")
    
    enhancements = []
    if deblock: enhancements.append("Deblock (spp)")
    if denoise: enhancements.append("Denoise (hqdn3d/bilateral)")
    if sharpen: enhancements.append("Sharpen (unsharp)")
    summary.add_row("[cyan]Quality Enhancements:[/cyan]", ", ".join(enhancements) if enhancements else "None")
    
    summary.add_row("[cyan]Audio Action:[/cyan]", audio.upper() if audio_stream else "None (Source lacks audio)")
    summary.add_row("[cyan]Comparison Video:[/cyan]", "YES (Split Screen)" if compare else "NO")
    
    console.print(Panel(summary, border_style="magenta"))
    
    # Check if file exists and confirm overwrite
    if os.path.exists(output):
        if not Confirm.ask(f"File '{output}' already exists. Overwrite?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Create temporary log file for FFmpeg errors
    log_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False)
    log_file_path = log_file.name
    log_file.close()

    try:
        if engine == 'fsrcnn':
            # RUNNING FSRCNN VIA OPENCV (PYTHON FRAME LOOP)
            run_fsrcnn_pipeline(
                input_path=input_path,
                output_path=output,
                model_path=model_path,
                scale=scale,
                start_sec=start_sec,
                duration_sec=duration_sec,
                target_w=target_w,
                target_h=target_h,
                fps=fps,
                total_frames=total_frames,
                codec_choice=codec,
                audio_action=audio if audio_stream else 'none',
                compare=compare,
                is_mac=is_mac,
                log_file_path=log_file_path,
                custom_quality=quality,
                deblock=deblock,
                denoise=denoise,
                sharpen=sharpen
            )
        else:
            # RUNNING NATIVE FFMPEG FILTERS (LANCZOS, HQX, XBR, EPX, SUPER2XSAI)
            run_ffmpeg_pipeline(
                input_path=input_path,
                output_path=output,
                engine=engine,
                scale=scale,
                start_sec=start_sec,
                duration_sec=duration_sec,
                target_w=target_w,
                target_h=target_h,
                fps=fps,
                total_frames=total_frames,
                codec_choice=codec,
                audio_action=audio if audio_stream else 'none',
                compare=compare,
                is_mac=is_mac,
                log_file_path=log_file_path,
                custom_quality=quality,
                deblock=deblock,
                denoise=denoise,
                sharpen=sharpen
            )
            
        console.print(f"\n[bold green]✓ Upscaling completed successfully![/bold green]")
        console.print(f"Output saved to: [cyan]{output}[/cyan]")
    except Exception as e:
        console.print(f"\n[bold red]Error upscaling video:[/bold red] {e}")
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as lf:
                logs = lf.read()
            if logs:
                console.print(Panel(logs, title="FFmpeg Logs / Errors", border_style="red"))
        sys.exit(1)
    finally:
        if os.path.exists(log_file_path):
            try:
                os.remove(log_file_path)
            except Exception:
                pass

def run_ffmpeg_pipeline(input_path, output_path, engine, scale, start_sec, duration_sec, 
                        target_w, target_h, fps, total_frames, codec_choice, audio_action, 
                        compare, is_mac, log_file_path, custom_quality=None, 
                        deblock=False, denoise=False, sharpen=False):
    """Executes the upscaling entirely within an FFmpeg subprocess using C-based filters."""
    ffmpeg_cmd = ['ffmpeg', '-y']
    
    # 1. Input configuration (trimming at input level is much faster)
    if start_sec > 0:
        ffmpeg_cmd.extend(['-ss', f'{start_sec:.3f}'])
    if duration_sec is not None:
        ffmpeg_cmd.extend(['-t', f'{duration_sec:.3f}'])
        
    ffmpeg_cmd.extend(['-i', input_path])
    
    # 2. Filter Graph Construction
    engine_filter = get_ffmpeg_filter(engine, scale, target_w, target_h, deblock=deblock, denoise=denoise, sharpen=sharpen)
    
    if compare:
        font_path = get_font_file()
        if font_path:
            escaped_font = font_path.replace(":", "\\:").replace("'", "'\\''")
            draw_left = f",drawtext=text='ORIGINAL (BILINEAR)':x=20:y=20:fontfile='{escaped_font}':fontsize={int(target_h*0.035)}:fontcolor=white:box=1:boxcolor=black@0.5"
            draw_right = f",drawtext=text='UPSCALED ({engine.upper()})':x=20:y=20:fontfile='{escaped_font}':fontsize={int(target_h*0.035)}:fontcolor=white:box=1:boxcolor=black@0.5"
        else:
            draw_left = ""
            draw_right = ""
            
        # Split screen split: Left = Bilinear Original, Right = Upscaled, Stacked side-by-side
        filter_str = (
            f"[0:v]split=2[v1][v2]; "
            f"[v1]scale={target_w}:{target_h}:flags=bilinear,crop={target_w//2}:{target_h}{draw_left}[left]; "
            f"[v2]{engine_filter},crop={target_w//2}:{target_h}{draw_right}[right]; "
            f"[left][right]hstack[out]"
        )
        ffmpeg_cmd.extend(['-filter_complex', filter_str, '-map', '[out]'])
    else:
        # Standard filter
        ffmpeg_cmd.extend(['-vf', engine_filter])
        
    # 3. Audio mapping
    if audio_action == 'copy':
        ffmpeg_cmd.extend(['-c:a', 'copy'])
    elif audio_action == 'aac':
        ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
    else:
        ffmpeg_cmd.extend(['-an'])
        
    # 4. Video encoding settings
    encoder_name, quality_args = get_codec_params(codec_choice, is_mac, custom_quality=custom_quality)
    ffmpeg_cmd.extend(['-c:v', encoder_name])
    ffmpeg_cmd.extend(quality_args)
    
    # 5. Progress logging
    ffmpeg_cmd.extend(['-progress', '-', '-nostats'])
    ffmpeg_cmd.append(output_path)
    
    # Start subprocess
    lf = open(log_file_path, 'w')
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=lf,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Process progress
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        FPSColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task(f"Processing ({engine.upper()})...", total=total_frames)
        
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith('frame='):
                try:
                    curr_frame = int(line.split('=')[1])
                    progress.update(task_id, completed=curr_frame)
                except Exception:
                    pass
            elif line.startswith('progress=end'):
                break
                
    lf.close()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ffmpeg_cmd)

def run_fsrcnn_pipeline(input_path, output_path, model_path, scale, start_sec, duration_sec, 
                        target_w, target_h, fps, total_frames, codec_choice, audio_action, 
                        compare, is_mac, log_file_path, custom_quality=None, 
                        deblock=False, denoise=False, sharpen=False):
    """Executes the upscaling by reading frames using OpenCV, applying FSRCNN DNN, and piping to FFmpeg."""
    # 1. Initialize FSRCNN
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", scale)
    
    # 2. Open OpenCV capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video with OpenCV: {input_path}")
        
    if start_sec > 0:
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
    # 3. Build FFmpeg command to receive raw frames
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{target_w}x{target_h}',
        '-r', str(fps),
        '-i', '-'  # Read raw frames from stdin pipe
    ]
    
    # If audio is needed, add original video as input 2
    if audio_action != 'none':
        audio_trim = []
        if start_sec > 0:
            audio_trim.extend(['-ss', f'{start_sec:.3f}'])
        if duration_sec is not None:
            audio_trim.extend(['-t', f'{duration_sec:.3f}'])
            
        ffmpeg_cmd.extend(audio_trim + ['-i', input_path])
        ffmpeg_cmd.extend(['-map', '0:v', '-map', '1:a?'])
        
        if audio_action == 'copy':
            ffmpeg_cmd.extend(['-c:a', 'copy'])
        elif audio_action == 'aac':
            ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
    else:
        ffmpeg_cmd.extend(['-an'])
        
    encoder_name, quality_args = get_codec_params(codec_choice, is_mac, custom_quality=custom_quality)
    ffmpeg_cmd.extend(['-c:v', encoder_name])
    ffmpeg_cmd.extend(quality_args)
    ffmpeg_cmd.append(output_path)
    
    # 4. Start subprocess
    lf = open(log_file_path, 'w')
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=lf
    )
    
    # 5. Process frame-by-frame loop
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        FPSColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Running AI FSRCNN...", total=total_frames)
        
        try:
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Pre-processing: Deblock / Denoise using Bilateral Filter
                if deblock or denoise:
                    frame = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
                    
                # Run AI Model
                upscaled = sr.upsample(frame)
                
                # Aspect Ratio sizing check (if target dims are not exactly scaled dims)
                if upscaled.shape[1] != target_w or upscaled.shape[0] != target_h:
                    upscaled = cv2.resize(upscaled, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Post-processing: Sharpen with Unsharp Mask
                if sharpen:
                    blurred = cv2.GaussianBlur(upscaled, (5, 5), 1.0)
                    upscaled = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
                
                if compare:
                    # Stretched or corrected bilinear resize for left half comparison
                    original_scaled = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    left = original_scaled[:, :target_w // 2]
                    right = upscaled[:, target_w // 2:]
                    out_frame = np.hstack((left, right))
                    
                    # Draw a nice green vertical division line
                    cv2.line(out_frame, (target_w // 2, 0), (target_w // 2, target_h), (0, 255, 0), 2)
                    
                    # Text overlays (black background shadow + white text)
                    h_scale = target_h / 500.0  # Make font size dynamic based on resolution
                    font_thickness = max(1, int(2 * h_scale))
                    font_scale = 0.8 * h_scale
                    
                    cv2.putText(out_frame, "ORIGINAL (BILINEAR)", (20, int(40 * h_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                    cv2.putText(out_frame, "ORIGINAL (BILINEAR)", (20, int(40 * h_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
                    cv2.putText(out_frame, "AI UPSCALED (FSRCNN)", (target_w // 2 + 20, int(40 * h_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                    cv2.putText(out_frame, "AI UPSCALED (FSRCNN)", (target_w // 2 + 20, int(40 * h_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                else:
                    out_frame = upscaled
                    
                # Write raw bytes to FFmpeg stdin
                proc.stdin.write(out_frame.tobytes())
                progress.update(task_id, completed=i+1)
                
        except Exception as e:
            raise e
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            lf.close()
            
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ffmpeg_cmd)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', '-o', type=click.Path(), help="Output file path.")
@click.option('--trim', '-t', default='5', help="Trim duration in seconds for the comparison clip. Default is 5s.")
@click.option('--scale', '-s', type=click.Choice(['2', '3', '4']), default='2', help="Upscaling factor.")
def compare_all(input_path, output, trim, scale):
    """Generate a combined grid video comparing multiple engines (Lanczos, HQX, XBR, FSRCNN)."""
    scale = int(scale)
    is_mac = sys.platform == 'darwin'
    
    # Probe input
    video, audio, fmt = get_video_info(input_path)
    if not video:
        console.print("[bold red]Error:[/bold red] Input file contains no video stream.")
        sys.exit(1)
        
    w, h = int(video.get('width', 0)), int(video.get('height', 0))
    sar = video.get('sample_aspect_ratio', '1:1')
    total_duration = float(fmt.get('duration', 0))
    
    fps_str = video.get('avg_frame_rate', video.get('r_frame_rate', '25/1'))
    if '/' in fps_str:
        num, den = map(float, fps_str.split('/'))
        fps = num / den if den != 0 else 25.0
    else:
        fps = float(fps_str)
        
    # Target size (aspect corrected)
    target_w, target_h = calculate_target_dims(w, h, sar, scale, 'auto')
    
    if not output:
        output = str(Path(input_path).parent / f"{Path(input_path).stem}_comparison_grid.mp4")
        
    # Start and duration
    start_sec, duration_sec = parse_trim(trim, total_duration)
    duration_to_use = duration_sec if duration_sec is not None else 5.0
    total_frames = int(duration_to_use * fps)
    
    console.print(f"[magenta]Generating a 4-way comparison grid (Original, Lanczos, HQX, FSRCNN) of {duration_to_use:.1f}s...[/magenta]")
    
    # We will build this comparison using Python frames to avoid complex shell filter syntax, 
    # ensuring we can overlay grid lines and texts accurately.
    # Download FSRCNN model
    model_path = download_fsrcnn_model(scale)
    
    # Load AI model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", scale)
    
    # Open capture
    cap = cv2.VideoCapture(input_path)
    if start_sec > 0:
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
    # Grid dimensions (2x2 grid, each cell is target_w x target_h, total size is 2*target_w x 2*target_h)
    grid_w = target_w * 2
    grid_h = target_h * 2
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{grid_w}x{grid_h}',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-an'       # No audio for comparison grid
    ]
    
    encoder_name, quality_args = get_codec_params('h264', is_mac)
    ffmpeg_cmd.extend(['-c:v', encoder_name])
    ffmpeg_cmd.extend(quality_args)
    ffmpeg_cmd.append(output)
    
    log_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False)
    log_file_path = log_file.name
    log_file.close()
    
    lf = open(log_file_path, 'w')
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=lf)
    
    # Initialize CPU-based filters (hqx and xbr via opencv fallbacks or ffmpeg filters? 
    # To keep this grid generator 100% self-contained and fast, we can use OpenCV resize (bilinear/lanczos) 
    # and OpenCV FSRCNN. For hqx/xbr in a 4-way grid, we can show:
    # 1. Top Left: Original (Nearest Neighbor)
    # 2. Top Right: Bilinear Interpolation (standard player scale)
    # 3. Bottom Left: Lanczos Interpolation (high-quality math)
    # 4. Bottom Right: FSRCNN (AI Super Resolution)
    # This is a very clean, representative grid!)
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        FPSColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Creating Comparison Grid...", total=total_frames)
        
        try:
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Cell 1: Original (Nearest Neighbor)
                c1 = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # Cell 2: Bilinear
                c2 = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                # Cell 3: Lanczos
                c3 = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Cell 4: FSRCNN AI
                c4 = sr.upsample(frame)
                if c4.shape[1] != target_w or c4.shape[0] != target_h:
                    c4 = cv2.resize(c4, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    
                # Overlay labels
                h_scale = target_h / 500.0
                font_scale = 0.8 * h_scale
                font_thickness = max(1, int(2 * h_scale))
                y_offset = int(40 * h_scale)
                
                for cell, label in [(c1, "1. ORIGINAL (NEAREST)"), 
                                    (c2, "2. BILINEAR (STANDARD)"), 
                                    (c3, "3. LANCZOS (HIGH-QUALITY MATH)"), 
                                    (c4, "4. AI FSRCNN (SUPER RESOLUTION)")]:
                    cv2.putText(cell, label, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                    cv2.putText(cell, label, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
                # Combine into grid (top row, bottom row)
                top_row = np.hstack((c1, c2))
                bottom_row = np.hstack((c3, c4))
                grid_frame = np.vstack((top_row, bottom_row))
                
                # Draw grid dividers (black borders)
                cv2.line(grid_frame, (target_w, 0), (target_w, grid_h), (0, 0, 0), 4)
                cv2.line(grid_frame, (0, target_h), (grid_w, target_h), (0, 0, 0), 4)
                
                # Write raw frames
                proc.stdin.write(grid_frame.tobytes())
                progress.update(task_id, completed=i+1)
                
        except Exception as e:
            console.print(f"[red]Error building comparison grid: {e}[/red]")
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            lf.close()
            
    if proc.returncode == 0:
        console.print(f"\n[bold green]✓ Comparison grid generated successfully![/bold green]")
        console.print(f"Output saved to: [cyan]{output}[/cyan]")
    else:
        console.print(f"[bold red]Failed to encode comparison grid.[/bold red]")
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as lf:
                console.print(Panel(lf.read(), title="FFmpeg Errors", border_style="red"))
                
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
        except Exception:
            pass

def run_interactive_wizard():
    """Launches an interactive wizard to configure and run upscaling."""
    console.print(Panel("[bold magenta]Welcome to the Video Upscaler Interactive Wizard![/bold magenta]\n"
                        "This tool will guide you through upscaling your low-resolution videos.",
                        border_style="magenta"))
    
    # 1. Ask for input file
    input_path = ""
    while not input_path:
        # Check for files in input-videos/ or current directory
        suggested_files = []
        for p in [Path("input-videos"), Path(".")]:
            if p.exists():
                suggested_files.extend(list(p.glob("*.DAT")) + list(p.glob("*.dat")) + 
                                       list(p.glob("*.mp4")) + list(p.glob("*.avi")) + 
                                       list(p.glob("*.mpg")) + list(p.glob("*.mpeg")))
        
        if suggested_files:
            console.print("\n[cyan]Detected video files in your project:[/cyan]")
            for idx, f in enumerate(suggested_files):
                console.print(f"  [{idx + 1}] {f}")
            
            choice = Prompt.ask("\nSelect a file number or type a custom path", default="1")
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(suggested_files):
                    input_path = str(suggested_files[choice_idx])
                else:
                    input_path = choice
            except ValueError:
                input_path = choice
        else:
            input_path = Prompt.ask("\nEnter the path to the low-resolution video file")
            
        if not os.path.exists(input_path):
            console.print(f"[red]Error: File '{input_path}' not found. Please try again.[/red]")
            input_path = ""
            
    # Show info about selected video
    console.print("\n[cyan]Gathering file metadata...[/cyan]")
    video, audio, fmt = get_video_info(input_path)
    
    if video:
        w, h = int(video.get('width', 0)), int(video.get('height', 0))
        console.print(f"  [green]✓ Detected {w}x{h} video stream ({video.get('codec_name')})[/green]")
        if audio:
            console.print(f"  [green]✓ Detected audio stream ({audio.get('codec_name')})[/green]")
    else:
        console.print("[red]Error: No video stream detected in this file.[/red]")
        sys.exit(1)
        
    # 2. Ask for engine
    console.print("\n[cyan]Select Upscaling Engine:[/cyan]")
    console.print("  [1] hqx (Recommended for sharp pixel-art / crisp lines)")
    console.print("  [2] fsrcnn (Recommended AI model, best quality for real scenes)")
    console.print("  [3] lanczos (Fastest high-quality mathematical scaling)")
    console.print("  [4] xbr (High-quality anti-aliasing curve scaling)")
    console.print("  [5] epx / super2xsai (Classic game magnification filters)")
    
    engine_choice = Prompt.ask("Select engine", choices=["1", "2", "3", "4", "5"], default="1")
    engine_map = {"1": "hqx", "2": "fsrcnn", "3": "lanczos", "4": "xbr", "5": "epx"}
    engine = engine_map[engine_choice]
    
    # 3. Ask for scaling factor
    scale = Prompt.ask("\nSelect scaling factor (2x, 3x, or 4x)", choices=["2", "3", "4"], default="4")
    
    # 4. Ask for trim
    trim_confirm = Confirm.ask("\nDo you want to trim the video first (highly recommended for testing)?", default=True)
    trim = None
    if trim_confirm:
        trim = Prompt.ask("Enter trim length in seconds (e.g. '5' or start,duration like '60,10')", default="10")
        
    # 5. Aspect Ratio correction
    aspect_confirm = Confirm.ask("\nCorrect aspect ratio to square pixels (e.g., PAL/VCD 4:3 correction)?", default=True)
    aspect = "auto" if aspect_confirm else "source"
    
    # 6. Compare Mode
    compare = False
    if trim_confirm:
        compare = Confirm.ask("\nCreate side-by-side comparison video (original vs upscaled)?", default=True)
        
    # Build target output file
    name_suffix = f"_{engine}_{scale}x"
    if compare:
        name_suffix += "_compare"
    default_out = str(Path(input_path).parent / f"{Path(input_path).stem}{name_suffix}.mp4")
    
    output = Prompt.ask("\nEnter output file path", default=default_out)
    
    # Run the command
    sys.argv = [
        "upscale.py", "upscale",
        input_path,
        "-o", output,
        "--engine", engine,
        "--scale", scale,
        "--aspect", aspect,
        "--audio", "aac" if audio else "none"
    ]
    if trim:
        sys.argv.extend(["--trim", trim])
    if compare:
        sys.argv.append("--compare")
        
    console.print(f"\n[cyan]Executing command:[/cyan] [yellow]uv run upscale.py {' '.join(sys.argv[1:])}[/yellow]\n")
    cli()

if __name__ == '__main__':
    cli()
