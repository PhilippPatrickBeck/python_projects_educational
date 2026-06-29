#!/usr/bin/env python3
"""
Recursively convert all audio files in a source directory to MP3 using
Variable Bitrate (VBR) targeting ~64 kbit/s.
Output is written to 'audio_converted' in the parent directory of the source.

Optimized with multiprocessing for parallel conversion across CPU cores.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# ====== CUSTOM FFMPEG PATHS ======
# If ffmpeg/ffprobe are not in your system PATH, set the full paths here.
FFMPEG_PATH = ""   # e.g. "C:/ffmpeg/bin/ffmpeg.exe" or "/usr/local/bin/ffmpeg"
FFPROBE_PATH = ""  # e.g. "C:/ffmpeg/bin/ffprobe.exe" or "/usr/local/bin/ffprobe"
# =================================

AUDIO_EXTS = {
    '.wma', '.aac', '.flac', '.ogg', '.wav',
    '.m4a', '.ape', '.ac3', '.mp4', '.m4b', '.opus'
}
TARGET_BITRATE = 64          # kbit/s (average target for VBR)
BITRATE_TOLERANCE = 8        # kbit/s (skip if existing bitrate is within this range)
VBR_QUALITY = 9              # LAME VBR scale: 0 (best) to 9 (smallest, ~64 kbps avg)
DEFAULT_WORKERS = cpu_count()  # use all available cores


def get_ffmpeg_cmd():
    return FFMPEG_PATH if FFMPEG_PATH else "ffmpeg"


def get_ffprobe_cmd():
    return FFPROBE_PATH if FFPROBE_PATH else "ffprobe"


def check_ffmpeg():
    ffmpeg = get_ffmpeg_cmd()
    ffprobe = get_ffprobe_cmd()
    if not shutil.which(ffmpeg):
        print(f"Error: ffmpeg not found at '{ffmpeg}'. Please install ffmpeg or set the correct path.", file=sys.stderr)
        sys.exit(1)
    if not shutil.which(ffprobe):
        print(f"Error: ffprobe not found at '{ffprobe}'. Please install ffmpeg or set the correct path.", file=sys.stderr)
        sys.exit(1)


def get_bitrate(filepath):
    """Read average bitrate of an MP3 file (works for CBR and VBR)."""
    ffprobe = get_ffprobe_cmd()
    cmd = [
        ffprobe, '-v', 'error',
        '-show_entries', 'format=bit_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        if output:
            return int(output) // 1000
    except (subprocess.CalledProcessError, ValueError):
        return None
    return None


def convert_to_mp3(input_path, output_path):
    """Convert any audio file to MP3 using VBR."""
    ffmpeg = get_ffmpeg_cmd()
    cmd = [
        ffmpeg, '-i', input_path,
        '-q:a', str(VBR_QUALITY),
        '-acodec', 'libmp3lame',
        '-y',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode()


def process_task(task):
    """
    Worker function for a single file conversion.
    task is a tuple (src_path, dst_path, is_mp3).
    Returns a status string and success flag.
    """
    src_path, dst_path, is_mp3 = task
    try:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Skip if target already has correct bitrate
        if os.path.exists(dst_path):
            bitrate = get_bitrate(dst_path)
            if bitrate is not None and abs(bitrate - TARGET_BITRATE) <= BITRATE_TOLERANCE:
                return f"Skipped {src_path} -> {dst_path} (already ~{bitrate} kbps)", True

        # If source is MP3 and bitrate already correct, just copy
        if is_mp3:
            src_bitrate = get_bitrate(src_path)
            if src_bitrate is not None and abs(src_bitrate - TARGET_BITRATE) <= BITRATE_TOLERANCE:
                shutil.copy2(src_path, dst_path)
                return f"Copied {src_path} -> {dst_path} (already ~{src_bitrate} kbps)", True

        # Otherwise convert using temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
        success = convert_to_mp3(src_path, tmp_path)
        if success is True:
            shutil.move(tmp_path, dst_path)
            return f"Converted {src_path} -> {dst_path} (VBR quality {VBR_QUALITY})", True
        else:
            os.remove(tmp_path)
            return f"FAILED: {src_path} - {success[1]}", False
    except Exception as e:
        return f"ERROR processing {src_path}: {e}", False


def collect_tasks(source_dir, target_dir):
    """Walk source_dir and build a list of tasks (src, dst, is_mp3)."""
    tasks = []
    for dirpath, _, filenames in os.walk(source_dir):
        for filename in filenames:
            src_path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in AUDIO_EXTS and ext != '.mp3':
                continue
            rel_path = os.path.relpath(src_path, source_dir)
            base_name = os.path.splitext(rel_path)[0] + '.mp3'
            dst_path = os.path.join(target_dir, base_name)
            tasks.append((src_path, dst_path, ext == '.mp3'))
    return tasks


def main(source_dir, target_dir, workers):
    if not os.path.isdir(source_dir):
        print(f"Error: '{source_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    if os.path.abspath(target_dir) == os.path.abspath(source_dir) or \
       os.path.commonpath([os.path.abspath(source_dir), os.path.abspath(target_dir)]) == os.path.abspath(source_dir):
        print("Error: target directory cannot be the same or a subdirectory of source.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)

    print("Collecting files...")
    tasks = collect_tasks(source_dir, target_dir)
    if not tasks:
        print("No audio files found.")
        return

    print(f"Found {len(tasks)} files to process. Using {workers} worker processes.")
    print("Starting parallel conversion...\n")

    successful = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            result_msg, success = future.result()
            print(result_msg)
            if success:
                successful += 1
            else:
                failed += 1

    print(f"\nDone. Successful: {successful}, Failed: {failed}")


if __name__ == '__main__':
    check_ffmpeg()

    if len(sys.argv) < 2:
        print("Usage: python convert_audio.py <source_dir> [target_dir] [workers]")
        print("  target_dir defaults to 'audio_converted' in the parent directory of source_dir.")
        print(f"  workers defaults to number of CPU cores ({DEFAULT_WORKERS}).")
        sys.exit(1)

    src = sys.argv[1]
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    else:
        parent_dir = os.path.dirname(os.path.abspath(src))
        dst = os.path.join(parent_dir, 'audio_converted')
        print(f"No target directory specified. Using default: {dst}")

    if len(sys.argv) > 3:
        try:
            workers = int(sys.argv[3])
            if workers < 1:
                raise ValueError
        except ValueError:
            print("Invalid number of workers. Using default.", file=sys.stderr)
            workers = DEFAULT_WORKERS
    else:
        workers = DEFAULT_WORKERS

    main(src, dst, workers)
