"""Script to extract video frames from video files.

Both single video files, whole directories and CSVs of videos are supported.
"""

from pathlib import Path
from typing import Optional, Any
import logging

import click
import ffmpeg
import numpy as np
from PIL import Image
from tqdm import tqdm

from spai import data_utils


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "2.1.0"
__revision__: int = 4

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHUNK_SIZE = 100  # number of frames to process per batch


@click.group()
def cli() -> None:
    pass


@cli.command(help="Preprocesses a csv file containing paths to videos.")
@click.option("-c", "--csv-path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-r", "--csv-root",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-d", "--csv-delimiter", type=str, default=",", show_default=True)
@click.option("-o", "--output-dir", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Path to the output directory. Inside this directory the relative "
                   "hierarchy that is present in the provided csv file will be maintained.")
@click.option("-p", "--output-csv", required=True,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Path to the new csv file. It will include all the column of the input csv "
                   "and the new columns specified by `--frame-num-column` and `frame-column`.")
@click.option("--video-column", type=str, default="video", show_default=True)
@click.option("--frame-num-column", type=str, default="frame_num", show_default=True)
@click.option("--frame-column", type=str, default="image", show_default=True)
@click.option("--split-frame", is_flag=True,
              help="When this flag is provided each frame is split into four images, "
                   "i.e. top-left, top-right, bottom-left, bottom-right. This flag is "
                   "useful for splitting videos that include many concatenated views "
                   "in a single video.")
@click.option("--fps", type=int, default=None,
              help="Number of frames per second to be extracted from the video.")
def preprocess_csv(
    csv_path: Path,
    csv_root: Optional[Path],
    csv_delimiter: str,
    output_dir: Path,
    output_csv: Path,
    video_column: str,
    frame_num_column: str,
    frame_column: str,
    split_frame: bool,
    fps: Optional[int]
) -> None:
    if csv_root is None:
        csv_root = csv_path.parent

    entries: list[dict[str, Any]] = data_utils.read_csv_file(csv_path, delimiter=csv_delimiter)
    frame_entries: list[dict[str, Any]] = []

    for e in tqdm(entries, desc="Preprocessing videos", unit="video"):
        video_path: Path = csv_root / e[video_column]
        assert video_path.exists(), f"{video_path} does not exist"
        video_out_dir: Path = (output_dir / e[video_column]).parent

        frames: list[Path] = extract_video_frames(
            video_path, video_out_dir, split_frame, fps
        )[video_path]

        for i, f in enumerate(frames):
            frame_entry: dict[str, Any] = e.copy()
            frame_entry[frame_num_column] = str(i)
            frame_entry[frame_column] = str(f.relative_to(csv_root))
            frame_entries.append(frame_entry)

    data_utils.write_csv_file(frame_entries, output_csv, delimiter=csv_delimiter)


@cli.command(help="Extracts the frames of videos as single images.")
@click.option("--video_path", "-v", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Path to a video file or a directory containing video files.")
@click.option("--output_dir", "-o", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Path to a directory where the frame images will be saved.")
@click.option("--split-frame", is_flag=True,
              help="When this flag is provided each frame is split into four images, "
                   "i.e. top-left, top-right, bottom-left, bottom-right. This flag is "
                   "useful for splitting videos that include many concatenated views "
                   "in a single video.")
@click.option("--fps", type=int, default=None,
              help="Number of frames per second to be extracted from the video.")
def extract_frames(
    video_path: Path,
    output_dir: Path,
    split_frame: bool,
    fps: Optional[int]
) -> None:
    extract_video_frames(video_path, output_dir, split_frame, fps)


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    split_frame: bool,
    fps: Optional[int]
) -> dict[Path, list[Path]]:
    if video_path.is_file():
        videos: list[Path] = [video_path]
    else:
        videos: list[Path] = [v for v in video_path.iterdir() if v.is_file()]

    frames_dir: Path = output_dir / ("frame_views" if split_frame else "frames")
    frames_dir.mkdir(exist_ok=True, parents=True)

    frame_paths: dict[Path, list[Path]] = {}

    for v in videos:
        try:
            total_frames: int = get_total_frames(v, fps)
        except Exception as e:
            logger.error(f"Failed to probe video: {v}")
            logger.exception(e)
            continue

        video_frames_dir: Path = frames_dir / v.stem
        video_frames_dir.mkdir(exist_ok=True, parents=True)

        video_frames_paths: list[Path] = []
        for start in range(0, total_frames, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE - 1, total_frames - 1)
            try:
                frames_chunk = load_video_chunk(
                    v, start_frame=start, end_frame=end, fps=fps
                )
            except Exception as e:
                logger.error(f"Failed to load frames {start}-{end} from: {v}")
                logger.exception(e)
                continue

            for idx, frame in enumerate(frames_chunk):
                frame_index = start + idx
                if split_frame:
                    h, w = frame.shape[:2]
                    hh, hw = h // 2, w // 2
                    views = [
                        frame[:hh, :hw], frame[:hh, hw:],
                        frame[hh:, :hw], frame[hh:, hw:]
                    ]
                    for vid_idx, view in enumerate(views):
                        out_path = video_frames_dir / f"frame_{frame_index}_view_{vid_idx}.png"
                        Image.fromarray(view).save(out_path)
                        video_frames_paths.append(out_path)
                else:
                    out_path = video_frames_dir / f"frame_{frame_index}.png"
                    Image.fromarray(frame).save(out_path)
                    video_frames_paths.append(out_path)

        frame_paths[v] = video_frames_paths

    return frame_paths


def get_total_frames(
    video_path: Path,
    fps: Optional[int]
) -> int:
    probe = ffmpeg.probe(str(video_path))
    vinfo = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    if 'nb_frames' in vinfo:
        return int(vinfo['nb_frames'])
    # fallback: estimate from duration and fps
    duration = float(vinfo['duration'])
    use_fps = fps if fps is not None else float(vinfo.get('r_frame_rate', '0').split('/')[0])
    return int(np.ceil(duration * use_fps))


def get_video_dimensions(
    video_path: Path
) -> tuple[int, int]:
    probe = ffmpeg.probe(str(video_path))
    vinfo = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return int(vinfo['width']), int(vinfo['height'])


def load_video_chunk(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    fps: Optional[int]
) -> np.ndarray:
    """Loads a chunk of video frames between frame indices [start_frame, end_frame], inclusive."""
    select_expr = f"between(n,{start_frame},{end_frame})"
    cap = ffmpeg.input(str(video_path))
    cap = cap.filter('select', select_expr)
    if fps is not None:
        cap = cap.filter('fps', fps=fps)
    out, _ = cap.output('pipe:', format='rawvideo', pix_fmt='rgb24') \
                 .global_args('-loglevel', 'error','-vsync','0').run(capture_stdout=True)
    width, height = get_video_dimensions(video_path)
    # calculate actual number of frames returned
    frame_bytes = 3 * width * height
    total_bytes = len(out)
    num_frames = total_bytes // frame_bytes
    return np.frombuffer(out, np.uint8).reshape(num_frames, height, width, 3)


if __name__ == "__main__":
    cli()
