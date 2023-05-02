import argparse
import csv
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

C_REGEX = re.compile(r"^\#C C", re.IGNORECASE)
parser = argparse.ArgumentParser()
parser.add_argument("fho_main_path")
parser.add_argument("video_dir")
parser.add_argument("frames_dir")
args = parser.parse_args()


def seconds_to_hms(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def process_narrated_action(
    video_uid: str, input_video: str, narrated_action: dict
) -> dict | None:
    if (
        narrated_action["is_rejected"]
        or not narrated_action["is_valid_action"]
        or C_REGEX.match(narrated_action["narration_text"]) is None
    ):
        return None
    narration_text = narrated_action["narration_text"]
    start_sec = narrated_action["start_sec"]
    start_hms = seconds_to_hms(start_sec)
    end_sec = narrated_action["end_sec"]
    end_hms = seconds_to_hms(end_sec)
    duration = round(end_sec - start_sec)
    narrated_action_id = "|".join([video_uid, start_hms, end_hms])

    # Create a dir for the extracted frames
    frames_dir = os.path.join(args.frames_dir, f"{narrated_action_id}")
    os.makedirs(frames_dir, exist_ok=True)

    # Extract frames using ffmpeg
    ffmpeg_command = [
        "ffmpeg",
        "-ss",
        start_hms,
        "-i",
        input_video,
        "-t",
        str(duration),
        "-vf",
        "fps=1,scale=224:224",
        "-q:v",
        "1",
        f"{frames_dir}/{narrated_action_id}_%04d.png",
    ]
    subprocess.run(
        ffmpeg_command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Create narration_data and return it
    narration_data = {
        "narration_action_id": narrated_action_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "narration_text": narration_text,
    }
    return narration_data


# Load fho_main.json
with open(args.fho_main_path) as f:
    data = json.load(f)

# Create a directory to save all the results
os.makedirs(args.frames_dir, exist_ok=True)

# Open narrations.csv file for writing
with open(os.path.join(args.frames_dir, "narrations.csv"), "w", newline="") as csvfile:
    # Initialize CSV writer
    csv_writer = csv.DictWriter(
        csvfile,
        [
            "narration_action_id",
            "start_sec",
            "end_sec",
            "narration_text",
        ],
    )

    # Write header row
    csv_writer.writeheader()

    # Iterate through videos
    for video in tqdm(data["videos"]):
        video_uid = video["video_uid"]
        input_video = os.path.join(args.video_dir, f"{video_uid}.mp4")

        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Use a list comprehension to submit tasks to the executor
            futures = [
                executor.submit(
                    process_narrated_action, video_uid, input_video, narrated_action
                )
                for interval in video["annotated_intervals"]
                for narrated_action in interval["narrated_actions"]
            ]

            # Wait for all tasks to complete and accumulate narration_data
            for future in tqdm(futures, desc=f"Processing video {video_uid}"):
                narration_data = future.result()
                if narration_data is not None:
                    csv_writer.writerow(narration_data)
