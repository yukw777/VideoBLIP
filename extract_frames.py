import argparse
import json
import os
import re
import subprocess

from tqdm import tqdm

C_REGEX = re.compile(r"^\#C C", re.IGNORECASE)
parser = argparse.ArgumentParser()
parser.add_argument("fho_main_path")
parser.add_argument("video_folder")
parser.add_argument("frames_folder")
args = parser.parse_args()


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


# Load fho_main.json
with open(args.fho_main_path) as f:
    data = json.load(f)

# Iterate through videos
for video in tqdm(data["videos"]):
    video_uid = video["video_uid"]
    input_video = os.path.join(args.video_folder, f"{video_uid}.mp4")

    for interval in video["annotated_intervals"]:
        for narrated_action in interval["narrated_actions"]:
            if (
                narrated_action["is_rejected"]
                or not narrated_action["is_valid_action"]
                or C_REGEX.match(narrated_action["narration_text"]) is None
            ):
                continue
            narration_text = narrated_action["narration_text"]
            start_sec = narrated_action["start_sec"]
            start_hms = seconds_to_hms(start_sec)
            end_sec = narrated_action["end_sec"]
            end_hms = seconds_to_hms(end_sec)
            duration = round(end_sec - start_sec)
            narrated_action_id = "|".join([video_uid, start_hms, end_hms])

            # Create a folder for the extracted frames
            frames_folder = os.path.join(args.frames_folder, f"{narrated_action_id}")
            os.makedirs(frames_folder, exist_ok=True)

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
                f"{frames_folder}/%04d.png",
            ]
            subprocess.run(
                ffmpeg_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Save narration.json
            narration_data = {
                "video_uid": video_uid,
                "start_sec": start_sec,
                "start_hms": start_hms,
                "end_sec": end_sec,
                "end_hms": end_hms,
                "narration_text": narration_text,
            }
            with open(os.path.join(frames_folder, "narration.json"), "w") as f:
                json.dump(narration_data, f, indent=2)
