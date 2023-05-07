"""Check if an extracted narrated action is in the correct split, and move it
to the correct split if not.

This script was written as the initial splits erroneously included
videos from `fho_main.json` that don't actually exist.
"""
import argparse
import glob
import json
import logging
import shutil
from collections import Counter
from csv import DictReader, DictWriter
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("split_output_path")
parser.add_argument("train_extracted_frames_path")
parser.add_argument("val_extracted_frames_path")
parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--verify-only", action="store_true", default=False)
args = parser.parse_args()


logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_narrated_actions_dict(frames_path: str) -> dict[str, dict[str, str]]:
    narrated_actions_dict: dict[str, dict[str, str]] = {}
    with open(Path(frames_path) / "narrated_actions.csv", newline="") as csvfile:
        csvreader = DictReader(csvfile)
        for row in csvreader:
            narrated_actions_dict[row["frame_path"]] = row
    return narrated_actions_dict


def dump_narrated_actions_dict(
    frames_path: str, narrated_actions_dict: dict[str, dict[str, str]]
) -> None:
    with open(Path(frames_path) / "narrated_actions.csv", "w", newline="") as csvfile:
        csvwriter = DictWriter(
            csvfile,
            [
                "frame_path",
                "video_uid",
                "clip_index",
                "narration_timestamp_sec",
                "narration_text",
            ],
        )
        csvwriter.writeheader()
        for _, narrated_action in narrated_actions_dict.items():
            csvwriter.writerow(narrated_action)


def move_frame_dirs(args: argparse.Namespace, from_split: str, to_split: str) -> None:
    with open(f"{args.split_output_path}/fho_main_{to_split}.json") as f:
        split = json.load(f)
    to_frames_path = getattr(args, f"{to_split}_extracted_frames_path")
    to_narrated_actions_dict = load_narrated_actions_dict(to_frames_path)
    from_frames_path = getattr(args, f"{from_split}_extracted_frames_path")
    from_narrated_actions_dict = load_narrated_actions_dict(from_frames_path)

    for video_uid, count in tqdm(split["videos"].items(), desc=f"Fixing {to_split}"):
        to_frame_dirs = glob.glob(to_frames_path + f"/{video_uid}|*")
        if len(to_frame_dirs) != count:
            # we're missing some frame directories from the to split.
            # move them from the from split
            logging.debug(
                f"Missing {to_split} frame dirs for {video_uid}. "
                f"Checking {from_split}..."
            )
            from_frame_dirs = glob.glob(f"{from_frames_path}/{video_uid}|*")
            if len(from_frame_dirs) != count:
                raise RuntimeError(
                    f"Missing {to_split} frame dirs for {video_uid} not found in "
                    f"{from_split}."
                )
            logging.debug(
                f"Missing {to_split} frame dirs for {video_uid} found in {from_split}. "
                "Moving..."
            )
            for from_frame_dir in from_frame_dirs:
                logging.debug(f"Moving {from_frame_dir} to {to_frames_path}")
                if args.dry_run:
                    logging.info("Dry run. Not actually moving.")
                    continue
                shutil.move(from_frame_dir, to_frames_path)
                frame_dir = Path(from_frame_dir).name
                to_narrated_actions_dict[frame_dir] = from_narrated_actions_dict.pop(
                    frame_dir
                )

    # write narrated_actions.csv files
    logging.info(f"Updating narrated_actions.csv in {to_frames_path}")
    if args.dry_run:
        logging.info("Dry run. Not actually updating.")
    else:
        dump_narrated_actions_dict(to_frames_path, to_narrated_actions_dict)

    logging.info(f"Updating narrated_actions.csv in {from_frames_path}")
    if args.dry_run:
        logging.info("Dry run. Not actually updating.")
    else:
        dump_narrated_actions_dict(from_frames_path, from_narrated_actions_dict)


def verify_frame_dirs(args: argparse.Namespace, split: str) -> None:
    # first verify that narrated_actions.csv agrees with the frame dirs
    frames_path = getattr(args, f"{split}_extracted_frames_path")
    narrated_actions_dict = load_narrated_actions_dict(frames_path)
    frame_dirs = {Path(path).name for path in glob.glob(frames_path + "/*|*")}
    frame_dirs_minus_narrated_actions_dict = frame_dirs - narrated_actions_dict.keys()
    if len(frame_dirs_minus_narrated_actions_dict) > 0:
        logging.warning(
            f"{split}: Following frame dirs exist in {frames_path} "
            f"but not in narrated_actions.csv: {frame_dirs_minus_narrated_actions_dict}"
        )
    narrated_actions_dict_minus_frame_dirs = narrated_actions_dict.keys() - frame_dirs
    if len(narrated_actions_dict_minus_frame_dirs) > 0:
        logging.warning(
            f"{split}: Following frame dirs exist in narrated_actions.csv "
            f"but not in {frames_path}: {narrated_actions_dict_minus_frame_dirs}"
        )

    # next verify that narrated_actions.csv agrees with fho_main_{split}.json
    with open(f"{args.split_output_path}/fho_main_{split}.json") as f:
        split_data = json.load(f)
    narrated_actions_counter = Counter(
        narrated_action["video_uid"]
        for _, narrated_action in narrated_actions_dict.items()
    )

    for video_uid, count in tqdm(
        split_data["videos"].items(), desc=f"Verifying {split}"
    ):
        if narrated_actions_counter[video_uid] != count:
            logging.warning(
                f"{split}: Narrated action counts differ between narrated_actions.csv "
                f"and fho_main_{split}.json for {video_uid}"
            )

    # finally verify that fho_main_{split}.json agrees with the frame dirs
    frame_dirs_counter = Counter(
        Path(path).name.split("|")[0] for path in glob.glob(frames_path + "/*|*")
    )
    for video_uid, count in split_data["videos"].items():
        if frame_dirs_counter[video_uid] != count:
            logging.warning(
                f"{split}: Narrated action counts differ between the frame dirs "
                f"and fho_main_{split}.json for {video_uid}"
            )


if not args.verify_only:
    move_frame_dirs(args, "val", "train")
    move_frame_dirs(args, "train", "val")
if args.verify_only or not args.dry_run:
    verify_frame_dirs(args, "train")
    verify_frame_dirs(args, "val")
else:
    logging.info("Dry run. Skipping verification.")
