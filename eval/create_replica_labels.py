import os
import cv2
import numpy as np
import yaml
import json
from pathlib import Path
from collections import Counter
import os
import glob
from replica_save_labels import create_labelme_annotation, save_annotations_to_json, get_top_labels, save_json_labels
from argparse import ArgumentParser
import re
import random


def get_image_list(directory):
    """Reads and returns the list of image filenames in a directory, sorted by numeric parts of filenames."""
    image_list = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    # Sort the image list based on numeric part extracted from the filename
    return sorted(image_list, key=lambda x: int(re.search(r"\d+", x).group()))


def extract_numeric_part(filename):
    """Extracts the numeric part from the filename."""
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else None


def select_images(img_dir, seed_keys):
    """
    Given an image directory and predefined seed keys,
    select images and return filenames and their indices.
    """
    img_list = get_image_list(img_dir)
    numeric_map = {extract_numeric_part(f): f for f in img_list}

    selected_keys = [k for k in seed_keys if k in numeric_map]
    selected_images = [numeric_map[k] for k in selected_keys]
    selected_indices = [img_list.index(img) for img in selected_images]

    return selected_images, selected_indices


def process_images_labels(
    args,
    selected_imgs,
    selected_idx,
    output_folder,
    img_list_path,
    top_labels,
    output_name,
):
    """
    Convert Replica GT semantic segmentation masks to json annotations.
    Save <img_name>.json and <img_name>.jpg under <output_folder>/<output_name>
    """

    # Assuming seg_feat_dir holds the semantic label folder path
    gt_segmentation_dir = args.seg_file_config.replace(
        "render_config.yaml", "semantic_class"
    )

    os.makedirs(os.path.join(output_folder, output_name), exist_ok=True)

    for i, img_file_name in enumerate(selected_imgs):
        rgb_img = cv2.imread(os.path.join(img_list_path, img_file_name))

        img_name = img_file_name.split("/")[-1].split(".")[0]
        img_numeric_part = extract_numeric_part(img_name)

        seg_label_path = os.path.join(
            gt_segmentation_dir, f"semantic_class_{img_numeric_part}.png"
        )

        seg_label = cv2.imread(
            seg_label_path, cv2.IMREAD_UNCHANGED
        ).astype(np.int32)

        seg_label_resized = cv2.resize(
            seg_label,
            (rgb_img.shape[1], rgb_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        output_json = os.path.join(
            output_folder, output_name, f"{img_name}.json"
        )

        success = save_json_labels(
            args.seg_file_config,
            seg_label_resized,
            output_json,
            img_name,
            selected_idx[i],
            user_label_names=top_labels,
        )

        if success:
            cv2.imwrite(
                os.path.join(output_folder, output_name, f"{img_name}.jpg"),
                rgb_img,
            )

if __name__ == "__main__":
    """
    The script converts the Replica's GT segmentation mask to json annotations to compute scores.
    It saves <img_name>.json under <langslam_dir>/gt/<output_name>

    """

    parser = ArgumentParser(description="prompt any label")

    parser.add_argument(
        "--langslam_dir",
        type=str,
        default=None,
        help="Root dataset folder containing LangSLAM results",
    )
    parser.add_argument(
        "--langsplat_dir",
        type=str,
        default=None,
        help="Root dataset folder containing LangSplat results",
    )
    
    parser.add_argument(
        "--seg_file_config", type=str, help="The path to render_config.yaml, contained in vmap Replica dataset"
    )
    parser.add_argument("--output_name", type=str, default="label", help="Label names for output")

    args = parser.parse_args()

    # If no explicit flags are provided, infer from available directories
    #if not args.use_langslam and not args.use_langsplat:
    args.use_langslam = args.langslam_dir is not None
    args.use_langsplat = args.langsplat_dir is not None
        
    # Step 1: read images list from langsplat and langslam
    # Randonmly selected common images used for all evals
    langslam_seed_keys = [5, 20, 120, 270, 340, 410, 490, 560, 630, 700, 780, 850, 920, 1050, 1410, 1850]
    langsplat_seed_keys = [0, 20, 120, 270, 340, 410, 490, 560, 630, 700, 780, 850, 920, 1050, 1410, 1850]

    # Step 2: get top-frequent labels from Replica GT
    gt_segmentation_dir = args.seg_file_config.replace(
        "render_config.yaml", "semantic_class"
    )

    # Get the top-frequent labels
    top_labels = get_top_labels(args.seg_file_config, gt_segmentation_dir)
    user_label_names = [label[1] for label in top_labels]

    # Step 3: create label files for LangSplat (optional)
    if args.use_langsplat:
        if args.langsplat_dir is None:
            raise ValueError("--use_langsplat set but --langsplat_dir not provided")

        langsplat_img_dir = os.path.join(args.langsplat_dir, "gt")

        langsplat_selected_images, langsplat_indices = select_images(
            langsplat_img_dir, langsplat_seed_keys
        )

        process_images_labels(
            args,
            selected_imgs=langsplat_selected_images,
            selected_idx=langsplat_indices,
            output_folder=args.langsplat_dir,
            img_list_path=langsplat_img_dir,
            top_labels=user_label_names,
            output_name=args.output_name,
        )

    # Step 4: create label files for LangSLAM (optional)
    if args.use_langslam:
        if args.langslam_dir is None:
            raise ValueError("--use_langslam set but --langslam_dir not provided")

        langslam_img_dir = os.path.join(args.langslam_dir, "gt")

        langslam_selected_images, langslam_indices = select_images(
            langslam_img_dir, langslam_seed_keys
        )

        process_images_labels(
            args,
            selected_imgs=langslam_selected_images,
            selected_idx=langslam_indices,
            output_folder=args.langslam_dir,
            img_list_path=langslam_img_dir,
            top_labels=user_label_names,
            output_name=args.output_name,
        )