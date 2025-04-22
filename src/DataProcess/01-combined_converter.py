import os
import json
import re
import cv2
from tqdm import tqdm

# Configuration
file_name = "success_(negative)_task" #"future_prediction_task"
IMAGE_ROOT = "images/mnt/hpfs/baaiei/jyShi/ShareRobot/planning/"
VIDEO_OUTPUT_DIR = f"Video_data/{file_name}"
VIDEO_FPS = 5  # Video frames per second

def convert_formatB_to_formatA(b_record, problem_id):
    """
    Convert a single record from format B to format A
    """
    # Extract human and GPT text from conversations
    human_text = ""
    gpt_text = ""
    for conv in b_record.get("conversations", []):
        if conv.get("from") == "human":
            human_text = conv.get("value", "")
        elif conv.get("from") == "gpt":
            gpt_text = conv.get("value", "")

    # Clean human text by removing <image> placeholders
    cleaned_human = re.sub(r"<image>\s*", "", human_text).strip()
    if "\n" in cleaned_human:
        parts = [p.strip() for p in cleaned_human.split("\n") if p.strip()]
        if parts:
            cleaned_human = parts[-1]

    # Construct process and solution fields
    process_text = f"<think>Human: {human_text}\nGPT: {gpt_text}</think>"
    solution_text = f"<answer>{gpt_text}</answer>"
    path_list = b_record.get("image", [])

    # Create format A record
    formatA_record = {
        "problem_id": problem_id,
        "problem": cleaned_human,
        "data_type": "image",
        "problem_type": "free-form",
        "options": [],
        "process": process_text,
        "solution": solution_text,
        "path": path_list,
        "data_source": "RoboLogicTask"
    }

    return formatA_record

def get_available_video_path(base_dir, base_name):
    """Get an available video file path to avoid duplicates"""
    index = 0
    while True:
        video_name = f"{base_name}_{index}.mp4"
        full_path = os.path.join(base_dir, video_name)
        if not os.path.exists(full_path):
            return full_path, video_name
        index += 1

def ensure_dir_exists(path):
    """Ensure directory exists, create if it doesn't"""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def remove_image_tags(process_str):
    """Remove image tags from process string"""
    x = process_str.replace("<image> ", "")
    x = x.replace("\n", "")
    x = x.replace("<image>", "")
    return x

def convert_images_to_video(data):
    """Convert image sequences to videos"""
    new_data = []
    
    for item in tqdm(data, desc="Converting images to videos", unit="item"):
        if item["data_type"] != "image":
            new_data.append(item)
            continue

        image_paths = item["path"]
        # Sort frames by number
        image_paths.sort(key=lambda x: int(x.split("frame_")[-1].split(".")[0]))

        if not image_paths:
            print(f"Warning: No image paths found for problem_id={item['problem_id']}")
            continue

        # Set up video output path
        video_subdir = os.path.dirname(image_paths[0])
        video_output_subdir = os.path.join(VIDEO_OUTPUT_DIR, video_subdir)
        os.makedirs(video_output_subdir, exist_ok=True)
        video_path, video_filename = get_available_video_path(video_output_subdir, "video")

        # Read and process images
        full_paths = [os.path.join(IMAGE_ROOT, p) for p in image_paths]
        images = [cv2.imread(p) for p in full_paths if cv2.imread(p) is not None]

        if len(images) == 0:
            print(f"Warning: No valid images found for problem_id={item['problem_id']}")
            continue

        # Create video
        height, width, _ = images[0].shape
        ensure_dir_exists(video_path)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (width, height))
        for img in images:
            out.write(img)
        out.release()

        # Update record
        item["path"] = os.path.join(video_subdir, video_filename)
        item["data_type"] = "video"
        item["process"] = remove_image_tags(item["process"])
        new_data.append(item)

    return new_data

def process_pipeline(input_file, intermediate_output, final_output):
    """Main processing pipeline"""
    # Step 1: Convert format B to format A
    print("Converting format B to format A...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON file should contain a list of records.")

    output_data = []
    count = 0
    for index, record in enumerate(data):
        converted_record = convert_formatB_to_formatA(record, index)
        output_data.append(converted_record)
        # count += 1
        # if count > 100:
        #     break

    # Save intermediate format A
    with open(intermediate_output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Step 1 complete: Saved format A JSON to {intermediate_output}")

    # Step 2: Convert images to videos
    print("\nConverting images to videos...")
    final_data = convert_images_to_video(output_data)

    # Save final output
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Step 2 complete: Saved final JSON with video paths to {final_output}")

if __name__ == "__main__":
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define file paths
    input_filename = os.path.join(current_dir, f"jsons/{file_name}.json")
    intermediate_filename = os.path.join(current_dir, f"./Output/{file_name}_format_gpt.json")
    final_filename = os.path.join(current_dir, f"./Output/{file_name}_video.json")
    
    # Run the pipeline
    process_pipeline(input_filename, intermediate_filename, final_filename) 