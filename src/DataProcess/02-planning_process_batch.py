import os
import json
import uuid
import logging
import time
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
file_name = "planning_remaining_steps_task_video"
OPENAI_API_KEY = "sk-**"
MODEL = "gpt-4.1-nano-2025-04-14"
BATCH_ID_FILE = f"Output/{file_name}_batches/submitted_batches.json"
BATCH_LOG_FILE = f"Output/{file_name}_batches/batch_status_log.txt"

def submit_batch_to_openai(jsonl_path):
    """
    Upload a JSONL file and submit a batch job to OpenAI
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        uploaded_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        logger.info(f"📁 Uploaded file: {uploaded_file.id}")

        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Batch for file {os.path.basename(jsonl_path)}"}
        )
        logger.info(f"🚀 Submitted batch job: {batch_job.id}")

        with open(BATCH_ID_FILE, 'a') as batch_id_file:
            batch_id_file.write(f"{batch_job.id}\n")
    except Exception as e:
        logger.error(f"Failed to submit batch for {jsonl_path}: {e}")

def track_and_download_batches():
    """
    Load batch IDs from submitted_batches.json and download completed batch results
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(BATCH_ID_FILE, 'r') as batch_id_file:
            batch_ids = batch_id_file.read().splitlines()

        for batch_id in batch_ids:
            try:
                batch = client.batches.retrieve(batch_id)
                if batch.status == "completed":
                    result = client.files.content(batch.output_file_id)
                    with open(f"result_{batch_id}.json", 'w', encoding='utf-8') as result_file:
                        result_file.write(result)
                    logger.info(f"✅ Downloaded result for batch: {batch_id}")
                else:
                    logger.warning(f"⚠️ Batch {batch_id} is not completed. Status: {batch.status}")
            except Exception as e:
                with open(BATCH_LOG_FILE, 'a') as log_file:
                    log_file.write(f"{datetime.now()}: Failed to retrieve or download batch {batch_id}: {e}\n")
                logger.error(f"Failed to retrieve or download batch {batch_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to track and download batches: {e}")

def generate_and_submit_batches(input_file, output_dir, batch_size=10000):
    """
    Split data into multiple JSONL files and submit each as a batch job
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    logger.info(f"🔢 Splitting {total} items into batches of {batch_size}")

    for i in tqdm(range(0, total, batch_size), desc="Batch Progress", unit="batch"):
        batch_data = data[i:i+batch_size]
        jsonl_path = os.path.join(output_dir, f"batch_{i//batch_size + 1}.jsonl")

        with open(jsonl_path, 'w', encoding='utf-8') as fout:
            for j, item in enumerate(batch_data):
                # import pdb; pdb.set_trace()
                custom_id = str(item.get("problem_id", f"request-{i+j}-{uuid.uuid4().hex[:8]}"))
                prompt = f"""Given the following task and its solution, provide a detailed step-by-step reasoning process.

Task: {item['problem']}
Solution: {item['solution']}

Please provide:
1. A breakdown of the task into logical steps
2. For each step, explain why it's necessary
3. If the task is about "next step", clearly identify what the next step should be and why
4. Ensure the final step aligns with the provided solution

Format your response as:
step 1: [description]
reasoning: [why this step is needed]

step 2: [description]
reasoning: [why this step is needed]

[continue for all steps]

Final step: [what needs to be done next]
reasoning: [why this is the correct next step]"""

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that provides detailed step-by-step reasoning for robotic tasks."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 500
                    }
                }
                fout.write(json.dumps(request, ensure_ascii=False) + "\n")

        logger.info(f"✅ Created batch file: {jsonl_path}")
        submit_batch_to_openai(jsonl_path)

    logger.info("🎉 All batches generated and submitted.")

if __name__ == "__main__":
    
    input_filename = f"Output/{file_name}.json"
    output_jsonl_dir = f"Output/{file_name}_batches"

    generate_and_submit_batches(input_filename, output_jsonl_dir, batch_size=10000)
    track_and_download_batches()