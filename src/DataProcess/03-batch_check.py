import time
import json
import os
from openai import OpenAI

OPENAI_API_KEY = "sk-proj-**"
client = OpenAI(api_key=OPENAI_API_KEY)

# 输出目录配置
file_name = "planning_with_context_task_video"
OUTPUT_DIR = f"/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Output/{file_name}_gpt_process_output"
POLL_INTERVAL = 10  # 每隔多少秒轮询一次
SUBMITTED_BATCHES_FILE = f"/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Output/{file_name}_batches/submitted_batches.json"
BATCH_STATUS_LOG_FILE = os.path.join(OUTPUT_DIR, "batch_status_log.txt")
FINAL_MERGED_JSON = os.path.join(OUTPUT_DIR, "final_merged_output.json")

def normalize_batch_ids_file(file_path):
    """
    规范化batch ID文件的格式，将其转换为标准JSON格式
    """
    try:
        # 读取原始文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 将内容按行分割并清理
        batch_ids = [line.strip() for line in content.split('\n') if line.strip()]
        
        # 创建临时文件路径
        temp_file = file_path + '.temp'
        
        # 将规范化的JSON写入临时文件
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(batch_ids, f, indent=2)
        
        # 替换原文件
        os.replace(temp_file, file_path)
        print(f"✅ Normalized batch IDs file format: {file_path}")
        return batch_ids
        
    except Exception as e:
        print(f"⚠️ Error normalizing batch IDs file: {e}")
        return None

def load_batch_ids(file_path):
    """
    加载并验证batch ID文件
    """
    try:
        # 首先尝试直接作为JSON读取
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                batch_ids = json.load(f)
                if isinstance(batch_ids, list):
                    return batch_ids
            except json.JSONDecodeError:
                print("📝 Batch IDs file is not in JSON format, attempting to normalize...")
                return normalize_batch_ids_file(file_path)
    except Exception as e:
        print(f"❌ Error loading batch IDs file: {e}")
        return None

def ensure_output_dir():
    """确保输出目录存在"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

def wait_for_batch(batch_id):
    print(f"⏳ Waiting for batch {batch_id} to complete...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"📦 Status: {status} | Completed: {batch.request_counts.completed} / {batch.request_counts.total}")
        if status == "completed":
            print("✅ Batch completed!")
            return batch.output_file_id
        elif status == "failed":
            print("❌ Batch failed.")
            if batch.error_file_id:
                print("Error file ID:", batch.error_file_id)
            return None
        elif status == "expired":
            print("⚠️ Batch expired before completion.")
            return None
        time.sleep(POLL_INTERVAL)

def download_output_file(file_id, output_path):
    print(f"⬇️ Downloading output file {file_id}...")
    response = client.files.content(file_id)
    content = response.text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📁 Saved output to {output_path}")

def merge_jsonl_to_json(jsonl_path, output_json):
    print(f"🔄 Merging {jsonl_path} into structured JSON...")
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                reasoning = obj.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                results.append({"custom_id": custom_id, "reasoning": reasoning})
            except Exception as e:
                print(f"⚠️ Error parsing line: {e}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Merged results saved to {output_json}")
    return results

def merge_all_json_files(batch_ids):
    """
    按照原始batch_ids的顺序合并所有JSON文件
    """
    print("🔄 Starting final merge of all JSON files...")
    all_results = []
    
    for batch_id in batch_ids:
        json_file = os.path.join(OUTPUT_DIR, f"merged_{batch_id}.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    batch_results = json.load(f)
                    all_results.extend(batch_results)
                print(f"✅ Successfully merged {json_file}")
            except Exception as e:
                print(f"⚠️ Error merging {json_file}: {e}")
        else:
            print(f"⚠️ Warning: {json_file} not found")
    
    # 保存最终合并的结果
    with open(FINAL_MERGED_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"🎉 Final merged file saved as {FINAL_MERGED_JSON}")
    print(f"📊 Total entries: {len(all_results)}")
    return all_results

def track_and_download_batches():
    # 确保输出目录存在
    ensure_output_dir()
    
    if not os.path.exists(SUBMITTED_BATCHES_FILE):
        print(f"⚠️ {SUBMITTED_BATCHES_FILE} does not exist.")
        return
    
    # 加载并规范化batch IDs
    batch_ids = load_batch_ids(SUBMITTED_BATCHES_FILE)
    if not batch_ids:
        print("❌ Failed to load batch IDs. Exiting...")
        return
    
    print(f"📋 Loaded {len(batch_ids)} batch IDs")
    
    with open(BATCH_STATUS_LOG_FILE, "w", encoding="utf-8") as log_file:
        for batch_id in batch_ids:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            log_file.write(f"Batch ID: {batch_id} | Status: {status}\n")
            print(f"Batch ID: {batch_id} | Status: {status}")
            time.sleep(POLL_INTERVAL)

            if status == "completed":
                output_file_id = batch.output_file_id
                jsonl_path = os.path.join(OUTPUT_DIR, f"batch_{batch_id}.jsonl")
                json_path = os.path.join(OUTPUT_DIR, f"merged_{batch_id}.json")
                download_output_file(output_file_id, jsonl_path)
                merge_jsonl_to_json(jsonl_path, json_path)
    
    # 在所有批次处理完成后，执行最终合并
    if batch_ids:
        print("\n🔄 All batches processed, starting final merge...")
        merge_all_json_files(batch_ids)
    else:
        print("\n⚠️ No completed batches to merge")

if __name__ == "__main__":
    track_and_download_batches()