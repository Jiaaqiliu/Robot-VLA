import os
import json
import glob
import shutil
from datetime import datetime
from tqdm import tqdm

# 配置文件路径
file_name = "past_description_task_video"
MERGED_DIR = f"/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Output/{file_name}_gpt_process_output"
PLANNING_TASK_FILE = f"/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Output/{file_name}.json"
OUTPUT_FILE = f"/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Output/{file_name}_update.json"
BACKUP_DIR = f"/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Output/backups"

def create_backup():
    """
    创建原始文件的备份
    """
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"planning_task_video_backup_{timestamp}.json")
    
    try:
        shutil.copy2(PLANNING_TASK_FILE, backup_file)
        print(f"📦 Created backup at: {backup_file}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to create backup: {e}")
        return False

def load_merged_files():
    """
    加载所有merged_batch*.json文件并合并其内容
    """
    print("🔍 Loading merged batch files...")
    all_reasoning = {}
    merged_files = glob.glob(os.path.join(MERGED_DIR, "merged_batch_*.json"))
    
    for file_path in tqdm(merged_files, desc="Processing batch files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                for item in batch_data:
                    # 将custom_id转换为整数
                    try:
                        problem_id = int(item['custom_id'])
                        reasoning = item['reasoning']
                        # 添加<think>标签
                        formatted_reasoning = f"<think>{reasoning}</think>"
                        all_reasoning[problem_id] = formatted_reasoning
                    except ValueError as e:
                        print(f"⚠️ Error converting custom_id in {file_path}: {e}")
                        continue
        except Exception as e:
            print(f"⚠️ Error processing file {file_path}: {e}")
            continue
    
    print(f"✅ Loaded reasoning for {len(all_reasoning)} items")
    return all_reasoning

def update_planning_task(reasoning_data):
    """
    更新planning_task_video.json文件中的process字段并保存为新文件
    """
    print("\n📝 Updating planning task data...")
    
    try:
        # 加载原始规划任务文件
        with open(PLANNING_TASK_FILE, 'r', encoding='utf-8') as f:
            planning_data = json.load(f)
        
        # 记录更新统计
        updated_count = 0
        total_count = len(planning_data)
        not_updated = []  # 记录未更新的problem_id
        
        # 更新process字段
        for item in planning_data:
            problem_id = item.get('problem_id')
            if problem_id is not None and problem_id in reasoning_data:
                item['process'] = reasoning_data[problem_id]
                updated_count += 1
            else:
                not_updated.append(problem_id)
        
        # 保存更新后的文件
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(planning_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully updated {updated_count} out of {total_count} items")
        print(f"💾 Saved to: {OUTPUT_FILE}")
        
        # 如果有未更新的项，打印详细信息
        if not_updated:
            print(f"\n⚠️ Warning: {len(not_updated)} items were not updated")
            print(f"📋 Not updated problem_ids: {not_updated[:10]}{'...' if len(not_updated) > 10 else ''}")
            
    except Exception as e:
        print(f"❌ Error updating planning task file: {e}")
        return False
    
    return True

def main():
    """
    主函数
    """
    print("🚀 Starting reasoning merge process...")
    
    # 创建原始文件的备份
    if not create_backup():
        if not input("⚠️ Backup failed. Continue anyway? (y/N): ").lower().startswith('y'):
            print("❌ Process aborted by user.")
            return
    
    # 加载所有reasoning数据
    reasoning_data = load_merged_files()
    
    if not reasoning_data:
        print("❌ No reasoning data found. Exiting...")
        return
    
    # 更新planning task文件
    success = update_planning_task(reasoning_data)
    
    if success:
        print("\n✨ Process completed successfully!")
        print(f"📁 Original file: {PLANNING_TASK_FILE}")
        print(f"📁 Updated file: {OUTPUT_FILE}")
    else:
        print("\n❌ Process completed with errors!")

if __name__ == "__main__":
    main() 