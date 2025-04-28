#!/usr/bin/env python3

import os
import sys
import glob
import cv2
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

class VideoChecker:
    def __init__(self, video_dir: str, log_file: str = "video_check_log.txt"):
        self.video_dir = video_dir
        self.log_file = log_file
        self.total_videos = 0
        self.error_videos = 0
        self.warning_videos = 0
        self.checked_videos = 0
        self.insufficient_frames_videos = []  # Store videos with insufficient frames
        
        # Create log file
        with open(self.log_file, 'w') as f:
            f.write(f"Video Check Report - {datetime.now()}\n")
            f.write("=" * 42 + "\n")
    
    def log_message(self, message: str):
        """Write log message"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def check_video(self, video_path: str) -> bool:
        """Check a single video file"""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                self.log_message(f"ERROR: {video_path} - Failed to open video file")
                return False
            
            # Get video information
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Check frame count
            if frame_count < 2:
                error_msg = f"ERROR: {video_path} - Insufficient frames (count: {frame_count})"
                self.log_message(error_msg)
                self.insufficient_frames_videos.append({
                    'path': video_path,
                    'frame_count': frame_count,
                    'fps': fps,
                    'resolution': f"{width}x{height}",
                    'duration': f"{duration:.2f}s"
                })
                return False
            
            # Release video object
            cap.release()
            
            return True
            
        except Exception as e:
            self.log_message(f"ERROR: {video_path} - Exception occurred: {str(e)}")
            return False
    
    def check_directory_recursive(self, dir_path: str, pbar: tqdm):
        """Recursively check directories"""
        # Check video files in current directory
        video_files = glob.glob(os.path.join(dir_path, "*.mp4"))
        for video_file in video_files:
            if not self.check_video(video_file):
                self.error_videos += 1
            pbar.update(1)
        
        # Recursively check subdirectories
        for subdir in os.listdir(dir_path):
            full_path = os.path.join(dir_path, subdir)
            if os.path.isdir(full_path):
                self.check_directory_recursive(full_path, pbar)
    
    def get_total_videos(self) -> int:
        """Get total number of video files"""
        return len(glob.glob(os.path.join(self.video_dir, "**/*.mp4"), recursive=True))
    
    def run_check(self):
        """Run the check"""
        self.log_message("Starting video check...")
        self.log_message("=" * 42)
        
        # Get total number of videos
        self.total_videos = self.get_total_videos()
        print(f"Found {self.total_videos} video files to check")
        
        # Create progress bar
        with tqdm(total=self.total_videos, desc="Checking videos", unit="video") as pbar:
            # Start recursive check
            self.check_directory_recursive(self.video_dir, pbar)
        
        # Output statistics
        self.log_message("=" * 42)
        self.log_message(f"Check completed at {datetime.now()}")
        self.log_message(f"Total videos checked: {self.total_videos}")
        self.log_message(f"Videos with insufficient frames: {len(self.insufficient_frames_videos)}")
        
        # Display results
        print("\nCheck completed. Results saved to", self.log_file)
        print(f"Total videos checked: {self.total_videos}")
        print(f"Videos with insufficient frames: {len(self.insufficient_frames_videos)}")
        
        # Print detailed information about videos with insufficient frames
        if self.insufficient_frames_videos:
            print("\nVideos with insufficient frames (< 2 frames):")
            print("=" * 80)
            print(f"{'Video Path':<50} {'Frames':<8} {'FPS':<8} {'Resolution':<12} {'Duration':<10}")
            print("-" * 80)
            for video in self.insufficient_frames_videos:
                print(f"{video['path']:<50} {video['frame_count']:<8} {video['fps']:<8.2f} {video['resolution']:<12} {video['duration']:<10}")
            print("=" * 80)

def main():
    # Set video data directory
    video_dir = "/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Video_data/planning_task/rt_frames_success"
    
    # Create checker and run
    checker = VideoChecker(video_dir)
    checker.run_check()

if __name__ == "__main__":
    main() 