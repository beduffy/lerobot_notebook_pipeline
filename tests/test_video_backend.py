#!/usr/bin/env python3
"""
Simple test to verify video backend fix
"""

import os
import warnings
warnings.filterwarnings("ignore")

# Force pyav backend globally for headless compatibility
os.environ['LEROBOT_VIDEO_BACKEND'] = 'pyav'

# Monkey patch to force pyav backend
try:
    from lerobot.datasets.video_utils import decode_video_frames
    original_decode_video_frames = decode_video_frames
    
    def patched_decode_video_frames(video_path, timestamps, tolerance_s, video_backend=None):
        """Patched version that forces pyav backend."""
        return original_decode_video_frames(video_path, timestamps, tolerance_s, "pyav")
    
    # Replace the function
    import lerobot.datasets.video_utils
    lerobot.datasets.video_utils.decode_video_frames = patched_decode_video_frames
    print("✅ Patched video decoding to use pyav backend")
except Exception as e:
    print(f"⚠️ Could not patch video decoding: {e}")

# Test dataset loading
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("📊 Testing dataset loading with pyav backend...")
    
    dataset = LeRobotDataset("bearlover365/red_cube_always_in_same_place", video_backend="pyav")
    sample = dataset[0]
    print("✅ Successfully loaded dataset with pyav backend!")
    print(f"   Sample keys: {list(sample.keys())}")
    
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    import traceback
    traceback.print_exc() 