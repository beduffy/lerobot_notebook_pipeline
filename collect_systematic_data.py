#!/usr/bin/env python3
"""
Systematic Data Collection Guide for Generalization Experiments

This script helps you collect data systematically to understand generalization.
It provides guided prompts and protocols for consistent data collection.

Usage:
    python collect_systematic_data.py --task cube_pickup --variations position
    python collect_systematic_data.py --task cube_pickup --variations position,lighting
    python collect_systematic_data.py --task cube_pickup --protocol

Examples:
    # Get data collection protocol for cube pickup
    python collect_systematic_data.py --task cube_pickup --protocol
    
    # Start guided position variation data collection
    python collect_systematic_data.py --task cube_pickup --variations position --episodes 5
    
    # Full systematic collection (position + lighting)
    python collect_systematic_data.py --task cube_pickup --variations position,lighting --episodes 5
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

def print_cube_pickup_protocol():
    """Print the systematic data collection protocol for cube pickup."""
    
    print("🎯 CUBE PICKUP - SYSTEMATIC DATA COLLECTION PROTOCOL")
    print("=" * 60)
    
    print("\n📏 SETUP REQUIREMENTS:")
    print("   1. Consistent table height and surface")
    print("   2. Fixed camera position (wrist camera)")
    print("   3. Red cube (same size/color)")
    print("   4. Pencil/tape for marking positions")
    print("   5. Consistent lighting setup")
    print("   6. Clear workspace (remove distractors)")
    
    print("\n📍 POSITION MARKING SYSTEM:")
    print("   Use pencil to mark 4 cube positions on table:")
    print("   Position A: Front-left   (closest to robot, left side)")
    print("   Position B: Front-right  (closest to robot, right side)")
    print("   Position C: Back-left    (farthest from robot, left side)")
    print("   Position D: Back-right   (farthest from robot, right side)")
    print("   ")
    print("   Distance guidelines:")
    print("   - Front positions: ~20cm from robot base")
    print("   - Back positions: ~35cm from robot base")
    print("   - Left/right: ±15cm from center line")
    
    print("\n🎬 RECORDING PROTOCOL:")
    print("   1. Place cube at marked position")
    print("   2. Start recording")
    print("   3. Pick up cube smoothly")
    print("   4. Lift to consistent height (~20cm)")
    print("   5. Hold for 2-3 seconds")
    print("   6. Place cube in consistent drop zone")
    print("   7. Stop recording")
    print("   8. Repeat for consistency")
    
    print("\n💡 LIGHTING CONDITIONS:")
    print("   Morning:   Natural light, no artificial lights")
    print("   Noon:     Bright natural light")
    print("   Evening:   Natural + artificial lights")
    print("   Lamp:      Only artificial lights")
    
    print("\n📊 RECOMMENDED DATA COLLECTION SCHEDULE:")
    print("   Phase 1: Baseline (same position, same lighting)")
    print("   - Position A, morning light, 10 episodes")
    print("   ")
    print("   Phase 2: Position variation")
    print("   - Position A: 5 episodes")
    print("   - Position B: 5 episodes")
    print("   - Position C: 5 episodes")
    print("   - Position D: 5 episodes")
    print("   ")
    print("   Phase 3: Lighting variation (Position A)")
    print("   - Morning: 3 episodes")
    print("   - Noon: 3 episodes")
    print("   - Evening: 3 episodes")
    print("   - Lamp: 3 episodes")
    print("   ")
    print("   Phase 4: Combined variation")
    print("   - All positions × all lighting conditions: 1-2 episodes each")


def print_collection_checklist(task: str, variations: List[str], episodes_per_condition: int):
    """Print a checklist for data collection."""
    
    print(f"\n✅ DATA COLLECTION CHECKLIST: {task.upper()}")
    print("=" * 50)
    
    if "position" in variations and "lighting" in variations:
        positions = ["A", "B", "C", "D"]
        lighting_conditions = ["morning", "noon", "evening", "lamp"]
        
        print(f"📊 FULL SYSTEMATIC COLLECTION")
        print(f"   Total conditions: {len(positions)} positions × {len(lighting_conditions)} lighting = {len(positions) * len(lighting_conditions)}")
        print(f"   Episodes per condition: {episodes_per_condition}")
        print(f"   Total episodes: {len(positions) * len(lighting_conditions) * episodes_per_condition}")
        print()
        
        for lighting in lighting_conditions:
            print(f"\n🌅 {lighting.upper()} LIGHTING:")
            for pos in positions:
                print(f"   [ ] Position {pos}: {episodes_per_condition} episodes")
    
    elif "position" in variations:
        positions = ["A", "B", "C", "D"]
        
        print(f"📍 POSITION VARIATION COLLECTION")
        print(f"   Total positions: {len(positions)}")
        print(f"   Episodes per position: {episodes_per_condition}")
        print(f"   Total episodes: {len(positions) * episodes_per_condition}")
        print()
        
        for pos in positions:
            print(f"   [ ] Position {pos}: {episodes_per_condition} episodes")
    
    elif "lighting" in variations:
        lighting_conditions = ["morning", "noon", "evening", "lamp"]
        
        print(f"💡 LIGHTING VARIATION COLLECTION")
        print(f"   Total lighting conditions: {len(lighting_conditions)}")
        print(f"   Episodes per condition: {episodes_per_condition}")
        print(f"   Total episodes: {len(lighting_conditions) * episodes_per_condition}")
        print()
        
        for lighting in lighting_conditions:
            print(f"   [ ] {lighting.capitalize()}: {episodes_per_condition} episodes")
    
    print(f"\n📋 BEFORE EACH RECORDING SESSION:")
    print(f"   [ ] Mark positions clearly with pencil")
    print(f"   [ ] Check camera position/focus")
    print(f"   [ ] Verify lighting setup")
    print(f"   [ ] Clear workspace of distractors")
    print(f"   [ ] Test robot movement/calibration")
    
    print(f"\n📁 DATASET ORGANIZATION:")
    print(f"   Suggested naming convention:")
    print(f"   - cube_pickup_baseline (your existing data)")
    print(f"   - cube_pickup_position_var")
    print(f"   - cube_pickup_lighting_var") 
    print(f"   - cube_pickup_combined_var")


def create_experiment_log(output_dir: Path, task: str, variations: List[str]):
    """Create a structured experiment log."""
    
    log_data = {
        "experiment_info": {
            "task": task,
            "variations": variations,
            "start_date": time.strftime("%Y-%m-%d"),
            "protocol_version": "1.0"
        },
        "setup": {
            "robot": "SO-100",
            "camera": "wrist_camera",
            "workspace": "bedroom_table",
            "cube_color": "red"
        },
        "positions": {
            "A": {"description": "Front-left", "coordinates": "~20cm front, 15cm left", "status": "not_started"},
            "B": {"description": "Front-right", "coordinates": "~20cm front, 15cm right", "status": "not_started"},
            "C": {"description": "Back-left", "coordinates": "~35cm front, 15cm left", "status": "not_started"},
            "D": {"description": "Back-right", "coordinates": "~35cm front, 15cm right", "status": "not_started"}
        },
        "lighting_conditions": {
            "morning": {"description": "Natural light, no artificial", "status": "not_started"},
            "noon": {"description": "Bright natural light", "status": "not_started"},
            "evening": {"description": "Natural + artificial", "status": "not_started"},
            "lamp": {"description": "Only artificial lights", "status": "not_started"}
        },
        "sessions": [],
        "notes": []
    }
    
    log_path = output_dir / "experiment_log.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📝 Experiment log created: {log_path}")
    return log_path


def guided_session_start():
    """Interactive guided session start."""
    
    print("\n🎬 GUIDED DATA COLLECTION SESSION")
    print("=" * 40)
    
    # Get session info
    position = input("📍 What position are you recording? (A/B/C/D): ").strip().upper()
    lighting = input("💡 What lighting condition? (morning/noon/evening/lamp): ").strip().lower()
    episode_num = input("📊 Episode number for this condition: ").strip()
    
    print(f"\n🎯 SESSION SETUP:")
    print(f"   Position: {position}")
    print(f"   Lighting: {lighting}")
    print(f"   Episode: #{episode_num}")
    
    input("\n👆 Press Enter when cube is positioned and you're ready to record...")
    
    print("🔴 RECORDING CHECKLIST:")
    print("   1. [ ] Cube placed at marked position")
    print("   2. [ ] Camera focused and clear")
    print("   3. [ ] Robot calibrated and ready")
    print("   4. [ ] Start LeRobot recording")
    print("   5. [ ] Execute smooth pickup")
    print("   6. [ ] Lift to consistent height")
    print("   7. [ ] Hold for 2-3 seconds")
    print("   8. [ ] Place in drop zone")
    print("   9. [ ] Stop recording")
    
    input("\n✅ Press Enter when recording is complete...")
    
    # Get quality assessment
    print("\n📊 QUALITY ASSESSMENT:")
    smooth = input("Was the movement smooth? (y/n): ").strip().lower() == 'y'
    success = input("Was the pickup successful? (y/n): ").strip().lower() == 'y'
    notes = input("Any notes about this episode? ").strip()
    
    session_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "position": position,
        "lighting": lighting,
        "episode_number": episode_num,
        "smooth_movement": smooth,
        "pickup_success": success,
        "notes": notes
    }
    
    print(f"\n✅ Session recorded successfully!")
    return session_data


def main():
    parser = argparse.ArgumentParser(description="Systematic Data Collection Guide")
    parser.add_argument("--task", choices=["cube_pickup"], default="cube_pickup",
                       help="Task to collect data for")
    parser.add_argument("--variations", type=str, default="position",
                       help="Comma-separated list of variations (position,lighting)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Episodes per condition (default: 5)")
    parser.add_argument("--protocol", action="store_true",
                       help="Show data collection protocol")
    parser.add_argument("--checklist", action="store_true",
                       help="Show collection checklist")
    parser.add_argument("--guided", action="store_true",
                       help="Start guided data collection session")
    parser.add_argument("--output-dir", type=str, default="./data_collection_logs",
                       help="Directory for logs and checklists")
    
    args = parser.parse_args()
    
    variations = [v.strip() for v in args.variations.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 SYSTEMATIC DATA COLLECTION GUIDE")
    print("=" * 50)
    print(f"🎯 Task: {args.task}")
    print(f"🔄 Variations: {', '.join(variations)}")
    print(f"📁 Output: {output_dir.absolute()}")
    print()
    
    if args.protocol:
        if args.task == "cube_pickup":
            print_cube_pickup_protocol()
    
    if args.checklist:
        print_collection_checklist(args.task, variations, args.episodes)
        
        # Create experiment log
        log_path = create_experiment_log(output_dir, args.task, variations)
        
        print(f"\n💾 EXPERIMENT TRACKING:")
        print(f"   Log file: {log_path}")
        print(f"   Update this file as you complete each condition")
    
    if args.guided:
        session_data = guided_session_start()
        
        # Save session data
        session_file = output_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"📝 Session data saved: {session_file}")
    
    if not any([args.protocol, args.checklist, args.guided]):
        print("💡 TIP: Use --protocol to see data collection guidelines")
        print("💡 TIP: Use --checklist to get organized collection plan")
        print("💡 TIP: Use --guided for interactive session recording")
    
    print(f"\n🚀 READY TO BUILD SYSTEMATIC DATASETS!")
    print(f"   Remember: Consistency is key for generalization studies")
    print(f"   Document everything - small details matter!")


if __name__ == "__main__":
    main() 