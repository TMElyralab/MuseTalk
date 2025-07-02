#!/usr/bin/env python3
"""
Prepare preset avatars for WebSocket service.

This script preprocesses videos to generate all intermediate data needed
for real-time lip-sync inference, then copies the data to the WebSocket service.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def prepare_avatars(config_path: str, output_dir: str = None):
    """
    Prepare preset avatars by running realtime_inference.py with preparation=True
    
    Args:
        config_path: Path to YAML config file with avatar definitions
        output_dir: Optional output directory (default: results/avatars/)
    """
    
    # Validate paths
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Change to StreamingMuseTalk root directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"🎭 Preparing preset avatars from config: {config_path}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Run realtime_inference.py with the config
    cmd = f"python scripts/realtime_inference.py --inference_config {config_path}"
    print(f"🚀 Running command: {cmd}")
    
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ Avatar preparation failed with exit code: {exit_code}")
        return False
    
    print("✅ Avatar preparation completed successfully!")
    
    # If output directory specified, copy results there
    if output_dir:
        copy_to_websocket_service(output_dir)
    
    return True


def copy_to_websocket_service(websocket_avatars_dir: str = "websocket_service/avatars"):
    """
    Copy prepared avatar data to WebSocket service directory
    
    Args:
        websocket_avatars_dir: Target directory for WebSocket service avatars
    """
    
    source_dir = "results/avatars"
    target_dir = websocket_avatars_dir
    
    print(f"📋 Copying avatar data from {source_dir} to {target_dir}")
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy each avatar directory
    for avatar_id in os.listdir(source_dir):
        source_avatar = os.path.join(source_dir, avatar_id)
        target_avatar = os.path.join(target_dir, avatar_id)
        
        if os.path.isdir(source_avatar):
            print(f"  📁 Copying {avatar_id}...")
            
            # Remove existing target if it exists
            if os.path.exists(target_avatar):
                shutil.rmtree(target_avatar)
            
            # Copy the entire avatar directory
            shutil.copytree(source_avatar, target_avatar)
            
            print(f"  ✅ {avatar_id} copied successfully")
    
    print("✅ All avatar data copied to WebSocket service!")
    return True


def list_prepared_avatars():
    """List all prepared avatars and their info"""
    
    avatars_dir = "results/avatars"
    
    if not os.path.exists(avatars_dir):
        print("❌ No avatars directory found. Run preparation first.")
        return
    
    print("📋 Prepared Avatars:")
    print("-" * 50)
    
    for avatar_id in os.listdir(avatars_dir):
        avatar_path = os.path.join(avatars_dir, avatar_id)
        
        if os.path.isdir(avatar_path):
            info_file = os.path.join(avatar_path, "avator_info.json")
            coords_file = os.path.join(avatar_path, "coords.pkl")
            latents_file = os.path.join(avatar_path, "latents.pt")
            
            # Check if essential files exist
            files_exist = {
                "info": os.path.exists(info_file),
                "coords": os.path.exists(coords_file),
                "latents": os.path.exists(latents_file)
            }
            
            status = "✅ Ready" if all(files_exist.values()) else "❌ Incomplete"
            
            print(f"  {avatar_id}: {status}")
            
            if not all(files_exist.values()):
                missing = [k for k, v in files_exist.items() if not v]
                print(f"    Missing: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare preset avatars for WebSocket service")
    parser.add_argument(
        "--config", 
        default="configs/inference/preset_avatars.yaml",
        help="Path to avatar config YAML file"
    )
    parser.add_argument(
        "--copy-to-websocket", 
        action="store_true",
        help="Copy prepared data to websocket_service/avatars/"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List prepared avatars"
    )
    parser.add_argument(
        "--websocket-dir",
        default="websocket_service/avatars",
        help="WebSocket service avatars directory"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_prepared_avatars()
        return
    
    # Prepare avatars
    output_dir = args.websocket_dir if args.copy_to_websocket else None
    success = prepare_avatars(args.config, output_dir)
    
    if success:
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Start the WebSocket service: cd websocket_service && python -m uvicorn src.server:app")
        print("2. Open frontend: http://localhost:8000")
        print("3. Select an avatar and start streaming!")
    else:
        print("\n❌ Setup failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()