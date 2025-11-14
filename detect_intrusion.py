import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from intrusion_detector import IntrusionDetector
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intrusion Detection System")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--zones", default="restricted_zones.json", 
                       help="Path to zones JSON file")
    parser.add_argument("--model", default="yolov8n.pt", 
                       help="Path to YOLO model file (will download if not found)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "0"],
                       help="Device to run inference on")
    parser.add_argument("--output", help="Path to save output video")
    parser.add_argument("--no-preview", action="store_true", 
                       help="Disable preview window")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        sys.exit(1)
    
    try:
        detector = IntrusionDetector(args.video, args.zones, args.model, args.device)
        detector.run(args.output, show_preview=not args.no_preview)
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


