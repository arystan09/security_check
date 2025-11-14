import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zone_marker import ZoneMarker

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mark_zones.py <video_path> [json_path]")
        print("Example: python mark_zones.py video.mp4 restricted_zones.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else "restricted_zones.json"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)
    
    marker = ZoneMarker(video_path, json_path)
    marker.run()

