import cv2
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from zone_manager import ZoneManager
from geometry_utils import bbox_center, check_person_in_zones

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class AlarmState:
    def __init__(self):
        self.active = False
        self.deactivate_time: Optional[float] = None
        self.deactivate_delay = 3.0

    def activate(self):
        self.active = True
        self.deactivate_time = None

    def check_deactivate(self, current_time: float):
        if self.active and not self.deactivate_time:
            self.deactivate_time = current_time + self.deactivate_delay
        
        if self.deactivate_time and current_time >= self.deactivate_time:
            self.active = False
            self.deactivate_time = None

    def reset(self):
        self.active = False
        self.deactivate_time = None


class IntrusionDetector:
    def __init__(self, video_path: str, zones_json: str = "restricted_zones.json", 
                 model_path: str = "yolov8n.pt", device: str = "cpu"):
        self.video_path = video_path
        self.zone_manager = ZoneManager(zones_json)
        self.zones = self.zone_manager.get_zones()
        self.device = device
        self.alarm_state = AlarmState()
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.person_class_id = 0

    def draw_zones(self, frame):
        for zone in self.zones:
            points = zone['points']
            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                if points:
                    cv2.putText(frame, f"Zone {zone['id']}", 
                               tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)

    def draw_alarm(self, frame):
        if self.alarm_state.active:
            h, w = frame.shape[:2]
            text = "ALARM!"
            font_scale = 2.0
            thickness = 4
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            x = (w - text_width) // 2
            y = (h + text_height) // 2
            
            cv2.putText(frame, text, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (0, 0, 255), thickness)
            cv2.rectangle(frame, (x - 10, y - text_height - 10), 
                         (x + text_width + 10, y + baseline + 10), 
                         (0, 0, 255), 3)

    def process_frame(self, frame, current_time: float):
        results = self.model(frame, classes=[self.person_class_id], 
                           device=self.device, verbose=False)
        
        has_intrusion = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                center = bbox_center((x1, y1, x2, y2))
                intersecting_zones = check_person_in_zones(center, self.zones)
                
                if intersecting_zones:
                    has_intrusion = True
                    color = (0, 0, 255)
                    label = f"Person (Zone {intersecting_zones[0]})"
                else:
                    color = (0, 255, 0)
                    label = f"Person {confidence:.2f}"
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                             color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, center, 5, color, -1)
        
        if has_intrusion:
            self.alarm_state.activate()
        else:
            self.alarm_state.check_deactivate(current_time)
        
        return frame

    def run(self, output_path: Optional[str] = None, show_preview: bool = True):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing video: {self.video_path}")
        print(f"Zones loaded: {len(self.zones)}")
        print(f"Device: {self.device}")
        print("Press 'q' to quit, 's' to save and quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time() - start_time
            frame = self.process_frame(frame, current_time)
            self.draw_zones(frame)
            self.draw_alarm(frame)
            
            status_text = f"Frame: {frame_count}/{total_frames} | Alarm: {'ON' if self.alarm_state.active else 'OFF'}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if show_preview:
                cv2.imshow("Intrusion Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and output_path:
                    print(f"Saving video to {output_path}...")
            
            if writer:
                writer.write(frame)
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"Processing complete. Processed {frame_count} frames.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intrusion Detection System")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--zones", default="restricted_zones.json", 
                       help="Path to zones JSON file")
    parser.add_argument("--model", default="yolov8n.pt", 
                       help="Path to YOLO model file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "0"],
                       help="Device to run inference on")
    parser.add_argument("--output", help="Path to save output video")
    parser.add_argument("--no-preview", action="store_true", 
                       help="Disable preview window")
    
    args = parser.parse_args()
    
    detector = IntrusionDetector(args.video, args.zones, args.model, args.device)
    detector.run(args.output, show_preview=not args.no_preview)

