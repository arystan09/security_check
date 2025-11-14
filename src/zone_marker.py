import cv2
import sys
import numpy as np
from typing import List, Tuple, Optional
from zone_manager import ZoneManager


class ZoneMarker:
    def __init__(self, video_path: str, json_path: str = "restricted_zones.json"):
        self.video_path = video_path
        self.zone_manager = ZoneManager(json_path)
        self.current_points: List[Tuple[int, int]] = []
        self.current_zone_id = len(self.zone_manager.get_zones()) + 1
        self.drawing = False
        self.current_frame = None
        self.frame_copy = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            self.drawing = True
            self._redraw_frame()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._redraw_frame()
            if len(self.current_points) > 0:
                temp_points = self.current_points + [(x, y)]
                if len(temp_points) > 1:
                    pts = np.array(temp_points, np.int32)
                    cv2.polylines(self.frame_copy, [pts], False, (0, 255, 0), 2)
                    for pt in temp_points:
                        cv2.circle(self.frame_copy, pt, 5, (0, 255, 0), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def _redraw_frame(self):
        self.frame_copy = self.current_frame.copy()
        
        for zone in self.zone_manager.get_zones():
            points = np.array(zone['points'], np.int32)
            cv2.polylines(self.frame_copy, [points], True, (0, 0, 255), 2)
            for pt in zone['points']:
                cv2.circle(self.frame_copy, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(self.frame_copy, f"Zone {zone['id']}", 
                       tuple(zone['points'][0]), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 2)

        if len(self.current_points) > 0:
            pts = np.array(self.current_points, np.int32)
            if len(pts) > 1:
                cv2.polylines(self.frame_copy, [pts], False, (0, 255, 0), 2)
            for pt in self.current_points:
                cv2.circle(self.frame_copy, pt, 5, (0, 255, 0), -1)

    def _draw_instructions(self, frame):
        instructions = [
            "Controls:",
            "LEFT CLICK - Add point to current zone",
            "ENTER - Finish current zone",
            "C - Clear current zone",
            "S - Save all zones and exit",
            "Q/ESC - Exit without saving",
            "SPACE - Next frame",
            "B - Previous frame"
        ]
        y_offset = 20
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        window_name = "Zone Marker - Press SPACE for next frame, Q to quit"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return

        self.current_frame = frame
        self.frame_copy = frame.copy()
        frame_number = 0

        while True:
            self._redraw_frame()
            display_frame = self.frame_copy.copy()
            self._draw_instructions(display_frame)
            
            status_text = f"Frame: {frame_number} | Current zone: {self.current_zone_id} | Points: {len(self.current_points)}"
            cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                if self.current_points and len(self.current_points) >= 3:
                    self.zone_manager.add_zone(self.current_points, self.current_zone_id)
                    self.current_points = []
                    self.current_zone_id += 1
                if self.zone_manager.save_zones():
                    print(f"Zones saved to {self.zone_manager.json_path}")
                break
            elif key == ord(' '):
                if self.current_points and len(self.current_points) >= 3:
                    self.zone_manager.add_zone(self.current_points, self.current_zone_id)
                    self.current_points = []
                    self.current_zone_id += 1
                ret, frame = cap.read()
                if ret:
                    self.current_frame = frame
                    frame_number += 1
                else:
                    print("End of video")
            elif key == ord('b'):
                if frame_number > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    ret, frame = cap.read()
                    if ret:
                        self.current_frame = frame
                        frame_number -= 1
            elif key == ord('c'):
                self.current_points = []
            elif key == 13:
                if len(self.current_points) >= 3:
                    self.zone_manager.add_zone(self.current_points, self.current_zone_id)
                    self.current_points = []
                    self.current_zone_id += 1
                    print(f"Zone {self.current_zone_id - 1} saved. Continue adding points for next zone.")
                else:
                    print("Need at least 3 points to create a zone")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zone_marker.py <video_path> [json_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else "restricted_zones.json"
    
    marker = ZoneMarker(video_path, json_path)
    marker.run()

