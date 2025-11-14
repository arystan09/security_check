import json
import os
from typing import List, Dict, Tuple


class ZoneManager:
    def __init__(self, json_path: str = "restricted_zones.json"):
        self.json_path = json_path
        self.zones: List[Dict] = []
        self.load_zones()

    def load_zones(self) -> None:
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.zones = data.get('zones', [])
            except (json.JSONDecodeError, IOError):
                self.zones = []

    def add_zone(self, points: List[Tuple[int, int]], zone_id: int = None) -> None:
        if zone_id is None:
            zone_id = len(self.zones) + 1
        
        zone = {
            'id': zone_id,
            'points': [[int(x), int(y)] for x, y in points]
        }
        self.zones.append(zone)

    def save_zones(self) -> bool:
        try:
            data = {
                'zones': self.zones
            }
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except IOError:
            return False

    def get_zones(self) -> List[Dict]:
        return self.zones

    def clear_zones(self) -> None:
        self.zones = []

