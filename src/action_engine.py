"""Rule-based action recommendation engine."""
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import IntEnum
from typing import List, Dict


class AlertLevel(IntEnum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class Recommendation:
    level: str
    title: str
    description: str
    actions: List[str]
    urgency: str
    evacuation_required: bool
    confidence: float
    timestamp: str
    details: Dict


class ActionEngine:
    RULES = {
        AlertLevel.NORMAL: {
            'title': 'Normal Operation',
            'description': 'All parameters within safe limits.',
            'actions': [
                'Continue routine monitoring',
                'Log data for trend analysis',
                'Maintain standard patrol schedule'
            ],
            'urgency': 'None',
            'evacuation_required': False
        },
        AlertLevel.WARNING: {
            'title': 'Warning - Elevated Risk',
            'description': 'Abnormal patterns detected. Increase monitoring.',
            'actions': [
                'Increase sensor sampling rate',
                'Notify shift supervisor',
                'Inspect affected zone visually',
                'Verify ventilation and drainage',
                'Prepare evacuation route',
                'Log all sensor deviations'
            ],
            'urgency': 'Within 15 minutes',
            'evacuation_required': False
        },
        AlertLevel.CRITICAL: {
            'title': 'CRITICAL - Immediate Action Required',
            'description': 'Significant anomaly detected. Collapse risk elevated.',
            'actions': [
                'ALERT all personnel in affected zone',
                'Begin controlled evacuation',
                'Activate backup monitoring',
                'Contact mine safety officer',
                'Stage emergency response team',
                'Freeze non-essential operations',
                'Document sensor readings for investigation'
            ],
            'urgency': 'Immediate',
            'evacuation_required': True
        }
    }
    
    def recommend(self, level: int, confidence: float, sensor_data: dict) -> Recommendation:
        try:
            lvl = AlertLevel(level)
        except ValueError:
            lvl = AlertLevel.NORMAL
        rule = self.RULES[lvl]
        details = self._extract_details(sensor_data)
        trend = self._analyze_trend(sensor_data, lvl)
        return Recommendation(
            level=lvl.name,
            title=rule['title'],
            description=rule['description'],
            actions=rule['actions'],
            urgency=rule['urgency'],
            evacuation_required=rule['evacuation_required'],
            confidence=float(confidence),
            timestamp=datetime.utcnow().isoformat() + 'Z',
            details={**details, 'trend': trend}
        )
    
    def _extract_details(self, d: dict) -> dict:
        return {
            'vibration': round(float(d.get('vibration_magnitude', 0)), 4),
            'load_kg': round(float(d.get('load_cell', 0)), 3),
            'displacement': int(d.get('displacement', 0)),
            'water_level': int(d.get('water_level', 0))
        }
    
    def _analyze_trend(self, d: dict, lvl: AlertLevel) -> str:
        vib_rate = float(d.get('vib_change_rate', 0))
        load_rate = float(d.get('load_change_rate', 0))
        if lvl == AlertLevel.CRITICAL:
            return "Rapid deterioration - collapse risk"
        if vib_rate > 0.05 and load_rate > 0.02:
            return "Rapidly deteriorating"
        if vib_rate > 0.02 or abs(load_rate) > 0.01:
            return "Slowly deteriorating"
        return "Stable"
    
    def to_dict(self, rec: Recommendation) -> dict:
        return asdict(rec)