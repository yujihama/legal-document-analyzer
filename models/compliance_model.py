from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Requirement:
    id: str
    text: str
    type: str  # 'requirement' or 'prohibition'
    source_section: str
    concrete_examples: List[str]

@dataclass
class ComplianceMatch:
    requirement_id: str
    regulation_text: str
    compliance_score: float
    explanation: str
    suggestions: List[str]

@dataclass
class ComplianceReport:
    timestamp: datetime
    legal_document_name: str
    internal_document_name: str
    total_requirements: int
    compliant_count: int
    non_compliant_count: int
    matches: List[ComplianceMatch]
    gaps: List[Requirement]
    summary: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'legal_document_name': self.legal_document_name,
            'internal_document_name': self.internal_document_name,
            'total_requirements': self.total_requirements,
            'compliant_count': self.compliant_count,
            'non_compliant_count': self.non_compliant_count,
            'matches': [vars(m) for m in self.matches],
            'gaps': [vars(g) for g in self.gaps],
            'summary': self.summary,
            'recommendations': self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComplianceReport':
        """Create report instance from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            legal_document_name=data['legal_document_name'],
            internal_document_name=data['internal_document_name'],
            total_requirements=data['total_requirements'],
            compliant_count=data['compliant_count'],
            non_compliant_count=data['non_compliant_count'],
            matches=[ComplianceMatch(**m) for m in data['matches']],
            gaps=[Requirement(**g) for g in data['gaps']],
            summary=data['summary'],
            recommendations=data['recommendations']
        )
