from dataclasses import dataclass

@dataclass(slots=True)
class PairRetriContrastRecord:
    text: str
    text_pos: str
    text_neg: list

@dataclass(slots=True)
class PairClsContrastRecord:
    text: str
    text_pos: str
    text_neg: list

@dataclass(slots=True)
class PairScoredRecord:
    text: str
    text_pair: str
    label: float