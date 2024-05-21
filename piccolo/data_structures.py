from dataclasses import dataclass


@dataclass(slots=True)
class PairRecord:
    text: str
    text_pos: str


@dataclass(slots=True)
class PairNegRecord:
    text: str
    text_pos: str
    text_neg: list


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
    label: int


@dataclass(slots=True)
class PairCLSRecord:
    text: str
    text_label: str
