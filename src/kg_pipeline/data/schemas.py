from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EntitySpan:
    start: int
    end: int
    label: str
    text: str = ""


@dataclass
class NERSample:
    text: str
    entities: List[EntitySpan] = field(default_factory=list)


@dataclass
class RelationSample:
    text: str
    head: EntitySpan
    tail: EntitySpan
    label: str


@dataclass
class KBEntity:
    entity_id: str
    name: str
    aliases: List[str]
    type: str
    description: str
