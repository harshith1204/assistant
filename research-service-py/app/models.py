from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, HttpUrl, Field


class Evidence(BaseModel):
    quote: str
    url: HttpUrl
    published: Optional[str] = None


class Finding(BaseModel):
    title: str
    summary: str
    evidence: List[Evidence]
    confidence: float
    recency: Optional[str] = None


class RICE(BaseModel):
    reach: int
    impact: float
    confidence: float
    effort: int


class Idea(BaseModel):
    idea: str
    RICE: RICE


class AttachmentRef(BaseModel):
    type: Literal["pdf", "screenshot", "other"] = "other"
    path: str


class ResearchBrief(BaseModel):
    briefId: str
    query: str
    date: str
    entities: List[str]
    keyQuestions: List[str]
    findings: List[Finding]
    ideas: List[Idea]
    attachments: List[AttachmentRef] = Field(default_factory=list)
    summary: Optional[str] = None


class KR(BaseModel):
    metric: str
    target: float


class OKR(BaseModel):
    objective: str
    keyResults: List[KR]


class PlanTask(BaseModel):
    id: str
    title: str
    assignee: str


class Initiative(BaseModel):
    id: str
    title: str
    tasks: List[PlanTask]


class PlanJSON(BaseModel):
    okrs: List[OKR]
    initiatives: List[Initiative]


class RunRequest(BaseModel):
    query: str
    scope: Optional[List[str]] = None
    geo: Optional[List[str]] = None
    freshness: Optional[dict | str] = None


class SaveAttachment(BaseModel):
    type: Literal["pdf", "screenshot", "other"] = "other"
    url: HttpUrl
    name: Optional[str] = None


class SaveRequest(BaseModel):
    briefId: str
    crmRef: Optional[dict] = None
    pmsRef: Optional[dict] = None
    attachments: Optional[List[SaveAttachment]] = None


class IdeasToPlanRequest(BaseModel):
    briefId: str
    selections: List[int]


class SubscribeRequest(BaseModel):
    query: str
    cadence: Literal["weekly", "monthly"]
    geo: Optional[List[str]] = None
    scope: Optional[List[str]] = None
