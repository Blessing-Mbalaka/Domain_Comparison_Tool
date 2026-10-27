from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("SEMANTIC_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
DB_PATH = Path(os.environ.get("SEMANTIC_DASHBOARD_DB", DATA / "dashboard.sqlite3"))
LEGACY_INDEX = ROOT / "index.html"
FRONTEND = ROOT / "frontend" / "dist"
ALLOWED = {".pdf", ".docx", ".txt"}
MODELS = [
    ("mpnet-primary", "MPNet primary", "Legacy-compatible hashed semantic projection"),
    ("bge-robustness", "BGE robustness", "Robustness comparison projection"),
    ("minilm-baseline", "MiniLM baseline", "Compact baseline projection"),
    ("word2vec-googlenews", "Word2Vec (GoogleNews)", "Legacy lexical projection"),
]
STOP = set("a an and are as at be been by for from has have in into is it its of on or that the their this to was were will with using used use across between through".split())
worker_stop = threading.Event()
worker_thread: threading.Thread | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    return db


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS domains(id INTEGER PRIMARY KEY,name TEXT UNIQUE NOT NULL,description TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS categories(id INTEGER PRIMARY KEY,domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE CASCADE,name TEXT NOT NULL,description TEXT NOT NULL DEFAULT '',synonyms TEXT NOT NULL DEFAULT '[]',position INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(domain_id,name));
CREATE TABLE IF NOT EXISTS papers(id INTEGER PRIMARY KEY,title TEXT NOT NULL,filename TEXT,source_path TEXT,text_content TEXT NOT NULL DEFAULT '',status TEXT NOT NULL DEFAULT 'ready',selected INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_categories(paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_comparisons(paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY,paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,chunk_index INTEGER NOT NULL,text TEXT NOT NULL,term_counts TEXT NOT NULL DEFAULT '{}',UNIQUE(paper_id,chunk_index));
CREATE TABLE IF NOT EXISTS embeddings(id INTEGER PRIMARY KEY,entity_type TEXT NOT NULL,entity_id INTEGER NOT NULL,model_id TEXT NOT NULL,vector TEXT NOT NULL,content_hash TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(entity_type,entity_id,model_id));
CREATE TABLE IF NOT EXISTS relationships(id INTEGER PRIMARY KEY,paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,model_id TEXT NOT NULL,semantic_score REAL NOT NULL,coverage REAL NOT NULL,exact_presence REAL NOT NULL,evidence TEXT NOT NULL DEFAULT '[]',top_terms TEXT NOT NULL DEFAULT '[]',updated_at TEXT NOT NULL,UNIQUE(paper_id,category_id,model_id));
CREATE TABLE IF NOT EXISTS legacy_scores(id INTEGER PRIMARY KEY,model_id TEXT NOT NULL,reference_category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,comparison_category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,best_match REAL NOT NULL,all_pairs REAL NOT NULL,source_href TEXT,UNIQUE(model_id,reference_category_id,comparison_category_id));
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'queued',payload TEXT NOT NULL DEFAULT '{}',progress REAL NOT NULL DEFAULT 0,message TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,event_type TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_rel_paper ON relationships(paper_id);
CREATE INDEX IF NOT EXISTS idx_rel_category ON relationships(category_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def migrate_schema(db: sqlite3.Connection) -> None:
    """Add compatible columns to databases created by earlier dashboard versions."""
    columns = {row["name"] for row in db.execute("PRAGMA table_info(relationships)").fetchall()}
    if "concept_presence" not in columns:
        db.execute("ALTER TABLE relationships ADD COLUMN concept_presence REAL NOT NULL DEFAULT 0")
    if "presence_reason" not in columns:
        db.execute("ALTER TABLE relationships ADD COLUMN presence_reason TEXT NOT NULL DEFAULT 'not detected'")
    db.execute("CREATE TABLE IF NOT EXISTS paper_categories(paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,created_at TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_comparisons(paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,created_at TEXT NOT NULL)")
    db.commit()


def emit(db: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> None:
    db.execute("INSERT INTO events(event_type,payload,created_at) VALUES(?,?,?)", (kind, json.dumps(payload), now()))


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", text.lower()) if w not in STOP]


def embed(text: str, model_id: str, dims: int = 192) -> list[float]:
    vec = [0.0] * dims
    toks = tokenize(text)
    features = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for term in features:
        digest = hashlib.blake2b((model_id + "|" + term).encode(), digest_size=16).digest()
        idx = int.from_bytes(digest[:8], "little") % dims
        vec[idx] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 7) for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return max(0.0, min(1.0, (sum(x * y for x, y in zip(a, b)) + 1.0) / 2.0))


def category_text(row: sqlite3.Row) -> str:
    return " ".join([row["name"], row["description"], *json.loads(row["synonyms"] or "[]")])


def chunks(text: str, size: int = 850, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            split = text.rfind(" ", start + size // 2, end)
            end = split if split > start else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return out


def cache_embedding(db: sqlite3.Connection, entity_type: str, entity_id: int, model_id: str, text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).hexdigest()
    row = db.execute("SELECT vector,content_hash FROM embeddings WHERE entity_type=? AND entity_id=? AND model_id=?", (entity_type, entity_id, model_id)).fetchone()
    if row and row["content_hash"] == digest:
        return json.loads(row["vector"])
    vector = embed(text, model_id)
    db.execute("INSERT INTO embeddings(entity_type,entity_id,model_id,vector,content_hash,created_at) VALUES(?,?,?,?,?,?) ON CONFLICT(entity_type,entity_id,model_id) DO UPDATE SET vector=excluded.vector,content_hash=excluded.content_hash,created_at=excluded.created_at", (entity_type, entity_id, model_id, json.dumps(vector), digest, now()))
    return vector


def top_phrases(text: str, limit: int = 12) -> list[dict[str, Any]]:
    toks = tokenize(text)
    counts = Counter(toks)
    counts.update(" ".join(x) for x in zip(toks, toks[1:]))
    return [{"phrase": p, "count": c} for p, c in counts.most_common(limit)]


def evaluate(db: sqlite3.Connection, paper_id: int, category_id: int) -> None:
    paper = db.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    cat = db.execute("SELECT * FROM categories WHERE id=?", (category_id,)).fetchone()
    if not paper or not cat:
        return
    ctext = category_text(cat)
    ptoks = Counter(tokenize(paper["text_content"]))
    terms = list(dict.fromkeys(tokenize(cat["name"] + " " + " ".join(json.loads(cat["synonyms"] or "[]")))))
    hits = [{"term": t, "count": ptoks[t]} for t in terms if ptoks[t]]
    coverage = len(hits) / max(1, len(terms))
    normalized_text = re.sub(r"\s+", " ", paper["text_content"].lower())
    exact = 1.0 if re.sub(r"\s+", " ", cat["name"].lower()).strip() in normalized_text else 0.0
    synonym = next((s for s in json.loads(cat["synonyms"] or "[]") if s.strip() and re.sub(r"\s+", " ", s.lower()).strip() in normalized_text), None)
    classified = db.execute("SELECT 1 FROM paper_categories WHERE paper_id=? AND category_id=? UNION SELECT 1 FROM paper_comparisons WHERE paper_id=? AND category_id=?", (paper_id, category_id, paper_id, category_id)).fetchone() is not None
    chunk_rows = db.execute("SELECT * FROM chunks WHERE paper_id=? ORDER BY chunk_index", (paper_id,)).fetchall()
    for model_id, _, _ in MODELS:
        cv = cache_embedding(db, "category", category_id, model_id, ctext)
        pv = cache_embedding(db, "paper", paper_id, model_id, paper["text_content"])
        semantic = cosine(pv, cv)
        ranked_terms = []
        for term, count in ptoks.most_common(80):
            if len(term) > 2:
                ranked_terms.append({"term": term, "count": count, "similarity": round(cosine(embed(term, model_id), cv), 4)})
        ranked_terms.sort(key=lambda x: (x["similarity"], x["count"]), reverse=True)
        evidence = []
        for ch in chunk_rows:
            score = cosine(embed(ch["text"], model_id), cv)
            evidence.append({"chunk": ch["chunk_index"], "score": round(score, 4), "text": ch["text"][:520]})
        evidence.sort(key=lambda x: x["score"], reverse=True)
        presence_similarity = max(semantic, evidence[0]["score"] if evidence else 0.0)
        concept = 1.0 if classified or exact or synonym or presence_similarity >= 0.80 else 0.0
        reason = "classified stage" if classified else "exact phrase" if exact else f"synonym: {synonym}" if synonym else f"semantic: {presence_similarity:.0%}" if presence_similarity >= 0.80 else "not detected"
        db.execute("INSERT INTO relationships(paper_id,category_id,model_id,semantic_score,coverage,exact_presence,concept_presence,presence_reason,evidence,top_terms,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(paper_id,category_id,model_id) DO UPDATE SET semantic_score=excluded.semantic_score,coverage=excluded.coverage,exact_presence=excluded.exact_presence,concept_presence=excluded.concept_presence,presence_reason=excluded.presence_reason,evidence=excluded.evidence,top_terms=excluded.top_terms,updated_at=excluded.updated_at", (paper_id, category_id, model_id, round(semantic, 6), round(coverage, 6), exact, concept, reason, json.dumps(evidence[:5]), json.dumps(ranked_terms[:12]), now()))


def prepare_paper(db: sqlite3.Connection, paper_id: int) -> None:
    paper = db.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not paper:
        return
    db.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
    for i, text in enumerate(chunks(paper["text_content"])):
        db.execute("INSERT INTO chunks(paper_id,chunk_index,text,term_counts) VALUES(?,?,?,?)", (paper_id, i, text, json.dumps(Counter(tokenize(text)))))
    for cat in db.execute("SELECT id FROM categories").fetchall():
        evaluate(db, paper_id, cat["id"])
    db.execute("UPDATE papers SET status='ready',updated_at=? WHERE id=?", (now(), paper_id))


def seed_demo_presence(db: sqlite3.Connection) -> None:
    """Add a transparent synthetic corpus so presence metrics can be previewed."""
    if db.execute("SELECT value FROM meta WHERE key='demo_presence_version'").fetchone():
        return
    demos = [
        ("Demo — Proposal and feasibility planning", """Research Design begins with a focused problem statement, literature gap and feasible research question. During Project Conception and Feasibility Analysis the team checks participant access, data availability, ethics constraints and expected contribution. Project Chartering and Business Case Development records the sponsor, supervisor, objectives, success criteria and intended research value. Stakeholder Analysis and Requirements Gathering captures expectations from supervisors, participants, funders and knowledge users. Scope Baseline and WBS Design separates literature review, protocol preparation, fieldwork, analysis and writing. Schedule Estimation and Milestone Planning assigns approval, recruitment and submission dates, while Resource Allocation and Budgeting covers researcher time, transcription, software and travel."""),
        ("Demo — Ethics, governance and stakeholder approval", """The Research Design is reviewed with the ethics committee and participant representatives before recruitment. Stakeholder Analysis and Requirements Gathering identifies consent, privacy, accessibility and reporting needs. Risk Management and Procurement Strategy maintains risks for delayed approval, vulnerable participants, data loss, supplier delays and conflicts of interest. Monitoring, Controlling, and Quality Assurance checks consent records, protocol adherence and secure handling every week. When the committee requests a revised sampling method, Change Control and Variance Mitigation records the reason, approval, schedule effect and new protocol version. The governance log gives the project team an auditable decision trail."""),
        ("Demo — Fieldwork and data collection operations", """Data Collection uses purposive sampling, interviews, a pilot survey and structured observation. Research Tools include the survey platform, encrypted recorder, transcription software, fieldwork tracker and secure database. Team Mobilization and Communication Planning assigns interviewers, training, escalation routes and weekly coordination. Project Execution and Iterative Implementation releases the instrument after a pilot and improves unclear questions between collection waves. Schedule Estimation and Milestone Planning tracks recruitment and interview targets. Resource Allocation and Budgeting monitors incentives, travel and transcription hours. Risk Management and Procurement Strategy addresses non-response and equipment failure, while Monitoring, Controlling, and Quality Assurance reviews missing fields, consent and sample balance."""),
        ("Demo — Analysis, validation and quality assurance", """Data Analysis combines descriptive statistics, thematic analysis, coding and comparison across participant groups. Research Tools include Python, qualitative coding software, notebooks and a versioned analysis repository. Validation and Quality applies reliability checks, triangulation, negative-case review, sensitivity analysis and reproducible scripts. Monitoring, Controlling, and Quality Assurance uses a quality checklist for cleaning, code review, traceability and evidence completeness. Risk Management and Procurement Strategy flags biased samples, unstable estimates and interpretation drift. Change Control and Variance Mitigation documents revised coding rules and model assumptions so every analytical change can be reproduced."""),
        ("Demo — Writing, review and controlled revision", """The writing stage converts Data Analysis into claims supported by tables, quotations and limitations. Validation and Quality requires every conclusion to trace back to evidence and every figure to a reproducible source. Schedule Estimation and Milestone Planning sets dates for chapter drafts, supervisor review and corrections. Change Control and Variance Mitigation records requested revisions, affected chapters, responsible authors and approval status. Monitoring, Controlling, and Quality Assurance checks citations, argument consistency, anonymisation and formatting. Dissemination of Research Findings is prepared early through an article outline, executive summary and stakeholder briefing."""),
        ("Demo — Dissemination and research benefits", """Dissemination of Research Findings includes the thesis submission, journal manuscript, research data statement, practitioner briefing and feedback workshop. Stakeholder Analysis and Requirements Gathering determines which findings matter to participants, managers, policy users and future researchers. Project Execution and Iterative Implementation pilots the briefing with a small user group and improves clarity. Monitoring, Controlling, and Quality Assurance verifies that public claims match the validated evidence and that confidential details are removed. Resource Allocation and Budgeting reserves editing, design and event costs. The team records uptake indicators, questions and follow-on actions to assess whether the research creates practical value."""),
    ]
    stamp = now()
    created = 0
    for title, text in demos:
        existing = db.execute("SELECT id FROM papers WHERE title=? AND source_path='synthetic demo presence corpus'", (title,)).fetchone()
        if existing:
            continue
        pid = db.execute("INSERT INTO papers(title,filename,source_path,text_content,status,selected,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (title, None, "synthetic demo presence corpus", text.strip(), "processing", 1, stamp, stamp)).lastrowid
        prepare_paper(db, pid)
        created += 1
    db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('demo_presence_version','1')")
    emit(db, "demo.presence.seeded", {"papers": created, "synthetic": True})
    db.commit()


def seed(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)
    migrate_schema(db)
    if db.execute("SELECT value FROM meta WHERE key='seed_version'").fetchone():
        seed_demo_presence(db)
        refresh_concept_presence_once(db)
        return
    stamp = now()
    pm = db.execute("INSERT INTO domains(name,description,created_at) VALUES(?,?,?)", ("Project Management", "Lifecycle stages imported from the verified multi-model heatmaps.", stamp)).lastrowid
    research = db.execute("INSERT INTO domains(name,description,created_at) VALUES(?,?,?)", ("Research Methodology / Tools", "Methods, evidence collection, analysis and dissemination.", stamp)).lastrowid
    legacy_rows: list[tuple[str, str, str, float, float, str]] = []
    if LEGACY_INDEX.exists():
        raw = LEGACY_INDEX.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(r"<tr><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>([0-9.]+)</td><td>([0-9.]+)</td><td><a href=\"(.*?)\"")
        legacy_rows = [(html.unescape(a), html.unescape(b), html.unescape(c), float(d), float(e), f) for a,b,c,d,e,f in pattern.findall(raw)]
    names: list[str] = []
    for _, ref, comp, *_ in legacy_rows:
        for name in (ref, comp):
            if name not in names:
                names.append(name)
    if not names:
        names = ["Project Conception and Feasibility Analysis", "Project Chartering and Business Case Development", "Stakeholder Analysis and Requirements Gathering", "Scope Baseline and WBS Design", "Schedule Estimation and Milestone Planning", "Resource Allocation and Budgeting", "Risk Management and Procurement Strategy", "Team Mobilization and Communication Planning", "Project Execution and Iterative Implementation", "Monitoring, Controlling, and Quality Assurance", "Change Control and Variance Mitigation", "Dissemination of Research Findings"]
    ids: dict[str, int] = {}
    for pos, name in enumerate(names):
        desc = f"Imported analytical stage: {name}. Edit this description and its synonyms to refine future document evaluation."
        ids[name] = db.execute("INSERT INTO categories(domain_id,name,description,synonyms,position,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", (pm, name, desc, "[]", pos, stamp, stamp)).lastrowid
    model_map = {label: mid for mid, label, _ in MODELS}
    for label, ref, comp, best, pairs, href in legacy_rows:
        mid = model_map.get(label)
        if mid and ref in ids and comp in ids:
            db.execute("INSERT OR IGNORE INTO legacy_scores(model_id,reference_category_id,comparison_category_id,best_match,all_pairs,source_href) VALUES(?,?,?,?,?,?)", (mid, ids[ref], ids[comp], best, pairs, href))
    research_defaults = [
        ("Research Design", "Study design, research questions, hypotheses and methodological alignment.", ["methodology", "study design"]),
        ("Data Collection", "Sampling, instruments, interviews, surveys, observation and source acquisition.", ["sampling", "measurement"]),
        ("Data Analysis", "Qualitative, quantitative and mixed-method analysis.", ["statistics", "coding", "thematic analysis"]),
        ("Validation and Quality", "Validity, reliability, triangulation, bias and reproducibility.", ["reliability", "triangulation"]),
        ("Research Tools", "Software, computational tools, databases and analytical workflows.", ["software", "tooling"]),
    ]
    for pos, (name, desc, synonyms) in enumerate(research_defaults):
        db.execute("INSERT INTO categories(domain_id,name,description,synonyms,position,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", (research, name, desc, json.dumps(synonyms), pos, stamp, stamp))
    # Seed one editable corpus record per imported stage. The underlying export contains phrase matrices, not source PDFs.
    for pos, name in enumerate(names):
        text = f"{name}. Imported from the verified legacy semantic heatmap corpus. " + name.replace(" and ", ". ").replace(",", ".")
        pid = db.execute("INSERT INTO papers(title,filename,source_path,text_content,status,selected,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (f"Legacy corpus — {name}", None, "legacy heatmap export", text, "processing", 1, stamp, stamp)).lastrowid
        prepare_paper(db, pid)
    db.execute("INSERT INTO meta(key,value) VALUES('seed_version','1')")
    emit(db, "seed.completed", {"categories": len(names) + len(research_defaults), "papers": len(names), "legacy_scores": len(legacy_rows)})
    db.commit()
    seed_demo_presence(db)
    refresh_concept_presence_once(db)


def refresh_concept_presence_once(db: sqlite3.Connection) -> None:
    """Backfill the richer presence rule once without changing user records."""
    if db.execute("SELECT value FROM meta WHERE key='concept_presence_version'").fetchone():
        return
    paper_ids = [row["id"] for row in db.execute("SELECT id FROM papers").fetchall()]
    category_ids = [row["id"] for row in db.execute("SELECT id FROM categories").fetchall()]
    for paper_id in paper_ids:
        for category_id in category_ids:
            evaluate(db, paper_id, category_id)
    db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('concept_presence_version','1')")
    emit(db, "concept.presence.migrated", {"papers": len(paper_ids), "categories": len(category_ids), "threshold": 0.80})
    db.commit()


def queue_job(db: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> int:
    jid = db.execute("INSERT INTO jobs(kind,status,payload,created_at,updated_at) VALUES(?,?,?,?,?)", (kind, "queued", json.dumps(payload), now(), now())).lastrowid
    emit(db, "job.queued", {"id": jid, "kind": kind})
    db.commit()
    return jid


def process_one_job() -> bool:
    with closing(connect()) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY id LIMIT 1").fetchone()
        if not row:
            db.rollback(); return False
        db.execute("UPDATE jobs SET status='running',progress=.05,message='Starting',updated_at=? WHERE id=?", (now(), row["id"]))
        db.commit()
        try:
            payload = json.loads(row["payload"])
            if row["kind"] == "paper.process":
                prepare_paper(db, int(payload["paper_id"]))
            elif row["kind"] == "paper.bulk_process":
                paper_ids = [int(x) for x in payload.get("paper_ids", [])]
                for index, paper_id in enumerate(paper_ids, 1):
                    prepare_paper(db, paper_id)
                    db.execute("UPDATE jobs SET progress=?,message=?,updated_at=? WHERE id=?", (index / max(1, len(paper_ids)), f"Analysed {index} of {len(paper_ids)} papers", now(), row["id"]))
                    db.commit()
            elif row["kind"] == "category.recalculate":
                cid = int(payload["category_id"])
                for paper in db.execute("SELECT id FROM papers").fetchall(): evaluate(db, paper["id"], cid)
            db.execute("UPDATE jobs SET status='completed',progress=1,message='Completed',updated_at=? WHERE id=?", (now(), row["id"]))
            emit(db, "job.completed", {"id": row["id"], "kind": row["kind"]})
            db.commit()
        except Exception as exc:
            db.execute("UPDATE jobs SET status='failed',message=?,updated_at=? WHERE id=?", (str(exc)[:500], now(), row["id"]))
            emit(db, "job.failed", {"id": row["id"], "error": str(exc)[:300]})
            db.commit()
        return True


def worker() -> None:
    while not worker_stop.is_set():
        if not process_one_job(): worker_stop.wait(0.5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global worker_thread
    with closing(connect()) as db: seed(db)
    worker_stop.clear()
    worker_thread = threading.Thread(target=worker, daemon=True, name="semantic-job-worker")
    worker_thread.start()
    yield
    worker_stop.set()
    if worker_thread: worker_thread.join(timeout=2)


app = FastAPI(title="Semantic Research Dashboard", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class DomainIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    description: str = Field(default="", max_length=1000)


class CategoryIn(BaseModel):
    domain_id: int
    name: str = Field(min_length=2, max_length=180)
    description: str = Field(default="", max_length=3000)
    synonyms: list[str] = Field(default_factory=list)
    position: int = 0


class PaperIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text_content: str = Field(min_length=1, max_length=5_000_000)
    stage_id: int | None = None
    comparison_stage_id: int | None = None


def rows(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    with closing(connect()) as db: return [dict(x) for x in db.execute(sql, args).fetchall()]


@app.get("/api/health")
def health():
    with closing(connect()) as db:
        return {"status": "ok", "database": str(DB_PATH), "counts": {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("domains","categories","papers","relationships","legacy_scores","jobs")}}


@app.get("/api/models")
def models(): return [{"id": a, "name": b, "description": c} for a,b,c in MODELS]


@app.get("/api/domains")
def domains():
    out = rows("SELECT * FROM domains ORDER BY id")
    cats = rows("SELECT * FROM categories ORDER BY domain_id,position,id")
    for c in cats: c["synonyms"] = json.loads(c["synonyms"])
    for d in out: d["categories"] = [c for c in cats if c["domain_id"] == d["id"]]
    return out


@app.post("/api/domains", status_code=201)
def create_domain(body: DomainIn):
    try:
        with closing(connect()) as db:
            did = db.execute("INSERT INTO domains(name,description,created_at) VALUES(?,?,?)", (body.name.strip(), body.description.strip(), now())).lastrowid
            emit(db, "domain.created", {"id": did}); db.commit()
            return dict(db.execute("SELECT * FROM domains WHERE id=?", (did,)).fetchone())
    except sqlite3.IntegrityError: raise HTTPException(409, "A domain with this name already exists")


@app.put("/api/domains/{domain_id}")
def update_domain(domain_id: int, body: DomainIn):
    with closing(connect()) as db:
        cur = db.execute("UPDATE domains SET name=?,description=? WHERE id=?", (body.name.strip(),body.description.strip(),domain_id))
        if not cur.rowcount: raise HTTPException(404,"Domain not found")
        emit(db,"domain.updated",{"id":domain_id}); db.commit()
        return dict(db.execute("SELECT * FROM domains WHERE id=?",(domain_id,)).fetchone())


@app.delete("/api/domains/{domain_id}", status_code=204)
def delete_domain(domain_id: int):
    with closing(connect()) as db:
        cur=db.execute("DELETE FROM domains WHERE id=?",(domain_id,))
        if not cur.rowcount: raise HTTPException(404,"Domain not found")
        emit(db,"domain.deleted",{"id":domain_id}); db.commit()


@app.post("/api/categories", status_code=201)
def create_category(body: CategoryIn):
    with closing(connect()) as db:
        cid=db.execute("INSERT INTO categories(domain_id,name,description,synonyms,position,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",(body.domain_id,body.name.strip(),body.description.strip(),json.dumps(body.synonyms),body.position,now(),now())).lastrowid
        jid=queue_job(db,"category.recalculate",{"category_id":cid})
        row=dict(db.execute("SELECT * FROM categories WHERE id=?",(cid,)).fetchone()); row["synonyms"]=json.loads(row["synonyms"]); row["job_id"]=jid; return row


@app.put("/api/categories/{category_id}")
def update_category(category_id: int, body: CategoryIn):
    with closing(connect()) as db:
        cur=db.execute("UPDATE categories SET domain_id=?,name=?,description=?,synonyms=?,position=?,updated_at=? WHERE id=?",(body.domain_id,body.name.strip(),body.description.strip(),json.dumps(body.synonyms),body.position,now(),category_id))
        if not cur.rowcount: raise HTTPException(404,"Category not found")
        db.execute("DELETE FROM embeddings WHERE entity_type='category' AND entity_id=?",(category_id,)); db.execute("DELETE FROM relationships WHERE category_id=?",(category_id,))
        jid=queue_job(db,"category.recalculate",{"category_id":category_id}); return {"id":category_id,"job_id":jid}


@app.delete("/api/categories/{category_id}", status_code=204)
def delete_category(category_id: int):
    with closing(connect()) as db:
        cur=db.execute("DELETE FROM categories WHERE id=?",(category_id,))
        if not cur.rowcount: raise HTTPException(404,"Category not found")
        db.execute("DELETE FROM embeddings WHERE entity_type='category' AND entity_id=?",(category_id,)); emit(db,"category.deleted",{"id":category_id}); db.commit()


def extract_file(path: Path) -> str:
    suffix=path.suffix.lower()
    if suffix==".txt": return path.read_text(encoding="utf-8",errors="ignore")
    if suffix==".pdf":
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if suffix==".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    raise ValueError("Unsupported file type")


@app.get("/api/papers")
def papers(): return rows("SELECT p.id,p.title,p.filename,p.source_path,p.status,p.selected,p.created_at,p.updated_at,length(p.text_content) AS characters,pc.category_id stage_id,c.name stage_name,cmp.category_id comparison_stage_id,cc.name comparison_stage_name FROM papers p LEFT JOIN paper_categories pc ON pc.paper_id=p.id LEFT JOIN categories c ON c.id=pc.category_id LEFT JOIN paper_comparisons cmp ON cmp.paper_id=p.id LEFT JOIN categories cc ON cc.id=cmp.category_id ORDER BY p.id DESC")


@app.get("/api/papers/{paper_id}")
def paper_detail(paper_id: int):
    with closing(connect()) as db:
        row = db.execute("SELECT p.id,p.title,p.filename,p.source_path,p.text_content,p.status,p.selected,p.created_at,p.updated_at,pc.category_id stage_id,c.name stage_name,cmp.category_id comparison_stage_id,cc.name comparison_stage_name FROM papers p LEFT JOIN paper_categories pc ON pc.paper_id=p.id LEFT JOIN categories c ON c.id=pc.category_id LEFT JOIN paper_comparisons cmp ON cmp.paper_id=p.id LEFT JOIN categories cc ON cc.id=cmp.category_id WHERE p.id=?", (paper_id,)).fetchone()
        if not row: raise HTTPException(404, "Paper not found")
        return dict(row)


@app.put("/api/papers/{paper_id}")
def update_paper(paper_id: int, body: PaperIn):
    title, text = body.title.strip(), body.text_content.strip()
    if not title or not text: raise HTTPException(422, "Title and extracted text cannot be blank")
    with closing(connect()) as db:
        current = db.execute("SELECT title,text_content,status FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not current: raise HTTPException(404, "Paper not found")
        text_changed = current["text_content"] != text
        old_stage = db.execute("SELECT category_id FROM paper_categories WHERE paper_id=?", (paper_id,)).fetchone()
        old_comparison = db.execute("SELECT category_id FROM paper_comparisons WHERE paper_id=?", (paper_id,)).fetchone()
        stage_supplied = "stage_id" in body.model_fields_set
        comparison_supplied = "comparison_stage_id" in body.model_fields_set
        stage_changed = stage_supplied and (old_stage["category_id"] if old_stage else None) != body.stage_id
        comparison_changed = comparison_supplied and (old_comparison["category_id"] if old_comparison else None) != body.comparison_stage_id
        if body.stage_id is not None and not db.execute("SELECT 1 FROM categories WHERE id=?", (body.stage_id,)).fetchone(): raise HTTPException(422, "Selected primary stage does not exist")
        if body.comparison_stage_id is not None and not db.execute("SELECT 1 FROM categories WHERE id=?", (body.comparison_stage_id,)).fetchone(): raise HTTPException(422, "Selected comparative stage does not exist")
        if stage_supplied:
            db.execute("DELETE FROM paper_categories WHERE paper_id=?", (paper_id,))
            if body.stage_id is not None: db.execute("INSERT INTO paper_categories(paper_id,category_id,created_at) VALUES(?,?,?)", (paper_id, body.stage_id, now()))
        if comparison_supplied:
            db.execute("DELETE FROM paper_comparisons WHERE paper_id=?", (paper_id,))
            if body.comparison_stage_id is not None: db.execute("INSERT INTO paper_comparisons(paper_id,category_id,created_at) VALUES(?,?,?)", (paper_id, body.comparison_stage_id, now()))
        recalculation = text_changed or stage_changed or comparison_changed
        status = "queued" if recalculation else current["status"]
        db.execute("UPDATE papers SET title=?,text_content=?,status=?,updated_at=? WHERE id=?", (title, text, status, now(), paper_id))
        if recalculation:
            db.execute("DELETE FROM embeddings WHERE entity_type='paper' AND entity_id=?", (paper_id,))
            db.execute("DELETE FROM relationships WHERE paper_id=?", (paper_id,))
            job_id = queue_job(db, "paper.process", {"paper_id": paper_id})
        else:
            emit(db, "paper.updated", {"id": paper_id, "recalculation": False}); db.commit(); job_id = None
        return {"id": paper_id, "status": status, "job_id": job_id, "recalculation": recalculation}


@app.post("/api/papers/upload", status_code=202)
async def upload_paper(file: UploadFile=File(...), title: str=Form(""), stage_id: int=Form(...), comparison_stage_id: int | None=Form(None)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED: raise HTTPException(415,"Upload a PDF, DOCX or TXT file")
    blob=await file.read()
    if len(blob)>30*1024*1024: raise HTTPException(413,"File exceeds the 30 MB limit")
    safe=f"{uuid.uuid4().hex}{suffix}"; path=UPLOADS/safe; path.write_bytes(blob)
    try: text=extract_file(path)
    except Exception as exc: path.unlink(missing_ok=True); raise HTTPException(422,f"Could not extract text: {exc}")
    if not text.strip(): path.unlink(missing_ok=True); raise HTTPException(422,"No readable text was extracted")
    with closing(connect()) as db:
        if not db.execute("SELECT 1 FROM categories WHERE id=?", (stage_id,)).fetchone(): path.unlink(missing_ok=True); raise HTTPException(422,"Selected research stage does not exist")
        if comparison_stage_id is not None and not db.execute("SELECT 1 FROM categories WHERE id=?", (comparison_stage_id,)).fetchone(): path.unlink(missing_ok=True); raise HTTPException(422,"Selected comparative stage does not exist")
        stamp=now(); pid=db.execute("INSERT INTO papers(title,filename,source_path,text_content,status,selected,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",(title.strip() or Path(file.filename or "Untitled").stem,file.filename,str(path),text,"queued",1,stamp,stamp)).lastrowid
        db.execute("INSERT INTO paper_categories(paper_id,category_id,created_at) VALUES(?,?,?)", (pid, stage_id, stamp))
        if comparison_stage_id is not None: db.execute("INSERT INTO paper_comparisons(paper_id,category_id,created_at) VALUES(?,?,?)", (pid, comparison_stage_id, stamp))
        jid=queue_job(db,"paper.process",{"paper_id":pid}); return {"paper_id":pid,"job_id":jid,"status":"queued","characters":len(text)}


@app.post("/api/papers/bulk-upload", status_code=202)
async def bulk_upload_papers(files: list[UploadFile]=File(...), stage_id: int=Form(...), comparison_stage_id: int | None=Form(None)):
    if not files: raise HTTPException(422, "Choose at least one paper")
    if len(files) > 100: raise HTTPException(413, "Bulk upload is limited to 100 papers at a time")
    accepted, errors = [], []
    with closing(connect()) as db:
        if not db.execute("SELECT 1 FROM categories WHERE id=?", (stage_id,)).fetchone(): raise HTTPException(422,"Selected primary stage does not exist")
        if comparison_stage_id is not None and not db.execute("SELECT 1 FROM categories WHERE id=?", (comparison_stage_id,)).fetchone(): raise HTTPException(422,"Selected comparative stage does not exist")
        for file in files:
            suffix=Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED: errors.append({"filename":file.filename,"error":"Unsupported file type"}); continue
            blob=await file.read()
            if len(blob)>30*1024*1024: errors.append({"filename":file.filename,"error":"File exceeds 30 MB"}); continue
            path=UPLOADS/f"{uuid.uuid4().hex}{suffix}"; path.write_bytes(blob)
            try: text=extract_file(path)
            except Exception as exc: path.unlink(missing_ok=True); errors.append({"filename":file.filename,"error":f"Extraction failed: {exc}"}); continue
            if not text.strip(): path.unlink(missing_ok=True); errors.append({"filename":file.filename,"error":"No readable text"}); continue
            stamp=now(); pid=db.execute("INSERT INTO papers(title,filename,source_path,text_content,status,selected,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",(Path(file.filename or "Untitled").stem,file.filename,str(path),text,"queued",1,stamp,stamp)).lastrowid
            db.execute("INSERT INTO paper_categories(paper_id,category_id,created_at) VALUES(?,?,?)",(pid,stage_id,stamp))
            if comparison_stage_id is not None: db.execute("INSERT INTO paper_comparisons(paper_id,category_id,created_at) VALUES(?,?,?)",(pid,comparison_stage_id,stamp))
            accepted.append({"paper_id":pid,"filename":file.filename,"title":Path(file.filename or "Untitled").stem,"characters":len(text)})
        if not accepted: db.rollback(); raise HTTPException(422,{"message":"No papers could be imported","files":errors})
        jid=queue_job(db,"paper.bulk_process",{"paper_ids":[item["paper_id"] for item in accepted]})
        return {"job_id":jid,"status":"queued","accepted":accepted,"errors":errors}


@app.patch("/api/papers/{paper_id}/selection")
def select_paper(paper_id:int, selected:bool):
    with closing(connect()) as db:
        cur=db.execute("UPDATE papers SET selected=?,updated_at=? WHERE id=?",(int(selected),now(),paper_id))
        if not cur.rowcount: raise HTTPException(404,"Paper not found")
        emit(db,"paper.selection",{"id":paper_id,"selected":selected}); db.commit(); return {"id":paper_id,"selected":selected}


@app.delete("/api/papers/{paper_id}",status_code=204)
def delete_paper(paper_id:int):
    with closing(connect()) as db:
        row=db.execute("SELECT source_path FROM papers WHERE id=?",(paper_id,)).fetchone()
        if not row: raise HTTPException(404,"Paper not found")
        db.execute("DELETE FROM papers WHERE id=?",(paper_id,)); db.execute("DELETE FROM embeddings WHERE entity_type='paper' AND entity_id=?",(paper_id,)); emit(db,"paper.deleted",{"id":paper_id}); db.commit()
        if row["source_path"] and str(row["source_path"]).startswith(str(UPLOADS)): Path(row["source_path"]).unlink(missing_ok=True)


@app.get("/api/jobs")
def jobs(): return rows("SELECT * FROM jobs ORDER BY id DESC LIMIT 40")


def _selected_ids(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.isdigit()]


def _stage_score(db: sqlite3.Connection, left: sqlite3.Row, right: sqlite3.Row, model_id: str) -> float:
    legacy = db.execute("SELECT best_match FROM legacy_scores WHERE model_id=? AND reference_category_id=? AND comparison_category_id=?", (model_id, left["id"], right["id"])).fetchone()
    if legacy:
        return float(legacy["best_match"])
    return cosine(embed(category_text(left), model_id), embed(category_text(right), model_id))


def _corpus_pair_metrics(db: sqlite3.Connection, left_id: int, right_id: int, model_ids: list[str], paper_ids: list[int]) -> tuple[float, float, float, int]:
    params: list[Any] = [left_id, right_id, *model_ids]
    paper_clause = ""
    if paper_ids:
        paper_clause = f" AND p.id IN ({','.join('?' * len(paper_ids))})"
        params.extend(paper_ids)
    found = db.execute(f"SELECT r.paper_id,r.model_id,r.category_id,r.semantic_score,r.exact_presence,r.concept_presence FROM relationships r JOIN papers p ON p.id=r.paper_id WHERE p.selected=1 AND r.category_id IN (?,?) AND r.model_id IN ({','.join('?' * len(model_ids))}){paper_clause}", params).fetchall()
    paired: dict[tuple[int, str], dict[int, tuple[float, float, float]]] = {}
    for row in found:
        paired.setdefault((row["paper_id"], row["model_id"]), {})[row["category_id"]] = (row["semantic_score"], row["exact_presence"], row["concept_presence"])
    supports, exacts, concepts = [], [], []
    for values in paired.values():
        if left_id in values and right_id in values:
            supports.append(math.sqrt(values[left_id][0] * values[right_id][0]))
            exacts.append(1.0 if values[left_id][1] and values[right_id][1] else 0.0)
            concepts.append(1.0 if values[left_id][2] and values[right_id][2] else 0.0)
    return (sum(supports) / len(supports) if supports else 0.0, sum(exacts) / len(exacts) if exacts else 0.0, sum(concepts) / len(concepts) if concepts else 0.0, len(supports))


def _category_document_presence(db: sqlite3.Connection, category_id: int, model_ids: list[str], paper_ids: list[int]) -> tuple[int, int]:
    params: list[Any] = [category_id, *model_ids]
    paper_clause = ""
    if paper_ids:
        paper_clause = f" AND p.id IN ({','.join('?' * len(paper_ids))})"
        params.extend(paper_ids)
    grouped = db.execute(f"SELECT r.paper_id,MAX(r.concept_presence) present FROM relationships r JOIN papers p ON p.id=r.paper_id WHERE p.selected=1 AND r.category_id=? AND r.model_id IN ({','.join('?' * len(model_ids))}){paper_clause} GROUP BY r.paper_id", params).fetchall()
    return sum(1 for row in grouped if row["present"]), len(grouped)


def _practice_guidance(research_name: str, pm_name: str, terms: list[str]) -> list[str]:
    name = pm_name.lower()
    actions = [
        ("conception", "Use a short feasibility review to test research value, constraints, access and viability before committing resources."),
        ("charter", "Create a research charter defining the question, intended contribution, sponsor, decision rights and success criteria."),
        ("stakeholder", "Map supervisors, participants, ethics bodies, funders and knowledge users; record their requirements and influence."),
        ("scope", "Turn the research aim into a bounded work breakdown structure covering literature, fieldwork, analysis and outputs."),
        ("schedule", "Plan approval gates, recruitment, data collection, analysis and writing as milestones with explicit dependencies."),
        ("resource", "Assign time, budget, software, data access and specialist support to each research work package."),
        ("risk", "Maintain a research risk register for ethics, recruitment, data quality, access, bias, privacy and schedule threats."),
        ("team", "Define collaboration, supervision, authorship and communication routines before research execution begins."),
        ("execution", "Run the study in short review cycles so evidence, assumptions and methods can be adjusted transparently."),
        ("monitor", "Use quality gates for protocol adherence, evidence completeness, validity, reproducibility and deliverable acceptance."),
        ("change", "Record method and scope changes with their rationale, evidence impact, approvals and version history."),
        ("dissemination", "Treat publication, stakeholder feedback, data sharing and implementation as planned project deliverables."),
    ]
    advice = next((text for key, text in actions if key in name), f"Adapt the controls and deliverables of {pm_name} to the objectives and evidence needs of {research_name}.")
    result = [advice]
    if terms:
        result.append(f"Use the corpus signals {', '.join(terms[:4])} as checkpoints when tailoring this practice.")
    result.append(f"Define one measurable outcome showing how {pm_name} improves the reliability, pace or usefulness of {research_name}.")
    return result


@app.get("/api/stage-alignment")
def stage_alignment(x_domain_id: int, y_domain_id: int, paper_ids: str = "", x_category_ids: str = "", y_category_ids: str = "", model_ids: str = "", metric: str = Query("alignment", pattern="^(alignment|corpus_support|concept_presence|exact_presence)$")):
    mids = [x for x in model_ids.split(",") if x] or [x[0] for x in MODELS]
    pids, xids, yids = _selected_ids(paper_ids), _selected_ids(x_category_ids), _selected_ids(y_category_ids)
    with closing(connect()) as db:
        xrows = db.execute("SELECT * FROM categories WHERE domain_id=? ORDER BY position,id", (x_domain_id,)).fetchall()
        yrows = db.execute("SELECT * FROM categories WHERE domain_id=? ORDER BY position,id", (y_domain_id,)).fetchall()
        if xids: xrows = [x for x in xrows if x["id"] in xids]
        if yids: yrows = [y for y in yrows if y["id"] in yids]
        presence = {category_id: _category_document_presence(db, category_id, mids, pids) for category_id in {row["id"] for row in [*xrows, *yrows]}}
        cells = []
        for y in yrows:
            for x in xrows:
                alignment = sum(_stage_score(db, y, x, mid) for mid in mids) / len(mids)
                support, exact, concept, observations = _corpus_pair_metrics(db, y["id"], x["id"], mids, pids)
                values = {"alignment": alignment, "corpus_support": support, "concept_presence": concept, "exact_presence": exact}
                x_present, x_total = presence[x["id"]]; y_present, y_total = presence[y["id"]]
                cells.append({"x_category_id": x["id"], "y_category_id": y["id"], "x_name": x["name"], "y_name": y["name"], "alignment": round(alignment, 6), "corpus_support": round(support, 6), "concept_presence": round(concept, 6), "exact_presence": round(exact, 6), "x_present_documents": x_present, "x_total_documents": x_total, "y_present_documents": y_present, "y_total_documents": y_total, "observations": observations, "value": round(values[metric], 6)})
        xd = db.execute("SELECT name FROM domains WHERE id=?", (x_domain_id,)).fetchone()
        yd = db.execute("SELECT name FROM domains WHERE id=?", (y_domain_id,)).fetchone()
        if not xd or not yd: raise HTTPException(404, "Domain not found")
        return {"mode": "self" if x_domain_id == y_domain_id else "bi", "metric": metric, "models": mids, "x_domain": xd["name"], "y_domain": yd["name"], "columns": [[x["id"], x["name"]] for x in xrows], "rows": [[y["id"], y["name"]] for y in yrows], "cells": cells}


@app.get("/api/stage-alignment/detail")
def stage_alignment_detail(x_category_id: int, y_category_id: int, paper_ids: str = "", model_ids: str = ""):
    mids = [x for x in model_ids.split(",") if x] or [x[0] for x in MODELS]
    pids = _selected_ids(paper_ids)
    with closing(connect()) as db:
        x = db.execute("SELECT c.*,d.name domain_name FROM categories c JOIN domains d ON d.id=c.domain_id WHERE c.id=?", (x_category_id,)).fetchone()
        y = db.execute("SELECT c.*,d.name domain_name FROM categories c JOIN domains d ON d.id=c.domain_id WHERE c.id=?", (y_category_id,)).fetchone()
        if not x or not y: raise HTTPException(404, "Category not found")
        alignment = sum(_stage_score(db, y, x, mid) for mid in mids) / len(mids)
        support, exact, concept, observations = _corpus_pair_metrics(db, y["id"], x["id"], mids, pids)
        params: list[Any] = []
        paper_clause = "p.selected=1"
        if pids:
            paper_clause += f" AND p.id IN ({','.join('?' * len(pids))})"; params.extend(pids)
        paper_rows = db.execute(f"SELECT p.id,p.title,p.text_content FROM papers p WHERE {paper_clause}", params).fetchall()
        counts = Counter(token for paper in paper_rows for token in tokenize(paper["text_content"]))
        xv = {mid: embed(category_text(x), mid) for mid in mids}; yv = {mid: embed(category_text(y), mid) for mid in mids}
        ranked = []
        for term, count in counts.most_common(150):
            score = sum((cosine(embed(term, mid), xv[mid]) + cosine(embed(term, mid), yv[mid])) / 2 for mid in mids) / len(mids)
            ranked.append({"term": term, "count": count, "similarity": round(score, 4)})
        ranked.sort(key=lambda item: (item["similarity"], item["count"]), reverse=True)
        passages = []
        chunk_params: list[Any] = []
        chunk_clause = "p.selected=1"
        if pids:
            chunk_clause += f" AND p.id IN ({','.join('?' * len(pids))})"; chunk_params.extend(pids)
        for row in db.execute(f"SELECT ch.chunk_index,ch.text,p.id paper_id,p.title FROM chunks ch JOIN papers p ON p.id=ch.paper_id WHERE {chunk_clause}", chunk_params).fetchall():
            score = sum((cosine(embed(row["text"], mid), xv[mid]) + cosine(embed(row["text"], mid), yv[mid])) / 2 for mid in mids) / len(mids)
            passages.append({"paper_id": row["paper_id"], "paper_title": row["title"], "chunk": row["chunk_index"], "score": round(score, 4), "text": row["text"][:620]})
        passages.sort(key=lambda item: item["score"], reverse=True)
        terms = [item["term"] for item in ranked[:8]]
        x_is_pm = "project management" in x["domain_name"].lower(); y_is_pm = "project management" in y["domain_name"].lower()
        if x_is_pm != y_is_pm:
            pm, research = (x, y) if x_is_pm else (y, x)
            recommendations = _practice_guidance(research["name"], pm["name"], terms)
        else:
            recommendations = [f"Use the overlap between {y['name']} and {x['name']} to define an explicit hand-off, shared output or decision gate.", "Review highly similar stages for duplicated work; review weakly aligned stages for missing dependencies.", "Assign an owner and measurable completion criterion to each stage so semantic overlap does not become responsibility overlap."]
        return {"x_category_id": x["id"], "y_category_id": y["id"], "x_name": x["name"], "y_name": y["name"], "x_domain": x["domain_name"], "y_domain": y["domain_name"], "alignment": round(alignment, 6), "corpus_support": round(support, 6), "concept_presence": round(concept, 6), "exact_presence": round(exact, 6), "observations": observations, "top_terms": ranked[:12], "evidence": passages[:6], "recommendations": recommendations}


@app.get("/api/heatmap")
def heatmap(domain_id:int|None=None,paper_ids:str="",category_ids:str="",model_ids:str="",metric:str=Query("semantic_score",pattern="^(semantic_score|coverage|concept_presence|exact_presence)$")):
    pids=[int(x) for x in paper_ids.split(",") if x.isdigit()]
    cids=[int(x) for x in category_ids.split(",") if x.isdigit()]
    mids=[x for x in model_ids.split(",") if x] or [x[0] for x in MODELS]
    with closing(connect()) as db:
        params:list[Any]=[]; where=["p.selected=1"]
        if domain_id: where.append("c.domain_id=?"); params.append(domain_id)
        if pids: where.append(f"p.id IN ({','.join('?'*len(pids))})"); params+=pids
        if cids: where.append(f"c.id IN ({','.join('?'*len(cids))})"); params+=cids
        where.append(f"r.model_id IN ({','.join('?'*len(mids))})"); params+=mids
        sql=f"SELECT p.id paper_id,p.title paper_title,c.id category_id,c.name category_name,AVG(r.{metric}) value,COUNT(*) model_count FROM relationships r JOIN papers p ON p.id=r.paper_id JOIN categories c ON c.id=r.category_id WHERE {' AND '.join(where)} GROUP BY p.id,c.id ORDER BY c.position,c.id,p.id"
        data=[dict(x) for x in db.execute(sql,params).fetchall()]
        return {"metric":metric,"models":mids,"cells":data,"papers":list({x["paper_id"]:x["paper_title"] for x in data}.items()),"categories":list({x["category_id"]:x["category_name"] for x in data}.items())}


@app.get("/api/evidence")
def evidence(paper_id:int,category_id:int,model_id:str="mpnet-primary"):
    with closing(connect()) as db:
        r=db.execute("SELECT r.*,p.title paper_title,c.name category_name FROM relationships r JOIN papers p ON p.id=r.paper_id JOIN categories c ON c.id=r.category_id WHERE paper_id=? AND category_id=? AND model_id=?",(paper_id,category_id,model_id)).fetchone()
        if not r: raise HTTPException(404,"No evaluated relationship")
        aggregate=db.execute("SELECT COUNT(*) total,SUM(r.concept_presence) present FROM relationships r JOIN papers p ON p.id=r.paper_id WHERE p.selected=1 AND r.category_id=? AND r.model_id=?",(category_id,model_id)).fetchone()
        out=dict(r); out["evidence"]=json.loads(out["evidence"]); out["top_terms"]=json.loads(out["top_terms"]); out["present_documents"]=int(aggregate["present"] or 0); out["total_documents"]=int(aggregate["total"] or 0); out["corpus_presence"]=round(out["present_documents"] / max(1,out["total_documents"]),6); return out


@app.get("/api/top-phrases")
def phrases(paper_ids:str="",limit:int=Query(20,ge=1,le=100)):
    ids=[int(x) for x in paper_ids.split(",") if x.isdigit()]
    with closing(connect()) as db:
        sql="SELECT text_content FROM papers WHERE selected=1"; args:list[Any]=[]
        if ids: sql+=f" AND id IN ({','.join('?'*len(ids))})"; args=ids
        text=" ".join(x[0] for x in db.execute(sql,args).fetchall()); return top_phrases(text,limit)


@app.get("/api/analytics/summary")
def analytics_summary():
    with closing(connect()) as db:
        paper_rows=db.execute("SELECT id,title,text_content,status FROM papers WHERE selected=1 ORDER BY title,id").fetchall()
        tokens=[]; document_words=[]
        for paper in paper_rows:
            paper_tokens=tokenize(paper["text_content"]); tokens.extend(paper_tokens); paper_counts=Counter(paper_tokens)
            document_words.append({"id":paper["id"],"title":paper["title"],"total_words":len(paper_tokens),"unique_words":len(paper_counts),"top_words":[{"word":word,"count":count} for word,count in paper_counts.most_common(100)]})
        counts=Counter(tokens)
        model_rows=db.execute("SELECT r.model_id,AVG(r.semantic_score) semantic_score,AVG(r.coverage) coverage,AVG(r.concept_presence) concept_presence,AVG(r.exact_presence) exact_presence,COUNT(*) relationships FROM relationships r JOIN papers p ON p.id=r.paper_id WHERE p.selected=1 GROUP BY r.model_id ORDER BY r.model_id").fetchall()
        model_names={model_id:name for model_id,name,_ in MODELS}
        model_metrics=[{**dict(row),"name":model_names.get(row["model_id"],row["model_id"])} for row in model_rows]
        stages=[]
        for category in db.execute("SELECT c.id,c.name,d.name domain FROM categories c JOIN domains d ON d.id=c.domain_id ORDER BY d.id,c.position,c.id").fetchall():
            grouped=db.execute("SELECT r.paper_id,MAX(r.concept_presence) present,AVG(r.semantic_score) semantic_score FROM relationships r JOIN papers p ON p.id=r.paper_id WHERE p.selected=1 AND r.category_id=? GROUP BY r.paper_id",(category["id"],)).fetchall()
            present=sum(1 for row in grouped if row["present"])
            stages.append({"id":category["id"],"name":category["name"],"domain":category["domain"],"present_documents":present,"total_documents":len(grouped),"presence_rate":round(present/max(1,len(grouped)),6),"semantic_score":round(sum(row["semantic_score"] for row in grouped)/max(1,len(grouped)),6)})
        job_counts={row["status"]:row["count"] for row in db.execute("SELECT status,COUNT(*) count FROM jobs GROUP BY status").fetchall()}
        return {"documents":len(paper_rows),"ready_documents":sum(1 for row in paper_rows if row["status"]=="ready"),"total_words":len(tokens),"unique_words":len(counts),"top_words":[{"word":word,"count":count} for word,count in counts.most_common(100)],"document_words":document_words,"model_metrics":model_metrics,"stage_metrics":stages,"jobs":job_counts}


@app.get("/api/legacy-scores")
def legacy_scores(model_id:str="mpnet-primary",metric:str=Query("best_match",pattern="^(best_match|all_pairs)$")):
    return rows(f"SELECT l.model_id,l.reference_category_id,l.comparison_category_id,a.name reference,b.name comparison,l.{metric} value,l.source_href FROM legacy_scores l JOIN categories a ON a.id=l.reference_category_id JOIN categories b ON b.id=l.comparison_category_id WHERE l.model_id=? ORDER BY a.position,b.position",(model_id,))


@app.get("/api/events")
async def events(after:int=0):
    async def stream():
        cursor=after
        while True:
            with closing(connect()) as db: found=db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT 50",(cursor,)).fetchall()
            for row in found:
                cursor=row["id"]; yield f"id: {cursor}\nevent: {row['event_type']}\ndata: {row['payload']}\n\n"
            yield ": keepalive\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


if FRONTEND.exists():
    app.mount("/assets",StaticFiles(directory=FRONTEND/"assets"),name="assets")
    @app.get("/{path:path}")
    def spa(path:str):
        target=FRONTEND/path
        return FileResponse(target if target.is_file() else FRONTEND/"index.html")
else:
    @app.get("/")
    def missing_frontend(): return JSONResponse({"message":"Frontend not built. Run npm install && npm run build in frontend."})
