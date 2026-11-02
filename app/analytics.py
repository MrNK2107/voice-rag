"""Question log behind the Insights view.

Each answered, abstained or blocked question is one row in a small SQLite
table. ``summary()`` turns it into what a document owner needs: volume,
abstention rate, languages, top questions and the **gap report** (questions
the documents could not answer, most frequent first).

Privacy: question text is stored so gaps can be read. Set
``ANALYTICS_ENABLED=false`` to keep no log at all.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    collection TEXT,
    question TEXT,
    norm TEXT,
    language TEXT,
    status TEXT,          -- answered | abstained | blocked
    reason TEXT,
    channel TEXT,         -- text | voice | widget
    latency_ms REAL,
    emergency INTEGER DEFAULT 0
)
"""


def normalise_question(q: str) -> str:
    q = re.sub(r"[\s　]+", " ", (q or "").strip().casefold())
    return re.sub(r"[?？!.。؟¿¡\s]+$", "", q)


def response_status(resp) -> str:
    if not resp.abstained:
        return "answered"
    reason = (resp.abstain_reason or "").lower()
    if "blocked" in reason or "unsafe" in reason:
        return "blocked"
    return "abstained"


class QuestionLog:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = str(path) if path else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def record(self, question: str, status: str, collection: Optional[str] = None, language: Optional[str] = None,
               reason: Optional[str] = None, channel: str = "text", latency_ms: Optional[float] = None,
               emergency: bool = False, ts: Optional[float] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO questions (ts, collection, question, norm, language, status, reason, channel, latency_ms, emergency)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts or time.time(), collection, (question or "")[:500], normalise_question(question)[:500],
                 language, status, (reason or "")[:300], channel, latency_ms, int(bool(emergency))),
            )
            self._conn.commit()

    def record_response(self, resp, collection: Optional[str], channel: str = "text") -> None:
        if not (resp.transcript or "").strip():
            return
        self.record(resp.transcript, response_status(resp), collection=collection, language=resp.language,
                    reason=resp.abstain_reason, channel=channel,
                    latency_ms=resp.timings_ms.get("total_ms"), emergency=bool(resp.advisory))

    def summary(self, collection: Optional[str] = None, days: int = 30, top: int = 10) -> Dict:
        since = time.time() - days * 86400
        where, args = "ts >= ?", [since]
        if collection:
            where += " AND collection = ?"
            args.append(collection)
        q = self._conn.execute
        with self._lock:
            counts = dict(q(f"SELECT status, COUNT(*) FROM questions WHERE {where} GROUP BY status", args).fetchall())
            langs = q(f"SELECT COALESCE(language,'und'), COUNT(*) c FROM questions WHERE {where} "
                      f"GROUP BY 1 ORDER BY c DESC", args).fetchall()
            channels = dict(q(f"SELECT channel, COUNT(*) FROM questions WHERE {where} GROUP BY channel", args).fetchall())
            top_q = q(f"SELECT norm, MIN(question), COUNT(*) c, SUM(status='answered') FROM questions "
                      f"WHERE {where} AND status != 'blocked' GROUP BY norm ORDER BY c DESC, MAX(ts) DESC LIMIT ?",
                      args + [top]).fetchall()
            gaps = q(f"SELECT norm, MIN(question), COUNT(*) c, MAX(ts), MIN(language), collection FROM questions "
                     f"WHERE {where} AND status = 'abstained' GROUP BY norm, collection "
                     f"ORDER BY c DESC, MAX(ts) DESC LIMIT ?",
                     args + [top]).fetchall()
            daily = q(f"SELECT date(ts, 'unixepoch', 'localtime') d, COUNT(*), SUM(status='abstained') "
                      f"FROM questions WHERE {where} GROUP BY d ORDER BY d", args).fetchall()
            emerg = q(f"SELECT COUNT(*) FROM questions WHERE {where} AND emergency = 1", args).fetchone()[0]
            lat = [r[0] for r in q(f"SELECT latency_ms FROM questions WHERE {where} AND latency_ms IS NOT NULL "
                                   f"ORDER BY latency_ms", args).fetchall()]
        total = sum(counts.values())
        answered = counts.get("answered", 0)
        abstained = counts.get("abstained", 0)
        return {
            "collection": collection,
            "days": days,
            "total": total,
            "answered": answered,
            "abstained": abstained,
            "blocked": counts.get("blocked", 0),
            "emergencies": emerg,
            # Share of non-blocked questions the documents could not answer.
            "abstention_rate": round(abstained / (answered + abstained), 3) if (answered + abstained) else 0.0,
            "languages": [{"code": c, "count": n} for c, n in langs],
            "channels": channels,
            "top_questions": [{"question": ex, "count": c, "answered": int(a or 0)} for _, ex, c, a in top_q],
            "gaps": [{"question": ex, "count": c, "last_asked": ts, "language": lang, "collection": col}
                     for _, ex, c, ts, lang, col in gaps],
            "daily": [{"date": d, "total": n, "abstained": int(a or 0)} for d, n, a in daily],
            "latency_p50_ms": round(lat[len(lat) // 2], 2) if lat else None,
        }
