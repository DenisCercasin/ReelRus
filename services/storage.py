"""SQLite persistence for videos, active chats, and Q&A history."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class VideoStore:
    def __init__(self, path, ttl_seconds=7 * 86400, clock=time.time):
        self.path = str(path)
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
                    original_url TEXT, source_transcript TEXT, segments_json TEXT NOT NULL DEFAULT '[]',
                    translation TEXT, summary TEXT, metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS videos_owner ON videos(user_id, chat_id, created_at);
                CREATE TABLE IF NOT EXISTS chat_state (
                    user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
                    active_video_id TEXT, question_video_id TEXT,
                    PRIMARY KEY (user_id, chat_id)
                );
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL, video_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL, created_at REAL NOT NULL,
                    FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS conversation_context
                    ON conversation(user_id, chat_id, video_id, id);
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def create_video(self, user_id, chat_id, original_url=None, metadata=None):
        video_id = uuid.uuid4().hex
        now = self.clock()
        with self._db() as db:
            db.execute("""INSERT INTO videos
                (id,user_id,chat_id,original_url,metadata_json,status,created_at,expires_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (video_id, user_id, chat_id, original_url, json.dumps(metadata or {}),
                 "processing", now, now + self.ttl_seconds))
        return video_id

    def update_video(self, video_id, **fields):
        allowed = {"source_transcript", "segments_json", "translation", "summary",
                   "metadata_json", "status"}
        if not fields or set(fields) - allowed:
            raise ValueError("Invalid video update")
        values = [json.dumps(value, ensure_ascii=False) if key.endswith("_json") else value
                  for key, value in fields.items()]
        with self._db() as db:
            db.execute("UPDATE videos SET " + ", ".join(key + "=?" for key in fields)
                       + " WHERE id=?", values + [video_id])

    @staticmethod
    def _video(row):
        if row is None:
            return None
        video = dict(row)
        video["segments"] = json.loads(video.pop("segments_json"))
        video["metadata"] = json.loads(video.pop("metadata_json"))
        return video

    def get_video(self, video_id, user_id, chat_id):
        with self._db() as db:
            row = db.execute("""SELECT * FROM videos WHERE id=? AND user_id=? AND chat_id=?
                AND expires_at>?""", (video_id, user_id, chat_id, self.clock())).fetchone()
        return self._video(row)

    def list_videos(self, user_id, chat_id, limit=10):
        with self._db() as db:
            rows = db.execute("""SELECT * FROM videos WHERE user_id=? AND chat_id=?
                AND expires_at>? AND status='ready' ORDER BY created_at DESC LIMIT ?""",
                (user_id, chat_id, self.clock(), limit)).fetchall()
        return [self._video(row) for row in rows]

    def select_video(self, user_id, chat_id, video_id, questions=False):
        video = self.get_video(video_id, user_id, chat_id)
        if video is None or video["status"] != "ready":
            return None
        with self._db() as db:
            db.execute("""INSERT INTO chat_state(user_id,chat_id,active_video_id,question_video_id)
                VALUES (?,?,?,?) ON CONFLICT(user_id,chat_id) DO UPDATE SET
                active_video_id=excluded.active_video_id,
                question_video_id=excluded.question_video_id""",
                (user_id, chat_id, video_id, video_id if questions else None))
        return video

    def get_state(self, user_id, chat_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM chat_state WHERE user_id=? AND chat_id=?",
                             (user_id, chat_id)).fetchone()
        state = dict(row) if row else {"active_video_id": None, "question_video_id": None}
        for key in ("active_video_id", "question_video_id"):
            if state[key] and not self.get_video(state[key], user_id, chat_id):
                state[key] = None
        return state

    def clear_question_state(self, user_id, chat_id):
        with self._db() as db:
            db.execute("UPDATE chat_state SET question_video_id=NULL WHERE user_id=? AND chat_id=?",
                       (user_id, chat_id))

    def history(self, user_id, chat_id, video_id, limit=8):
        with self._db() as db:
            rows = db.execute("""SELECT role,content FROM conversation WHERE user_id=? AND chat_id=?
                AND video_id=? ORDER BY id DESC LIMIT ?""",
                (user_id, chat_id, video_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def add_exchange(self, user_id, chat_id, video_id, question, answer, history_limit=8):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            now = self.clock()
            db.executemany("""INSERT INTO conversation
                (user_id,chat_id,video_id,role,content,created_at) VALUES (?,?,?,?,?,?)""",
                [(user_id, chat_id, video_id, "user", question, now),
                 (user_id, chat_id, video_id, "assistant", answer, now)])
            db.execute("""DELETE FROM conversation WHERE user_id=? AND chat_id=? AND video_id=?
                AND id NOT IN (SELECT id FROM conversation WHERE user_id=? AND chat_id=?
                AND video_id=? ORDER BY id DESC LIMIT ?)""",
                (user_id, chat_id, video_id, user_id, chat_id, video_id, history_limit))

    def prune_expired(self):
        with self._db() as db:
            expired = [row[0] for row in db.execute(
                "SELECT id FROM videos WHERE expires_at<=?", (self.clock(),)).fetchall()]
            db.execute("DELETE FROM videos WHERE expires_at<=?", (self.clock(),))
            db.execute("""UPDATE chat_state SET active_video_id=NULL
                WHERE active_video_id NOT IN (SELECT id FROM videos)""")
            db.execute("""UPDATE chat_state SET question_video_id=NULL
                WHERE question_video_id NOT IN (SELECT id FROM videos)""")
        return expired
