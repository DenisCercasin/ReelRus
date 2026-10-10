"""Per-user, per-chat, per-video Q&A session management."""

import threading


class ConversationManager:
    def __init__(self, store, history_limit=8):
        self.store = store
        self.history_limit = history_limit
        self._locks = {}
        self._locks_lock = threading.Lock()

    def lock_for(self, user_id, chat_id, video_id):
        key = (user_id, chat_id, video_id)
        with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = threading.RLock()
            return self._locks[key]

    def history(self, user_id, chat_id, video_id):
        return self.store.history(user_id, chat_id, video_id, self.history_limit)

    def record(self, user_id, chat_id, video_id, question, answer):
        self.store.add_exchange(user_id, chat_id, video_id, question, answer,
                                self.history_limit)
