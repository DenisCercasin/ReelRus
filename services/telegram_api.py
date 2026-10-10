"""Small Telegram Bot API transport using only Python's standard library."""

import json
import mimetypes
import uuid
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class TelegramAPI:
    def __init__(self, token, timeout=90):
        self.root = "https://api.telegram.org/bot" + token + "/"
        self.file_root = "https://api.telegram.org/file/bot" + token + "/"
        self.timeout = timeout

    def _call(self, method, payload=None, file_path=None):
        if file_path is None:
            body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
            content_type = "application/json"
        else:
            boundary = "reelrus-" + uuid.uuid4().hex
            pieces = []
            for key, value in (payload or {}).items():
                pieces.append(("--" + boundary + "\r\nContent-Disposition: form-data; "
                               "name=\"" + key + "\"\r\n\r\n" + str(value) + "\r\n").encode())
            path = Path(file_path)
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            pieces.append(("--" + boundary + "\r\nContent-Disposition: form-data; "
                           "name=\"video\"; filename=\"" + path.name + "\"\r\n"
                           "Content-Type: " + mime + "\r\n\r\n").encode())
            pieces.append(path.read_bytes())
            pieces.append(("\r\n--" + boundary + "--\r\n").encode())
            body = b"".join(pieces)
            content_type = "multipart/form-data; boundary=" + boundary
        request = Request(self.root + method, data=body,
                          headers={"Content-Type": content_type}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not result.get("ok"):
            raise RuntimeError("Telegram API error: " + str(result))
        return result["result"]

    def send_message(self, chat_id, text, reply_markup=None):
        # Telegram limits text messages to 4096 characters.
        result = None
        while text:
            chunk, text = text[:4000], text[4000:]
            payload = {"chat_id": chat_id, "text": chunk}
            if reply_markup and not text:
                payload["reply_markup"] = reply_markup
            result = self._call("sendMessage", payload)
        return result

    def send_video(self, chat_id, path, caption=""):
        return self._call("sendVideo", {"chat_id": chat_id, "caption": caption}, path)

    def answer_callback(self, callback_id, text=None):
        return self._call("answerCallbackQuery", {"callback_query_id": callback_id,
                                                   "text": text or ""})

    def get_file_url(self, file_id):
        result = self._call("getFile", {"file_id": file_id})
        return self.file_root + result["file_path"]

    def get_updates(self, offset=None, timeout=30):
        payload = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            payload["offset"] = offset
        return self._call("getUpdates", payload)
