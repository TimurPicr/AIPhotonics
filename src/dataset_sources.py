"""Bounded, retryable public downloads and selective ZIP reads (no full video archive)."""
from __future__ import annotations

import io
import http.client
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib


def download(url: str, *, limit: int = 32 * 1024 * 1024,
             headers: dict | None = None, attempts: int = 4) -> tuple[bytes, dict]:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "photonics-dataset/1.0",
                                                          **(headers or {})})
            with urllib.request.urlopen(request, timeout=60) as response:
                length = int(response.headers.get("Content-Length", "0"))
                if length > limit:
                    raise ValueError(f"Response exceeds {limit} bytes: {url.split('?')[0]}")
                payload = response.read(limit + 1)
                if len(payload) > limit:
                    raise ValueError("Download exceeds its configured byte limit")
                if response.headers.get("Content-Length") is not None and len(payload) != length:
                    raise OSError(f"Incomplete HTTP body: received {len(payload)} of {length} bytes")
                return payload, {"status": response.status, "url": response.url,
                                 **dict(response.headers.items())}
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"Download failed: {url.split('?')[0]}: {error}") from error
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


class RangeReader(io.RawIOBase):
    """Seekable HTTP object; a server ignoring Range is rejected before a large read."""
    def __init__(self, url: str):
        first, info = download(url, limit=1, headers={"Range": "bytes=0-0"})
        lower = {key.lower(): value for key, value in info.items()}
        if info["status"] != 206 or len(first) != 1 or "content-range" not in lower:
            raise ValueError("Source must support HTTP byte ranges")
        self.url = info["url"]
        self.size = int(lower["content-range"].rsplit("/", 1)[1])
        self.etag = lower.get("etag")
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def fetch(self, start: int, end: int) -> bytes:
        end = min(end, self.size)
        if end <= start:
            return b""
        if end - start > 8 * 1024 * 1024:
            chunk_size = 128 * 1024
            spans = [(offset, min(offset + chunk_size, end)) for offset in range(start, end, chunk_size)]
            print(f"Reading ZIP index: {(end - start) / 1024**2:.1f} MiB in {len(spans)} ranges", flush=True)
            chunks = [None] * len(spans)
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures = {pool.submit(self.fetch, *span): index for index, span in enumerate(spans)}
                for completed, future in enumerate(as_completed(futures), 1):
                    chunks[futures[future]] = future.result()
                    if completed % 64 == 0 or completed == len(spans):
                        print(f"ZIP index ranges {completed}/{len(spans)}", flush=True)
            return b"".join(chunks)
        payload, info = download(self.url, limit=end - start,
                                 headers={"Range": f"bytes={start}-{end - 1}"})
        lower = {key.lower(): value for key, value in info.items()}
        expected = f"bytes {start}-{end - 1}/{self.size}"
        if info["status"] != 206 or lower.get("content-range") != expected or len(payload) != end - start:
            raise ValueError("Server returned an incorrect byte range")
        if self.etag and lower.get("etag") != self.etag:
            raise ValueError("Remote archive changed during download")
        return payload

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        result = self.fetch(self.position, end)
        self.position = end
        return result


class RemoteZip:
    def __init__(self, url: str, cache: Path):
        self.reader = RangeReader(url)
        self.url = url
        saved = None
        if cache.exists():
            try:
                saved = json.loads(cache.read_text())
            except json.JSONDecodeError:
                print("Cached ZIP index is incomplete; rebuilding", flush=True)
        if saved is not None:
            if (saved["url"], saved["size"], saved["etag"]) != (url, self.reader.size, self.reader.etag):
                raise ValueError(f"Archive differs from cached index: {cache}")
            self.entries = saved["entries"]
        else:
            with zipfile.ZipFile(self.reader) as archive:
                self.entries = {item.filename: {"offset": item.header_offset,
                                "compressed_size": item.compress_size, "size": item.file_size,
                                "compression": item.compress_type, "crc": item.CRC}
                                for item in archive.infolist() if not item.is_dir()}
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, suffix=".tmp", delete=False) as handle:
                json.dump({"url": url, "size": self.reader.size,
                           "etag": self.reader.etag, "entries": self.entries}, handle)
                temporary = Path(handle.name)
            temporary.replace(cache)

    def read(self, name: str) -> bytes:
        entry = self.entries[name]
        # Extra fields in a local ZIP header have at most 65,535 bytes.
        start = entry["offset"]
        end = start + 30 + len(name.encode("utf-8")) + 65535 + entry["compressed_size"]
        payload = self.reader.fetch(start, end)
        return self._decode(name, payload)

    def read_many(self, names: list[str]) -> dict[str, bytes]:
        """Read adjacent video frames with one request, rather than four handshakes."""
        start = min(self.entries[name]["offset"] for name in names)
        end = max(self.entries[name]["offset"] + 30 + len(name.encode("utf-8")) +
                  65535 + self.entries[name]["compressed_size"] for name in names)
        if end - start > 16 * 1024 * 1024:
            return {name: self.read(name) for name in names}
        payload = self.reader.fetch(start, end)
        return {name: self._decode(name, payload[self.entries[name]["offset"] - start:]) for name in names}

    def _decode(self, name: str, payload: bytes) -> bytes:
        entry = self.entries[name]
        header = struct.unpack_from("<4s5H3I2H", payload)
        if header[0] != b"PK\x03\x04" or header[2] & 1:
            raise ValueError("Invalid or encrypted ZIP member")
        offset = 30 + header[-2] + header[-1]
        compressed = payload[offset:offset + entry["compressed_size"]]
        if entry["compression"] == zipfile.ZIP_DEFLATED:
            result = zlib.decompress(compressed, -15)
        elif entry["compression"] == zipfile.ZIP_STORED:
            result = compressed
        else:
            raise ValueError("Unsupported ZIP compression")
        if len(result) != entry["size"] or zlib.crc32(result) != entry["crc"]:
            raise ValueError(f"ZIP size/CRC mismatch: {name}")
        return result
