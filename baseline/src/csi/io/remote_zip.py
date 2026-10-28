"""Extract selected members of a large remote zip via HTTP Range requests (no full download).

Used for OPERAnet (figshare: wificsi1.zip 34.8 GB, wificsi2.zip 32.8 GB), where only some
experiments are needed. The connection to the storage is unstable, so the compressed bytes of a
member are fetched in 32 MB chunks, each retried and kept on disk (a restart resumes), then
inflated locally and checked against the zip CRC32.

Usage: python -m csi.io.remote_zip <url> <out_dir> <member> [<member> ...]
"""
from __future__ import annotations

import http.client
import io
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 32 << 20
WORKERS = 4


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; the signed storage URL is refreshed on errors."""

    def __init__(self, url: str):
        self.source = url
        self.size = 0
        self.url = url
        self._retry(self._resolve)
        self.pos = 0

    def _resolve(self):
        req = urllib.request.Request(self.source, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
            self.url = r.geturl()

    def _retry(self, fn, attempts: int = 10):
        for attempt in range(attempts):
            try:
                return fn()
            except (OSError, http.client.HTTPException) as e:  # HTTPError, URLError, SSL, timeouts, IncompleteRead
                if attempt == attempts - 1:
                    raise
                wait = min(5 * 2 ** attempt, 300)
                print(f"  retry in {wait}s: {e}", flush=True)
                time.sleep(wait)
                try:
                    self._resolve()
                except (OSError, http.client.HTTPException):
                    pass

    def fetch(self, start: int, end: int, fresh: bool = False) -> bytes:
        """Bytes [start, end] inclusive. fresh=True asks figshare for a new signed URL first."""
        def get():
            url = self.url
            if fresh:
                req0 = urllib.request.Request(self.source, headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(req0, timeout=60) as r0:
                    url = r0.geturl()
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise OSError(f"short read {len(data)} of {end - start + 1}")
            return data
        return self._retry(get)

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.fetch(self.pos, self.pos + n - 1)
        b[:n] = data
        self.pos += n
        return n


def member_data_range(f: HttpRangeFile, info: zipfile.ZipInfo) -> tuple[int, int]:
    """Absolute offset and length of the compressed data of a member (after its local header)."""
    hdr = f.fetch(info.header_offset, info.header_offset + 29)
    sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", hdr)
    if sig != 0x04034B50:
        raise ValueError("bad local header")
    return info.header_offset + 30 + name_len + extra_len, info.compress_size


def extract(url: str, out_dir: Path, members: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = HttpRangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = {Path(i.filename).name: i for i in z.infolist()}
    for m in members:
        info = names[m]
        dst = out_dir / m
        if dst.exists() and dst.stat().st_size == info.file_size:
            print("skip", m, flush=True)
            continue
        start, length = member_data_range(f, info)
        parts = out_dir / (m + ".parts")
        parts.mkdir(exist_ok=True)
        n_chunks = (length + CHUNK - 1) // CHUNK
        t0 = time.time()
        todo = []
        for k in range(n_chunks):
            p = parts / f"{k:05d}"
            size = min(CHUNK, length - k * CHUNK)
            if not (p.exists() and p.stat().st_size == size):
                todo.append((k, p, size))

        def get_chunk(item):
            k, p, size = item
            data = f.fetch(start + k * CHUNK, start + k * CHUNK + size - 1, fresh=True)
            tmp = parts / f"{k:05d}.tmp"
            tmp.write_bytes(data)
            tmp.replace(p)
            return size

        got = 0
        with ThreadPoolExecutor(WORKERS) as ex:
            for size in ex.map(get_chunk, todo):
                got += size
                print(f"  {m}: +{got / 1e6:.0f} MB of {sum(s for *_, s in todo) / 1e6:.0f} MB left "
                      f"({got / 1e6 / max(time.time() - t0, 1e-3):.2f} MB/s)", flush=True)
        # inflate locally, verify CRC
        dec = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
        if info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            raise ValueError(f"unsupported compression {info.compress_type}")
        crc = 0
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as out:
            for k in range(n_chunks):
                raw = (parts / f"{k:05d}").read_bytes()
                data = dec.decompress(raw) if dec else raw
                crc = zlib.crc32(data, crc)
                out.write(data)
            if dec:
                tail = dec.flush()
                crc = zlib.crc32(tail, crc)
                out.write(tail)
        if crc != info.CRC or tmp.stat().st_size != info.file_size:
            raise ValueError(f"{m}: CRC or size mismatch")
        tmp.replace(dst)
        for p in parts.iterdir():
            p.unlink()
        parts.rmdir()
        print(f"ok {m} {info.file_size / 1e6:.0f} MB, CRC verified", flush=True)


if __name__ == "__main__":
    extract(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
