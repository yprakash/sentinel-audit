import hashlib
from typing import Optional


def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def calculate_sha256(file_path: str) -> Optional[str]:
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()
