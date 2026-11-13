"""Install only ZeroScratches' pretrained detection checkpoint."""

import hashlib
from pathlib import Path
import urllib.request


URL = (
    "https://huggingface.co/leonelhs/zeroscratches/resolve/main/"
    "detection/FT_Epoch_latest.pt"
)
SHA256 = "b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95"
SIZE = 451663663
DESTINATION = (
    Path(__file__).resolve().parents[1] / "models" / "zeroscratches"
    / "detection" / "FT_Epoch_latest.pt"
)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if DESTINATION.exists():
        if DESTINATION.stat().st_size == SIZE and checksum(DESTINATION) == SHA256:
            print(f"Verified existing ZeroScratches checkpoint: {DESTINATION}")
            return
        raise RuntimeError(
            f"Unexpected checkpoint at {DESTINATION}. Move it aside before downloading."
        )
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    partial = DESTINATION.with_suffix(".pt.download")
    print(f"Downloading ZeroScratches detector ({SIZE / 1_000_000:.1f} MB)...", flush=True)
    downloaded = 0
    last_report = 0
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 32 * 1024 * 1024:
                    print(f"  {min(100, downloaded * 100 / SIZE):.0f}%", flush=True)
                    last_report = downloaded
        if downloaded != SIZE or checksum(partial) != SHA256:
            raise RuntimeError("ZeroScratches download failed size/SHA-256 verification.")
        partial.replace(DESTINATION)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"Verified and installed: {DESTINATION}")


if __name__ == "__main__":
    main()
