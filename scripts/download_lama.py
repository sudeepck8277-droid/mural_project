"""Download the pinned pretrained Big-LaMa TorchScript release into models/."""

import hashlib
from pathlib import Path
import urllib.request


URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/"
    "releases/download/v0.1.0/big-lama.pt"
)
SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
SIZE = 205803670
DESTINATION = Path(__file__).resolve().parents[1] / "models" / "big-lama.pt"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if DESTINATION.exists():
        if DESTINATION.stat().st_size == SIZE and checksum(DESTINATION) == SHA256:
            print(f"Verified existing LaMa checkpoint: {DESTINATION}")
            return
        raise RuntimeError(
            f"An unexpected checkpoint already exists at {DESTINATION}. "
            "Move it aside before downloading the pinned release."
        )

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    partial = DESTINATION.with_suffix(".pt.download")
    print(f"Downloading pretrained Big-LaMa ({SIZE / 1_000_000:.1f} MB)...", flush=True)
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
                if downloaded - last_report >= 16 * 1024 * 1024:
                    print(f"  {min(100, downloaded * 100 / SIZE):.0f}%", flush=True)
                    last_report = downloaded
        if downloaded != SIZE or checksum(partial) != SHA256:
            raise RuntimeError("LaMa download failed size/SHA-256 verification.")
        partial.replace(DESTINATION)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"Verified and installed: {DESTINATION}")


if __name__ == "__main__":
    main()
