"""
Download ArcFace ONNX model for face embedding (employee recognition).
Run once: python -m scripts.download_arcface
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "arcface.onnx"


def main():
    out = DEFAULT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Model already exists: {out}")
        return
    print(f"Downloading ArcFace ONNX to {out} ...")
    urllib.request.urlretrieve(URL, out)
    print("Done. Set config: face_embedding_model_path = './models/arcface.onnx'")


if __name__ == "__main__":
    main()
