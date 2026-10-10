import os
import subprocess
import sys

import uvicorn

if __name__ == "__main__":
    on_space = bool(os.getenv("SPACE_ID"))  # set by Hugging Face Spaces
    if not os.path.exists("chroma_db"):
        # ponytail: rebuilt on every Space cold start (~minutes); commit chroma_db via LFS if that gets annoying
        subprocess.run([sys.executable, "-m", "src.ingest"], check=True)
    # Spaces expect the app on 7860; locally keep 8000 with auto-reload
    uvicorn.run("src.api:app", host="0.0.0.0", port=7860 if on_space else 8000, reload=not on_space)
