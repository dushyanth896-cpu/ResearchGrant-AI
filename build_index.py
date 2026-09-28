import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

# Anchor paths to this file's location instead of the current
# working directory, so `python build_index.py` produces the
# same result no matter where it's run from.
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
INDEX_FILE = BASE_DIR / "grant_index.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# JSON → SEARCHABLE TEXT
# ============================================================

def json_to_text(data):
    lines = []

    if isinstance(data, dict):

        for key, value in data.items():

            if isinstance(value, (dict, list)):

                lines.append(f"{key}:")

                nested = json_to_text(value)

                for line in nested.splitlines():
                    lines.append("  " + line)

            else:
                lines.append(f"{key}: {value}")

    elif isinstance(data, list):

        for item in data:

            if isinstance(item, (dict, list)):

                nested = json_to_text(item)

                for line in nested.splitlines():
                    lines.append("- " + line)

            else:
                lines.append(f"- {item}")

    else:
        lines.append(str(data))

    return "\n".join(lines)


# ============================================================
# LOAD GRANTS
# ============================================================

def load_grants():

    json_files = sorted(DATA_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {DATA_DIR.resolve()}"
        )

    print(f"Found {len(json_files)} JSON files.")

    grants = []

    for file_path in json_files:

        try:

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):

                if "grants" in data and isinstance(data["grants"], list):
                    items = data["grants"]

                elif "documents" in data and isinstance(data["documents"], list):
                    items = data["documents"]

                else:
                    items = [data]

            elif isinstance(data, list):

                items = data

            else:

                print(f"Skipping invalid JSON: {file_path}")
                continue

            for item in items:

                if not isinstance(item, dict):
                    continue

                text = json_to_text(item)

                if text.strip():

                    grants.append({
                        # Store a plain filename (not an absolute or
                        # OS-specific path) so the index is portable
                        # between machines and operating systems.
                        "source": file_path.name,
                        "data": item,
                        "text": text
                    })

        except Exception as e:

            print(f"Error reading {file_path}: {e}")

    if not grants:
        raise ValueError("No usable grant documents found.")

    print(f"Loaded {len(grants)} grant documents.")

    return grants


# ============================================================
# CREATE INDEX
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING RESEARCHGRANT AI INDEX")
    print("=" * 70)

    print()
    print("Loading grant documents...")

    grants = load_grants()

    print()
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print()
    print("Creating embeddings...")

    texts = [grant["text"] for grant in grants]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print()
    print("Saving index...")

    index_data = {
        "embedding_model": EMBEDDING_MODEL,
        "grants": grants,
        "embeddings": embeddings
    }

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index_data, f)

    print()
    print("=" * 70)
    print("INDEX CREATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(f"Grant documents : {len(grants)}")
    print(f"Embeddings      : {embeddings.shape}")
    print(f"Index file      : {INDEX_FILE.resolve()}")
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
