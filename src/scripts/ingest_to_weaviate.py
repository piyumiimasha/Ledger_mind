"""
Direct Ingestion Script: Chunks JSONL -> Weaviate
Loads pre-chunked data from chunks.jsonl to ensure consistency with Q/A pairs.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from src.services.weaviate_client import get_weaviate_client, create_schema
from src.services.llm_services import load_config, get_text_embeddings


def ingest_chunks_to_weaviate(force_recreate: bool = False):
    """
    Load pre-chunked data from chunks.jsonl and ingest into Weaviate.

    Uses the same chunks that were used for Q/A pair generation to ensure
    chunk_index consistency between Weaviate and the evaluation dataset.

    Args:
        force_recreate: If True, delete and recreate Weaviate collection

    Returns:
        Ingestion success status
    """
    print("=" * 80)
    print("CHUNKS JSONL -> WEAVIATE INGESTION")
    print("=" * 80)

    # 1. Load configuration
    print("\n[1] Loading configuration...")
    config = load_config("src/config/config.yaml")
    print("    [OK] Config loaded")

    # 2. Load pre-chunked data from JSONL
    print("\n[2] Loading chunks from JSONL...")
    chunks_path = Path("data/cleaned/chunks.jsonl")

    if not chunks_path.exists():
        print(f"    [ERROR] Chunks file not found: {chunks_path}")
        print("    Please run notebooks/01_clean_dataset.ipynb first to generate chunks.")
        return False

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))

    num_chunks = len(chunks)
    print(f"    [OK] Loaded {num_chunks} chunks from {chunks_path.name}")

    # 3. Initialize embeddings
    print("\n[3] Loading embedding model...")
    embeddings = get_text_embeddings(config)
    print(f"    [OK] Embeddings ready: {config['text_emb_model']}")

    # 4. Generate embeddings for chunks
    print(f"\n[4] Generating embeddings for {num_chunks} chunks...")
    chunk_embeddings = []

    for chunk in tqdm(chunks, desc="Encoding chunks", unit="chunk"):
        vector = embeddings.embed_query(chunk["content"])
        chunk_embeddings.append(vector)

    print(f"    [OK] Generated {len(chunk_embeddings)} embeddings")
    print(f"         Vector dimensions: {len(chunk_embeddings[0])}")

    # 5. Initialize Weaviate
    print("\n[5] Initializing Weaviate...")
    client = get_weaviate_client(config)
    class_name = create_schema(client, config, force_recreate=force_recreate)

    # 6. Batch import to Weaviate
    print(f"\n[6] Importing to Weaviate...")
    print(f"    Collection: {class_name}")
    print(f"    Batch size: 100")

    collection_obj = client.collections.get(class_name)

    # Import with progress bar
    with tqdm(total=num_chunks, desc="Ingesting chunks", unit="chunk") as pbar:
        # Process in batches of 100
        batch_size = 100
        for i in range(0, num_chunks, batch_size):
            batch_end = min(i + batch_size, num_chunks)

            # Prepare batch data
            with collection_obj.batch.dynamic() as batch:
                for j in range(i, batch_end):
                    # Extract source filename from path
                    source_path = chunks[j].get("source", "unknown")
                    source_name = Path(source_path).name if source_path else "unknown"

                    data_object = {
                        "content": chunks[j]["content"],
                        "chunk_index": chunks[j]["chunk_index"],
                        "source": source_name,
                        "splitter": "fixed",
                    }

                    batch.add_object(properties=data_object, vector=chunk_embeddings[j])

            pbar.update(batch_end - i)

    # 7. Verify ingestion
    print(f"\n[7] Verifying ingestion...")
    from src.services.weaviate_client import get_collection_stats

    stats = get_collection_stats(client, class_name)
    weaviate_count = stats["total_objects"]

    print(f"\n{'=' * 80}")
    print("INGESTION SUMMARY")
    print(f"{'=' * 80}")
    print(f"    Source file: {chunks_path.name}")
    print(f"    Chunks loaded: {num_chunks}")
    print(f"    Weaviate objects: {weaviate_count}")

    if num_chunks == weaviate_count:
        print("    Status: [OK] SUCCESS - All chunks ingested!")
    else:
        print("    Status: [WARN] MISMATCH - Check for errors")

    # 8. Save ingestion metadata
    print(f"\n[8] Saving ingestion metadata...")
    metadata = {
        "source": "chunks_jsonl",
        "source_path": str(chunks_path),
        "destination": "weaviate",
        "destination_collection": class_name,
        "num_chunks": num_chunks,
        "num_weaviate_objects": weaviate_count,
        "chunking_strategy": "fixed",
        "embedding_model": config["text_emb_model"],
        "embedding_dimensions": len(chunk_embeddings[0]),
        "ingestion_timestamp": datetime.now().isoformat(),
        "success": num_chunks == weaviate_count,
    }

    metadata_path = Path(config["artifacts_root"]) / "weaviate_ingestion.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    [OK] Metadata saved: {metadata_path}")

    # 9. Test query
    print(f"\n[9] Testing query...")
    try:
        # Simple test query
        test_vector = embeddings.embed_query("Uber revenue")
        response = collection_obj.query.near_vector(
            near_vector=test_vector, limit=3, return_properties=["content", "chunk_index"]
        )

        if response.objects:
            print("    [OK] Test query successful")
            print(f"    Top result: Chunk {response.objects[0].properties['chunk_index']}")
            print(f"    Preview: {response.objects[0].properties['content'][:100]}...")
        else:
            print("    [WARN] Test query returned no results")

    except Exception as e:
        print(f"    [WARN] Test query failed: {e}")

    # Close connection
    client.close()
    print("\nWeaviate connection closed")

    print(f"\n{'=' * 80}")
    print("[OK] INGESTION COMPLETE!")
    print(f"{'=' * 80}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest chunks from JSONL to Weaviate")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate Weaviate collection (deletes existing data)",
    )
    args = parser.parse_args()

    success = ingest_chunks_to_weaviate(force_recreate=args.force)

    if success:
        print("\n[OK] You can now use query_librarian() to query the Weaviate collection!")
        print("\nQuick test:")
        print('  from src.services.query_librarian import query_librarian')
        print('  answer = query_librarian("What is Uber\'s mission?", verbose=True)')
        print('  print(answer)')
        sys.exit(0)
    else:
        print("\n[ERROR] Ingestion failed. Check errors above.")
        sys.exit(1)
