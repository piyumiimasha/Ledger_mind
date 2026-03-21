"""
Weaviate Client Factory and Schema Management
Centralized Weaviate initialization following llm_services.py pattern.
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances, Tokenization


def get_weaviate_client(config: Dict[str, Any], max_retries: int = 3):
    """
    Factory function to create and return a Weaviate client.

    Args:
        config: Configuration dictionary with weaviate settings
        max_retries: Maximum number of connection retry attempts

    Returns:
        Connected Weaviate client instance

    Raises:
        RuntimeError: If connection fails after max_retries
    """
    mode = config["weaviate"]["mode"]

    for attempt in range(max_retries):
        try:
            if mode == "embedded":
                return _get_embedded_client(config)
            elif mode == "cloud":
                return _get_cloud_client(config)
            elif mode == "docker":
                return _get_docker_client(config)
            else:
                raise ValueError(f"Unknown Weaviate mode: {mode}")

        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to connect to Weaviate after {max_retries} attempts: {e}"
                )
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)


def _get_embedded_client(config: Dict[str, Any]):
    """Create Weaviate embedded client (Python-only, no server)."""
    embedded_config = config["weaviate"]["embedded"]

    # Ensure persistence directories exist
    persistence_path = Path(embedded_config["persistence_data_path"])
    persistence_path.mkdir(parents=True, exist_ok=True)

    binary_path = Path(embedded_config["binary_path"])
    binary_path.mkdir(parents=True, exist_ok=True)

    # Create embedded client
    client = weaviate.connect_to_embedded(
        version="1.24.4",  # Specify version for reproducibility
        persistence_data_path=str(persistence_path),
        binary_path=str(binary_path),
        headers={"X-OpenAI-Api-Key": ""},  # Dummy key for embedded mode
    )

    print("[OK] Connected to Weaviate Embedded")
    print(f"     Data: {persistence_path}")
    print(f"     Binary: {binary_path}")

    return client


def _get_cloud_client(config: Dict[str, Any]):
    """Create Weaviate Cloud Services (WCS) client."""
    cloud_config = config["weaviate"]["cloud"]

    if not cloud_config["cluster_url"] or not cloud_config["api_key"]:
        raise ValueError(
            "Weaviate cloud mode requires cluster_url and api_key in config.yaml"
        )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cloud_config["cluster_url"],
        auth_credentials=weaviate.auth.AuthApiKey(cloud_config["api_key"]),
    )

    print(f"[OK] Connected to Weaviate Cloud: {cloud_config['cluster_url']}")

    return client


def _get_docker_client(config: Dict[str, Any]):
    """Create Weaviate Docker client (self-hosted)."""
    docker_config = config["weaviate"]["docker"]

    client = weaviate.connect_to_local(
        host=docker_config["host"],
        port=docker_config["port"],
        grpc_port=docker_config["grpc_port"],
    )

    print(f"[OK] Connected to Weaviate Docker: {docker_config['host']}:{docker_config['port']}")

    return client


def create_schema(client, config: Dict[str, Any], force_recreate: bool = False):
    """
    Create Weaviate schema/collection for UberReportChunk.

    Args:
        client: Connected Weaviate client
        config: Configuration dictionary
        force_recreate: If True, delete and recreate collection

    Returns:
        Collection name
    """
    schema_config = config["weaviate"]["schema"]
    class_name = schema_config["class_name"]

    # Check if collection exists
    if client.collections.exists(class_name):
        if force_recreate:
            print(f"[WARN] Deleting existing collection: {class_name}")
            client.collections.delete(class_name)
        else:
            print(f"[OK] Collection already exists: {class_name}")
            return class_name

    # Verify embedding dimensions (should be 384 for MiniLM-L6-v2)
    from src.services.llm_services import get_text_embeddings

    embeddings = get_text_embeddings(config)
    test_vector = embeddings.embed_query("test")
    vector_dim = len(test_vector)

    if vector_dim != 384:
        print(f"[WARN] Warning: Expected 384 dimensions, got {vector_dim}")

    # Create collection with schema
    vector_config = schema_config["vector_index_config"]

    client.collections.create(
        name=class_name,
        description="Chunks from Uber 2024 Annual Report for hybrid RAG retrieval",
        vectorizer_config=Configure.Vectorizer.none(),  # We provide our own embeddings
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=vector_config["ef_construction"],
            ef=vector_config["ef"],
        ),
        properties=[
            Property(
                name="content",
                data_type=DataType.TEXT,
                description="Chunk text content",
                index_inverted=True,  # Enable BM25 search
                tokenization=Tokenization.WORD,  # Preserve entities like "Form 10-K"
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
                description="Original chunk index from text splitter",
            ),
            Property(
                name="source",
                data_type=DataType.TEXT,
                description="Source document filename",
            ),
            Property(
                name="splitter",
                data_type=DataType.TEXT,
                description="Chunking strategy used (fixed, recursive, etc.)",
            ),
        ],
    )

    print(f"[OK] Created collection: {class_name}")
    print(f"     Vector dimensions: {vector_dim}")
    print(f"     Distance metric: COSINE")
    print(f"     BM25 enabled: True")

    return class_name


def verify_connection(client):
    """
    Verify Weaviate connection and print cluster info.

    Args:
        client: Connected Weaviate client

    Returns:
        bool: True if connected successfully
    """
    try:
        # Check if cluster is ready
        if client.is_ready():
            print("[OK] Weaviate cluster is ready")

            # Get cluster metadata
            meta = client.get_meta()
            print(f"     Version: {meta.get('version', 'unknown')}")

            return True
        else:
            print("[ERROR] Weaviate cluster is not ready")
            return False

    except Exception as e:
        print(f"[ERROR] Connection verification failed: {e}")
        return False


def get_collection_stats(client, class_name: str) -> Dict[str, Any]:
    """
    Get statistics about a Weaviate collection.

    Args:
        client: Connected Weaviate client
        class_name: Name of the collection

    Returns:
        Dictionary with collection statistics
    """
    try:
        collection = client.collections.get(class_name)

        # Get aggregation data
        agg_result = collection.aggregate.over_all(total_count=True)

        stats = {
            "class_name": class_name,
            "total_objects": agg_result.total_count,
            "exists": True,
        }

        print(f"Collection: {class_name}")
        print(f"     Total objects: {agg_result.total_count}")

        return stats

    except Exception as e:
        print(f"[ERROR] Failed to get collection stats: {e}")
        return {"class_name": class_name, "exists": False, "error": str(e)}
