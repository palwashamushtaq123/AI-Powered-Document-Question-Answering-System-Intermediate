# Vector Storage Mechanics and Similarity Search in ChromaDB

Vector databases are specialized storage solutions designed to store, index, and query dense mathematical vectors efficiently. Traditional SQL or NoSQL databases index structured scalar values, making semantic or distance-based queries computationally prohibitive at scale.

ChromaDB uses Hierarchical Navigable Small World (HNSW) graphs to conduct approximate nearest neighbor (ANN) searches. Rather than scanning every single vector exhaustively across large datasets, HNSW builds a multi-layer graph structure that allows logarithmic search complexity $O(\log N)$.

Distance metrics dictate how vector similarity is computed:
1. Cosine Similarity / Distance: Measures the angle between two vectors regardless of magnitude. It is the standard choice for normalized text embeddings where directional alignment represents semantic similarity.
2. Euclidean Distance (L2): Measures the direct straight-line spatial distance between two vectors.
3. Dot Product: Measures vector magnitude and angle combination, commonly used in fine-tuned dense retrieval models.