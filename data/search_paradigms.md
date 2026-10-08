# Comprehensive Comparison: Lexical Keyword vs Semantic Vector Search

Information retrieval systems are broadly categorized into two fundamental search paradigms: Lexical Search and Semantic Search.

Lexical Keyword Search:
Lexical systems (such as BM25, TF-IDF, and Lucene) operate on exact term frequency and inverted index matching. They evaluate how often query tokens appear in documents relative to the entire corpus. While highly performant for exact alphanumeric codes, SKU lookups, and specialized domain acronyms, lexical search fails when queries contain natural language synonyms, paraphrasing, or conceptual intent without word overlap.

Semantic Vector Search:
Semantic systems leverage dense vector embeddings to evaluate conceptual relationships. By embedding query strings into the same vector space as stored document chunks, semantic search identifies relevant context based on intent and mathematical proximity. For instance, a search query for "automobile maintenance" can retrieve document chunks discussing "car repair", despite sharing zero matching lexical keywords.