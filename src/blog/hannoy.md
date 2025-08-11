{
	"published": "2025-08-01",
  "description": "A rust port of Meta's FAISS HNSW implementation using LMDB"
}

# KV-backed HNSW in rust, Part I

This is the first in a series of articles on my experiences building [hannoy](bleh); a rust-based vector db based on hierarchical small worlds ([HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) ) and [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) for [Meilisearch](https://www.meilisearch.com/).

The point of this article is to introduce some concepts and motivate the HNSW, in later installments we'll dig into the code.

## Vector databases 101 {#vector-db}
Vector databases (or vector db's) allow you to perform search over a set of, well, vectors.

Generally speaking we have a collection of embeddings, $X = \\{x_{i}; i \in 1...N\\}$ with $x_{i} \in \mathbb{R}^{d}$, obtained through either sparse traditional feature selection methods (think [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) or, more commonly these days through dense neural nets, and we want to retrieve the most similar one given some query vector. Mathematically, that breaks down into a finding: $$x^{*} = \arg\min_{x_{i} \in X} d(q,x_{i})$$

Here, $d$ is a [metric](https://en.wikipedia.org/wiki/Metric_space) satisfying $d: \mathbb{R}^{d} \times \mathbb{R}^{d} \mapsto \mathbb{R}$.

The industry standard is just to use the cosine distance all the time which has the nice advantage of being semantically interpretable ($x^{*}$ is the vector in our db that makes the smallest angle with $q$) but there are plenty of other valid choices. For instance, Hannoy currently supports Euclidean, Cosine, Hamming, Manhattan, and various quantized versions thereof !

Solving the above equation can be done through brute force, but the compute and search latency scales linearly with respect to the size of the db. The goal of [approximate nearest neighbours (ANNs)](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods) search is to run in sub-linear time at the expense of missing a few good matches.

Like all worthwhile problems in life though, there's more than one way to skin a cat.

I previously contributed to [arroy](https://github.com/meilisearch/arroy), a rust-based [k-d trees](https://en.wikipedia.org/wiki/K-d_tree) approach used for a long time in Meilisearch's search engine, where vector search was accomplished through binary search over an ensemble of trees.

## HNSW {#hnsw}

## References {#references}
