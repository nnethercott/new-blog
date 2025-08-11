{
	"published": "2025-08-01",
  "description": "A rust port of Meta's FAISS HNSW implementation using LMDB"
}

# KV-backed HNSW in rust, Part I

This is the first in a series of articles on my experiences building [hannoy](bleh); a rust-based vector db based on hierarchical small worlds ([HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) ) and [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) used in [Meilisearch](https://www.meilisearch.com/).

The point of this article is to introduce some concepts and motivate the HNSW, in later installments we'll dig into the code.

## Vector databases 101 {#vector-db}
Vector databases (or vector db's) allow you to perform search over a set of, well, vectors.

Generally speaking we start with a collection of embeddings

$$X = \\{x_{i}; i \in 1...N\\}, \quad x_{i} \in \mathbb{R}^{d},$$

obtained through either sparse traditional feature selection methods (think [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) or, more commonly these days through dense neural nets. Given some query vector, $q$, we want to find the most similar embedding in $X$. Mathematically, that boils down to:

<div id="eq1">
  $$x^{*} = \arg\min_{x_{i} \in X} d(q,x_{i})\tag{1}$$
</div>

Here, $d$ is a [metric](https://en.wikipedia.org/wiki/Metric_space) satisfying $d: \mathbb{R}^{d} \times \mathbb{R}^{d} \mapsto \mathbb{R}$.

The industry standard is just to use the cosine distance all the time which has the nice advantage of being semantically interpretable ($x^{*}$ is the vector in our db that makes the smallest angle with $q$) but there are plenty of other valid choices. For instance, Hannoy currently supports Euclidean, Cosine, Hamming, Manhattan, and various quantized versions thereof !

Solving [(1)](#eq1) can be done through brute force, but the compute and search latency scales linearly with respect to the size of the db. The goal of [approximate nearest neighbours (ANNs)](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods) search is instead to run in sub-linear time at the expense of missing a few good matches. _maybe comment about general acceptable ranges?_

Like all worthwhile problems in life though, there's more than one way to skin a cat. For an overview of what's out there, check out the anns benchmark [link](bleh).


## HNSW {#hnsw}

## temp {#temp}
Let's take a look at how they solve this in [arroy](https://github.com/meilisearch/arroy); a rust-based [k-d trees](https://en.wikipedia.org/wiki/K-d_tree) solution.

## References {#references}
