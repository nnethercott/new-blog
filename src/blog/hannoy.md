{
	"published": "2025-08-01",
  "description": "This is the first in a series of articles on my experiences building hannoy; a key-value backed hnsw implementation in rust using LMDB."
}

<!-- # From 2-Day Builds to 2-Hour KV-Backed HNSW -->
# KV-backed HNSW in rust, Part I

## A bit of context {#context}
Back in April I started contributing to an open-source vector db called [arroy](https://github.com/meilisearch/arroy) created by [Meilisearch](https://github.com/meilisearch/meilisearch) to handle semantic search in their product. It was a lot of fun, and eventually I got talking with their CTO about rewriting the project using a graph-based techniques instead of k-d trees purely out of scientific curiosity. 

Their current system was using [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) to circumvent the limitations of most vector db's which store the index entirely in memory. This way you could perform search over 1TB of data on a single replica without needing 1TB of RAM, albeit at the cost of higher latency due to SSD reads. It was important to keep this feature for a few reasons, firstly I didn't want to offend the CTO who also maintains the official rust bindings for LMDB, and second, I'm all about democratizing access to AI -- [you shouldn't need 3 machines to run your index](https://blog.wilsonl.in/diskann/).

The project that came out of this effort was [**hannoy**](https://github.com/nnethercott/hannoy). As it turns out, making this switch brought significant performance boosts; indexes took up half the previous disk space, build times were cut down from 2 days to 2 hours, and search became 10x faster. Currently [hannoy is on track to replace arroy in Meilisearch](https://github.com/meilisearch/meilisearch/pull/5767).

I learned a lot developing hannoy and figured I'd write it down for posteriority as well as for others who may be interested by the subject. I'll also make the bold claim here that hannoy is more DiskANN than it is hnsw. 

In this and subsequent articles, I'll go through vector search fundamentals, provide overviews of popular approaches, and show code samples for various implementation details.

## Vector databases 101 {#vector-db}
Vector databases (or vector db's) allow you to perform search over a set of, well, vectors.

Generally speaking we start with a collection of embeddings obtained through either traditional feature selection methods (think [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) or, more commonly these days through dense neural nets.

$$X = \\{x_{i}; i \in 1...N\\}, \quad x_{i} \in \mathbb{R}^{d},$$

These "embeddings" correspond to vectorized media like documents or images, and they have spatial semantics which make them navigable in the first place. Given some query vector, $q$, we want to find the most similar embedding in $X$. Mathematically, that looks like:

<div id="eq1">
  $$x^{*} = \arg\min_{x_{i} \in X} d(q,x_{i})\tag{1}$$
</div>

Here, $d$ is a [metric](https://en.wikipedia.org/wiki/Metric_space) satisfying $d: \mathbb{R}^{d} \times \mathbb{R}^{d} \mapsto \mathbb{R}$.

The industry standard is just to use the cosine distance all the time which has the nice advantage of being semantically interpretable ($x^{*}$ is the vector in our db that makes the smallest angle with $q$) but there are plenty of other valid choices. For instance, Hannoy currently supports Euclidean, Cosine, Hamming, Manhattan, and various quantized versions thereof. 

It turns out [(1)](#eq1) is actually just a specific case of $k$-nearest neighbour search with $k=1$, who's objective can be generalized as:

<div id="eq2">
  $$\left(x_{1}^{*}, ..., x_{k}^{*}, *,...,*\right) = \underset{x_{i} \in X}{\text{argsort }} d(q,x_{i})\tag{2}$$
</div>


Solving [(2)](#eq2) can be done through brute force, but the compute and search latency scales linearly with respect to the size of the db. The goal of [approximate nearest neighbours (ANNs)](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods) search is instead to run in sub-linear time at the expense of missing a few good matches. Solutions in the market today run search in ~4ms for 10 million documents. FIXME and add citation



Like all worthwhile problems in life though, there's more than one way to skin a cat. For an overview of what's out there, check out the anns benchmark [link](bleh).

## Locality-sensitive hashing (LSH) {#lsh}
Let's take a look at how they solve this in [arroy](https://github.com/meilisearch/arroy); a rust-based [k-d trees](https://en.wikipedia.org/wiki/K-d_tree) solution. Maybe discuss locality sensitive hashing

## HNSW {#hnsw}


## References {#references}
- [The FAISS Library](https://arxiv.org/abs/2401.08281), by Douze et al.
