{
	"published": "2024-05-10",
  "description": "An O(m*log n) optimization I found to speed up tokenizer throughput"
}

# Fast byte pair encoding in rust

In this article I'll run through how I used Rust and pyo3 to implement a fast BPE tokenizer (4x faster than [tokenizers](https://github.com/huggingface/tokenizers) and as fast as [tiktoken](https://github.com/openai/tiktoken)) which you can install from PyPI today!

All the code mentioned in this post can be found [on github](https://github.com/nnethercott/tok).

<figure>
<div style="text-align: center;">
    <img src="/static/images/tok_post/performance.png" style="width: 80%; display: block; margin: 0 auto;" >
    <figcaption>
      Speed comparison between my tokenizer (yellow) and popular libraries like Hugging Face and OpenAI
    </figcaption>
</div>
</figure>



This article is about an efficient way I discovered to quickly encode and decode sequences using a pretrained tokenizer rather than an in-depth review of byte-pair encoding. For that feel free to check out any of these links: [wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding), [huggingface](https://huggingface.co/learn/nlp-course/en/chapter6/5), [tiktoken](https://github.com/openai/tiktoken).


## A naive approach - *O(mn)* {#naive}

Suppose you've already trained a tokenizer (doing this is trivial), i.e. you have a some hashmap-like object that lets you map a sequence of bytes to unique tokens. What's the fastest way to use this lookup table to encode and decode your data?

<!-- ### A basic approach -->

The easiest approach is to loop over all the possible token merges in the order we learned them and replace any matches we find along the way. For example, if your tokenizer has the following token mapping rules:

```
{(97, 97): 128, (128,97): 129, (129, 98): 130}
```

Then the encoding for "aaabcaaab" (or [97,97,97,98,99,97,97,97,98] as a byte array) would look like this (notice how everything gets *compressed*):

```
1. [97,97,97,98,99,97,97,97,98] -> [128,97,98,99,128,97,98]
2. [128,97,98,99,128,97,98] -> [129,98,99,129,98]
3. [129,98,99,129,98] -> [130,99,130]
```

In Rust you can write that loop as:

```rust
use std::collections::HashMap;
type Rank = u32;

fn _byte_pair_merge(pieces: &mut Vec<Rank>, pair: (Rank, Rank), replace: Rank) {
    let mut i = 0;
    while i < pieces.len() - 1 {
        if (pieces[i], pieces[i + 1]) == pair {
            pieces[i] = replace;
            pieces.remove(i + 1);
        }
        i += 1;
    }
}

fn encode(text: &str, map: &HashMap<(Rank, Rank), Rank>) -> Vec<Rank> {
    let mut pieces: Vec<Rank> = text.as_bytes().iter().map(|&x| x as Rank).collect();
    //reverse (k,v) to (v,k)
    let reverse_map: HashMap<Rank, (Rank, Rank)> = map.iter().map(|(&p, &r)| (r, p)).collect();

    //O(m*n)
    //assume first token has index 128 since we're encoding for ascii
    (128..=reverse_map.len() + 128).rev().for_each(|i| {
        let &pair = reverse_map.get(&(i as Rank)).unwrap();
        _byte_pair_merge(&mut pieces, pair, i as Rank);
    });

    pieces
}
```

For a vocabulary size of 50257, the token throughput with this approach for a 2.5MB subset of the [wikitext dataset](https://huggingface.co/datasets/wikitext) is somewhere in the neighborhood of **0.09MB/s**.

This approach definitely gets the job done but it's incredibly inefficient! Indeed, this solution has a time complexity of $O(mn)$, where $m$ is the vocab size and $n$ is the length of the text you want to encode. As the vocabulary size and/or length of the text increase we get significant slowdowns.

## A better solution  - *O(m log(n))* {#efficient}

The approach I ended up stumbling across after around 6 hours of refactoring is closer to $O(m\log{n})$. It relies on the fact that we don't need to loop over every entry in the hashmap when it's sufficient to notice that we can apply merges in a way which respects the order the tokenizer learned them in. This lets us apply multiple different token merges in a single pass instead of only searching for a single pattern each time. We can also detect early on if no more token merging is possible and break out of the function.

If you've been grinding your fair share of Leetcode this sort of [two-pointers approach](https://www.geeksforgeeks.org/dsa/two-pointers-technique/) should feel familiar...

```rust
//lib.rs
fn encode(text: &str, map: Map<(Rank, Rank), Rank>) -> Vec<Rank> {
    let mut pieces: Vec<Rank> = text.as_bytes().iter().map(|&x| x as Rank).collect();

    loop {
        let mut merges = Vec::new();
        for i in 0..pieces.len() - 1 {
            if let Some(&rank) = map.get(&(pieces[i], pieces[i + 1])) {
                merges.push((i, rank));
            }
        }
        //early stopping
        if merges.is_empty() {
            break;
        }

        // apply merges and swap in tokens from reverse
        let mut i = merges.len() - 1;
        while i > 0 {
            let x = &mut merges[i - 1..=i];
            let l = x[0];
            let r = x[1];

            if r.0 - l.0 > 1 && r.1 != Rank::MAX {
                pieces[r.0] = r.1;
                pieces.remove(r.0 + 1);
            } else if r.1 < l.1 {
                pieces[r.0] = r.1;
                pieces.remove(r.0 + 1);

                x[0].1 = Rank::MAX;
                i -= 1;
            }
            //avoid overflow on usize 0-1
            if i == 0 {
                break;
            }
            i -= 1;
        }
        if merges.len() == 1 || merges[0].1 < merges[1].1 {
            pieces[merges[0].0] = merges[0].1;
            pieces.remove(merges[0].0 + 1);
        }
    }
    pieces
}
```

On the same wikitext split our throughput using this encoding algorithm jumps to **24.35MB/s**. That's over a 100x improvement with respect to where we started from.

I took a lot of inspiration from official [openai implementation](https://github.com/openai/tiktoken/blob/main/src/lib.rs) in their repo `tiktoken` but handled the merging aspect quite differently by leveraging the fact that we could store the prospective merges in a stack instead of finding the single-best merge at each iteration.

## Python bindings {#python-bindings}

To expose the Rust code in Python I made use of [pyo3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin). Getting started with these libraries is incredibly easy and just requires adding a few pyo3 attributes to your existing rust code. What's also nice is that maturin automatically adds a CI workflow to your project which makes distributing your python package infinitely easier. By default the workflow listens for new tag pushes to the main branch and builds the wheels for all the major platforms.

I encourage you to check out [a few official examples](https://github.com/PyO3/setuptools-rust/tree/main/examples) and the [pyo3 docs](https://pyo3.rs/v0.21.2/), overall though its a pretty frictionless experience.

Using maturin I published [`toktokenizer`](https://pypi.org/project/toktokenizer/) - a lightweight python package for BPE tokenizers - to PyPI. The only class the library exposes is `BPETokenizer`. The class itself is pretty minimal, with all major methods being showed below:

```python
# demo.py
from toktokenizer import BPETokenizer

bpe = tok.BPETokenizer()

# train a byte-pair tokenizer on some corpus
train_corpus = "this is some training data"
vocab_size = 8
bpe.train(train_corpus, vocab_size)

# save tokenizer state
bpe.save_encoder("8word.json")

# load tokenizer from dumped file
bpe.load_encoder("8word.json")

# encode and decode
input_ids = bpe.encode("some data")
decoded = bpe.decode(input_ids)
```

To get started with `toktokenizer` today you can install it with pip as follows:

```
pip install toktokenizer
```

I'm looking forward to using this library moving forwards as I build up various components of modern NLP models from scratch!

