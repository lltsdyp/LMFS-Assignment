//! Shared model tables, reusable worker state, and ordered parallel encoding.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use rayon::prelude::*;

use crate::encoding::{MergeRanks, MergeScratch, merge_word};
use crate::pretokenization::Span;

const MAX_CACHE_ENTRIES: usize = 65536;
const MAX_CACHE_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;
const MAX_CACHE_WORD_BYTES: usize = 128;
const PARALLEL_MIN_WORDS: usize = 1024;
const WORDS_PER_TASK: usize = 512;

type Cache = HashMap<Vec<u8>, Vec<usize>>;

struct Worker {
    hot: Cache,
    cold: Cache,
    hot_bytes: usize,
    generation_entries: usize,
    generation_bytes: usize,
    scratch: MergeScratch,
}

impl Worker {
    fn new(slots: usize) -> Self {
        Self {
            hot: Cache::new(),
            cold: Cache::new(),
            hot_bytes: 0,
            generation_entries: MAX_CACHE_ENTRIES / slots / 2,
            generation_bytes: MAX_CACHE_PAYLOAD_BYTES / slots / 2,
            scratch: MergeScratch::default(),
        }
    }

    fn insert(&mut self, word: &[u8], ids: Vec<usize>) {
        let bytes = word.len() + ids.len() * size_of::<usize>();
        if self.generation_entries == 0 || bytes > self.generation_bytes {
            return;
        }
        if self.hot.len() >= self.generation_entries
            || self.hot_bytes + bytes > self.generation_bytes
        {
            // Retain the recent generation instead of clearing every cached word.
            std::mem::swap(&mut self.hot, &mut self.cold);
            self.hot.clear();
            self.hot_bytes = 0;
        }
        self.hot_bytes += bytes;
        self.hot.insert(word.to_vec(), ids);
    }
}

pub struct Encoder {
    token_to_index: HashMap<Vec<u8>, usize>,
    merge_ranks: MergeRanks,
    workers: Vec<Mutex<Worker>>,
}

impl Encoder {
    pub fn new(token_to_index: HashMap<Vec<u8>, usize>, merges: Vec<(Vec<u8>, Vec<u8>)>) -> Self {
        let mut merge_ranks: MergeRanks = HashMap::new();
        for (rank, (left, right)) in merges.into_iter().enumerate() {
            merge_ranks
                .entry(left)
                .or_default()
                .entry(right)
                .or_insert(rank);
        }
        let slots = rayon::current_num_threads() + 1;
        Self {
            token_to_index,
            merge_ranks,
            workers: (0..slots).map(|_| Mutex::new(Worker::new(slots))).collect(),
        }
    }

    pub fn encode_spans(&self, text: &str, spans: &[Span]) -> Result<Vec<usize>, Vec<u8>> {
        if spans.len() < PARALLEL_MIN_WORDS {
            return self.encode_chunk(text, spans, self.workers.len() - 1);
        }
        let chunks: Vec<Result<Vec<usize>, Vec<u8>>> = spans
            .par_chunks(WORDS_PER_TASK)
            .map(|chunk| {
                let slot = rayon::current_thread_index().unwrap_or(self.workers.len() - 1);
                self.encode_chunk(text, chunk, slot)
            })
            .collect();
        let mut ids = Vec::new();
        for chunk in chunks {
            ids.extend(chunk?);
        }
        Ok(ids)
    }

    fn encode_chunk(&self, text: &str, spans: &[Span], slot: usize) -> Result<Vec<usize>, Vec<u8>> {
        let mut worker = self.workers[slot]
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut ids = Vec::with_capacity(spans.len());
        for span in spans {
            let word = &text.as_bytes()[span.start..span.end];
            if span.special {
                ids.push(*self.token_to_index.get(word).ok_or_else(|| word.to_vec())?);
            } else if let Some(cached) = worker.hot.get(word).or_else(|| worker.cold.get(word)) {
                ids.extend_from_slice(cached);
            } else {
                let encoded = merge_word(
                    word,
                    &self.token_to_index,
                    &self.merge_ranks,
                    &mut worker.scratch,
                )?;
                ids.extend_from_slice(encoded);
                if word.len() <= MAX_CACHE_WORD_BYTES {
                    let cached = encoded.to_vec();
                    worker.insert(word, cached);
                }
            }
        }
        worker.scratch.release_large_buffers();
        Ok(ids)
    }
}
