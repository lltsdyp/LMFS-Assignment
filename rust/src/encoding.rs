//! Ranked merging over byte spans, without allocating one object per symbol.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub type MergeRanks = HashMap<Vec<u8>, HashMap<Vec<u8>, usize>>;
type PairHeap = BinaryHeap<Reverse<(usize, usize, usize)>>;
const NONE: usize = usize::MAX;

#[derive(Default)]
pub struct MergeScratch {
    ends: Vec<usize>,
    previous: Vec<usize>,
    following: Vec<usize>,
    heap: PairHeap,
    ids: Vec<usize>,
}

impl MergeScratch {
    pub fn release_large_buffers(&mut self) {
        if self.ends.capacity() > 4096 || self.heap.capacity() > 16384 {
            *self = Self::default();
        }
    }
}

struct Word<'a> {
    bytes: &'a [u8],
    ends: &'a mut [usize],
    previous: &'a mut [usize],
    following: &'a mut [usize],
}

impl<'a> Word<'a> {
    fn symbol(&self, index: usize) -> &[u8] {
        &self.bytes[index..self.ends[index]]
    }

    fn rank(&self, left: usize, right: usize, ranks: &MergeRanks) -> Option<usize> {
        ranks
            .get(self.symbol(left))?
            .get(self.symbol(right))
            .copied()
    }

    fn push_pair(&self, heap: &mut PairHeap, left: usize, ranks: &MergeRanks) {
        if left == NONE {
            return;
        }
        let right = self.following[left];
        if right != NONE
            && let Some(rank) = self.rank(left, right, ranks)
        {
            heap.push(Reverse((rank, left, right)));
        }
    }
}

pub fn merge_word<'a>(
    bytes: &[u8],
    token_to_index: &HashMap<Vec<u8>, usize>,
    ranks: &MergeRanks,
    scratch: &'a mut MergeScratch,
) -> Result<&'a [usize], Vec<u8>> {
    let MergeScratch {
        ends,
        previous,
        following,
        heap,
        ids,
    } = scratch;
    let length = bytes.len();
    ends.resize(length, 0);
    previous.resize(length, 0);
    following.resize(length, 0);
    for i in 0..length {
        ends[i] = i + 1;
        previous[i] = i.checked_sub(1).unwrap_or(NONE);
        following[i] = if i + 1 < length { i + 1 } else { NONE };
    }
    let word = Word {
        bytes,
        ends,
        previous,
        following,
    };
    heap.clear();
    ids.clear();
    for left in 0..bytes.len().saturating_sub(1) {
        word.push_pair(heap, left, ranks);
    }
    while let Some(Reverse((rank, left, right))) = heap.pop() {
        if word.ends[left] == left
            || word.following[left] != right
            || word.rank(left, right, ranks) != Some(rank)
        {
            continue;
        }
        word.ends[left] = word.ends[right];
        word.ends[right] = right;
        word.following[left] = word.following[right];
        if word.following[right] != NONE {
            word.previous[word.following[right]] = left;
        }
        word.push_pair(heap, word.previous[left], ranks);
        word.push_pair(heap, left, ranks);
    }
    for index in 0..bytes.len() {
        if word.ends[index] != index {
            let symbol = word.symbol(index);
            ids.push(*token_to_index.get(symbol).ok_or_else(|| symbol.to_vec())?);
        }
    }
    Ok(ids.as_slice())
}
