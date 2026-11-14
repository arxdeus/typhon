//! Append-only vector stored in fixed-size chunks.
//!
//! Large node tables (AST, types) grow to hundreds of thousands of entries. A plain `Vec`
//! doubles and copies, and the abandoned buffers stay resident in the allocator, so the peak
//! is about twice the live size. Fixed chunks never move: the peak equals the live size and
//! no element is copied on growth. Chunks are large (64K elements, several MB) so the
//! allocator maps them directly: pages are resident only once written, and freeing a chunk
//! returns its memory to the OS.

use std::ops::{Index, IndexMut};

const SHIFT: usize = 20;
const CHUNK: usize = 1 << SHIFT;
const MASK: usize = CHUNK - 1;

pub struct ChunkVec<T> {
    chunks: Vec<Vec<T>>,
    len: usize,
}

impl<T> Default for ChunkVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ChunkVec<T> {
    pub const fn new() -> Self {
        Self { chunks: Vec::new(), len: 0 }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Append and return the index of the new element.
    pub fn push(&mut self, v: T) -> usize {
        if self.len & MASK == 0 {
            self.chunks.push(Vec::with_capacity(CHUNK));
        }
        self.chunks.last_mut().unwrap().push(v);
        self.len += 1;
        self.len - 1
    }
    pub fn get(&self, i: usize) -> Option<&T> {
        self.chunks.get(i >> SHIFT).and_then(|c| c.get(i & MASK))
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().flat_map(|c| c.iter())
    }
}

impl<T> Index<usize> for ChunkVec<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.chunks[i >> SHIFT][i & MASK]
    }
}

impl<T> IndexMut<usize> for ChunkVec<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.chunks[i >> SHIFT][i & MASK]
    }
}

#[cfg(test)]
mod tests {
    use super::ChunkVec;

    #[test]
    fn push_index_iter() {
        let mut v = ChunkVec::new();
        for i in 0..10_000u32 {
            assert_eq!(v.push(i), i as usize);
        }
        assert_eq!(v.len(), 10_000);
        assert_eq!(v[4097], 4097);
        v[5000] = 7;
        assert_eq!(v.iter().filter(|&&x| x == 7).count(), 2);
        assert!(v.get(10_000).is_none());
    }
}
