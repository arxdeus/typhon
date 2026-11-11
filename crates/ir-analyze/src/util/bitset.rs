//! Small fixed-size bit set (`llvm::BitVector` / the `BitSet` helper in reaching.cpp).

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
}

const B: usize = 64;

impl BitSet {
    pub fn new(size: usize) -> Self {
        Self { words: vec![0; size.div_ceil(B)] }
    }
    pub fn set(&mut self, bit: usize) {
        self.words[bit / B] |= 1u64 << (bit % B);
    }
    pub fn get(&self, bit: usize) -> bool {
        self.words.get(bit / B).map(|w| w & (1u64 << (bit % B)) != 0).unwrap_or(false)
    }
    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }
    /// Sets bits `0..size`.
    pub fn set_all(&mut self, size: usize) {
        for i in 0..size {
            self.set(i);
        }
    }
    pub fn intersect(&mut self, other: &BitSet) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
    }
    /// `update`: union.
    pub fn union(&mut self, other: &BitSet) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
    }
    /// `out = (in & ~kill) | gen`; returns whether `self` changed.
    pub fn apply_transfer(&mut self, input: &BitSet, kill: &BitSet, gen_: &BitSet) -> bool {
        let mut changed = false;
        for i in 0..self.words.len() {
            let next = (input.words[i] & !kill.words[i]) | gen_.words[i];
            changed |= self.words[i] != next;
            self.words[i] = next;
        }
        changed
    }
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, w)| {
            let w = *w;
            (0..B).filter(move |b| w & (1u64 << b) != 0).map(move |b| wi * B + b)
        })
    }
}
