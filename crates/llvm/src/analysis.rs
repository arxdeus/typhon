//! Minimal CFG analyses over raw LLVM functions.
//!
//! PORT-NOTE: the C++ passes query `DominatorTree`, `PostDominatorTree`,
//! `LoopInfo` and `CycleInfo` from the new pass manager. The LLVM C API exposes
//! none of these, so they are recomputed here (Cooper/Harvey/Kennedy dominators,
//! natural loops from back edges, Tarjan SCCs for irreducible cycles).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ffi::{self, BB, V};

/// Control-flow graph of one function, blocks numbered in layout order.
pub struct Cfg {
    pub blocks: Vec<BB>,
    pub index: FxHashMap<BB, usize>,
    pub succs: Vec<Vec<usize>>,
    pub preds: Vec<Vec<usize>>,
}

impl Cfg {
    pub fn new(f: V) -> Self {
        let blocks = ffi::blocks(f);
        let index: FxHashMap<BB, usize> = blocks.iter().enumerate().map(|(i, &b)| (b, i)).collect();
        let mut succs = vec![Vec::new(); blocks.len()];
        let mut preds = vec![Vec::new(); blocks.len()];
        for (i, &b) in blocks.iter().enumerate() {
            for s in ffi::successors(b) {
                if let Some(&j) = index.get(&s) {
                    if !succs[i].contains(&j) {
                        succs[i].push(j);
                        preds[j].push(i);
                    }
                }
            }
        }
        Cfg { blocks, index, succs, preds }
    }

    pub fn id(&self, b: BB) -> usize {
        self.index[&b]
    }
}

/// Generic dominator tree over a graph given by `succ`/`pred` lists with a root.
#[derive(Clone)]
pub struct DomTree {
    /// Immediate dominator per node (`None` for root / unreachable / virtual).
    idom: Vec<Option<usize>>,
    reachable: Vec<bool>,
    depth: Vec<usize>,
}

impl DomTree {
    fn compute(n: usize, roots: &[usize], succ: &dyn Fn(usize) -> Vec<usize>, pred: &dyn Fn(usize) -> Vec<usize>) -> Self {
        // reverse postorder from roots
        let mut visited = vec![false; n];
        let mut post = Vec::with_capacity(n);
        for &r in roots {
            if visited[r] {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = vec![(r, 0)];
            visited[r] = true;
            while let Some(&mut (node, ref mut k)) = stack.last_mut() {
                let ss = succ(node);
                if *k < ss.len() {
                    let s = ss[*k];
                    *k += 1;
                    if !visited[s] {
                        visited[s] = true;
                        stack.push((s, 0));
                    }
                } else {
                    post.push(node);
                    stack.pop();
                }
            }
        }
        let mut order = vec![usize::MAX; n];
        for (i, &b) in post.iter().enumerate() {
            order[b] = i;
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        let root_set: FxHashSet<usize> = roots.iter().copied().collect();
        for &r in roots {
            idom[r] = Some(r);
        }
        let intersect = |idom: &Vec<Option<usize>>, mut a: usize, mut b: usize| -> Option<usize> {
            while a != b {
                while order[a] < order[b] {
                    let next = idom[a]?;
                    if next == a {
                        return None;
                    }
                    a = next;
                }
                while order[b] < order[a] {
                    let next = idom[b]?;
                    if next == b {
                        return None;
                    }
                    b = next;
                }
            }
            Some(a)
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &b in post.iter().rev() {
                if root_set.contains(&b) {
                    continue;
                }
                let mut new: Option<usize> = None;
                let mut failed = false;
                for p in pred(b) {
                    if !visited[p] || idom[p].is_none() {
                        continue;
                    }
                    new = match new {
                        None => Some(p),
                        Some(cur) => match intersect(&idom, p, cur) {
                            Some(x) => Some(x),
                            None => {
                                failed = true;
                                None
                            }
                        },
                    };
                    if failed {
                        break;
                    }
                }
                // Multiple roots with no common dominator: treat as virtual root.
                if failed {
                    if idom[b] != Some(b) {
                        idom[b] = Some(b);
                        changed = true;
                    }
                    continue;
                }
                if new.is_some() && idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let mut depth = vec![0usize; n];
        for &b in post.iter().rev() {
            if let Some(d) = idom[b] {
                if d != b {
                    depth[b] = depth[d] + 1;
                }
            }
        }
        DomTree { idom, reachable: visited, depth }
    }

    /// Dominators of a function CFG rooted at the entry block.
    pub fn dominators(cfg: &Cfg) -> Self {
        if cfg.blocks.is_empty() {
            return DomTree { idom: vec![], reachable: vec![], depth: vec![] };
        }
        Self::compute(cfg.blocks.len(), &[0], &|b| cfg.succs[b].clone(), &|b| cfg.preds[b].clone())
    }

    /// Post-dominators (roots = blocks without successors).
    pub fn post_dominators(cfg: &Cfg) -> Self {
        let n = cfg.blocks.len();
        let mut roots: Vec<usize> = (0..n).filter(|&b| cfg.succs[b].is_empty()).collect();
        // Blocks in infinite loops never reach an exit: add them as roots like LLVM does
        // (LLVM picks "furthest away" nodes; we use any unvisited node in reverse order).
        let mut reach = vec![false; n];
        let mut stack = roots.clone();
        for &r in &roots {
            reach[r] = true;
        }
        while let Some(b) = stack.pop() {
            for &p in &cfg.preds[b] {
                if !reach[p] {
                    reach[p] = true;
                    stack.push(p);
                }
            }
        }
        for b in (0..n).rev() {
            if !reach[b] {
                roots.push(b);
                let mut st = vec![b];
                reach[b] = true;
                while let Some(x) = st.pop() {
                    for &p in &cfg.preds[x] {
                        if !reach[p] {
                            reach[p] = true;
                            st.push(p);
                        }
                    }
                }
            }
        }
        Self::compute(n, &roots, &|b| cfg.preds[b].clone(), &|b| cfg.succs[b].clone())
    }

    pub fn is_reachable(&self, b: usize) -> bool {
        self.reachable.get(b).copied().unwrap_or(false)
    }

    fn parent(&self, b: usize) -> Option<usize> {
        match self.idom[b] {
            Some(d) if d != b => Some(d),
            _ => None,
        }
    }

    /// Does `a` dominate `b` (reflexive)?
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        if !self.is_reachable(b) {
            return true;
        }
        if !self.is_reachable(a) {
            return false;
        }
        let mut x = b;
        loop {
            if x == a {
                return true;
            }
            match self.parent(x) {
                Some(p) => x = p,
                None => return false,
            }
        }
    }

    /// Nearest common dominator, `None` if only a virtual root dominates both.
    pub fn nearest_common(&self, mut a: usize, mut b: usize) -> Option<usize> {
        if !self.is_reachable(a) || !self.is_reachable(b) {
            return None;
        }
        while self.depth[a] > self.depth[b] {
            a = self.parent(a)?;
        }
        while self.depth[b] > self.depth[a] {
            b = self.parent(b)?;
        }
        while a != b {
            a = self.parent(a)?;
            b = self.parent(b)?;
        }
        Some(a)
    }
}

/// `Instruction` dominance within a function (block dominance + intra-block order).
#[allow(dead_code)]
pub fn inst_dominates(dt: &DomTree, cfg: &Cfg, a: V, b: V) -> bool {
    let (ba, bb) = (cfg.id(ffi::parent_block(a)), cfg.id(ffi::parent_block(b)));
    if ba != bb {
        return dt.dominates(ba, bb);
    }
    for i in ffi::instructions(ffi::parent_block(a)) {
        if i == a {
            return true;
        }
        if i == b {
            return false;
        }
    }
    false
}

/// A natural loop.
#[derive(Clone, Debug)]
pub struct Loop {
    pub header: usize,
    pub blocks: Vec<usize>,
    pub block_set: FxHashSet<usize>,
    pub parent: Option<usize>,
    pub subloops: Vec<usize>,
    pub depth: usize,
}

impl Loop {
    pub fn contains(&self, b: usize) -> bool {
        self.block_set.contains(&b)
    }
}

/// Port of the subset of `LoopInfo` used by Codon's passes.
pub struct LoopInfo {
    pub loops: Vec<Loop>,
    /// Innermost loop of each block.
    pub block_loop: Vec<Option<usize>>,
}

impl LoopInfo {
    pub fn new(cfg: &Cfg, dt: &DomTree) -> Self {
        let n = cfg.blocks.len();
        let mut by_header: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
        let mut headers_order = Vec::new();
        for b in 0..n {
            if !dt.is_reachable(b) {
                continue;
            }
            for &s in &cfg.succs[b] {
                if dt.dominates(s, b) {
                    // back edge b -> s
                    let set = by_header.entry(s).or_insert_with(|| {
                        headers_order.push(s);
                        let mut h = FxHashSet::default();
                        h.insert(s);
                        h
                    });
                    let mut stack = vec![b];
                    while let Some(x) = stack.pop() {
                        if set.insert(x) {
                            for &p in &cfg.preds[x] {
                                if dt.is_reachable(p) {
                                    stack.push(p);
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut loops: Vec<Loop> = headers_order
            .into_iter()
            .map(|h| {
                let set = by_header.remove(&h).unwrap_or_default();
                let mut blocks: Vec<usize> = set.iter().copied().collect();
                blocks.sort_unstable();
                // header first, like LLVM
                blocks.retain(|&x| x != h);
                blocks.insert(0, h);
                Loop { header: h, blocks, block_set: set, parent: None, subloops: vec![], depth: 1 }
            })
            .collect();
        // nesting: parent = smallest strictly-containing loop
        let count = loops.len();
        for i in 0..count {
            let mut best: Option<usize> = None;
            for j in 0..count {
                if i == j {
                    continue;
                }
                if loops[j].block_set.len() > loops[i].block_set.len() && loops[j].contains(loops[i].header) {
                    if best.is_none_or(|bj| loops[j].block_set.len() < loops[bj].block_set.len()) {
                        best = Some(j);
                    }
                }
            }
            loops[i].parent = best;
        }
        for i in 0..count {
            if let Some(p) = loops[i].parent {
                loops[p].subloops.push(i);
            }
        }
        for i in 0..count {
            let mut d = 1;
            let mut p = loops[i].parent;
            while let Some(x) = p {
                d += 1;
                p = loops[x].parent;
            }
            loops[i].depth = d;
        }
        let mut block_loop: Vec<Option<usize>> = vec![None; n];
        for (i, l) in loops.iter().enumerate() {
            for &b in &l.blocks {
                match block_loop[b] {
                    Some(cur) if loops[cur].block_set.len() <= l.block_set.len() => {}
                    _ => block_loop[b] = Some(i),
                }
            }
        }
        LoopInfo { loops, block_loop }
    }

    pub fn loop_for(&self, b: usize) -> Option<usize> {
        self.block_loop.get(b).copied().flatten()
    }

    /// `Loop::getLoopPreheader()`.
    pub fn preheader(&self, cfg: &Cfg, l: usize) -> Option<usize> {
        let lp = &self.loops[l];
        let outside: Vec<usize> = cfg.preds[lp.header].iter().copied().filter(|p| !lp.contains(*p)).collect();
        if outside.len() != 1 {
            return None;
        }
        let p = outside[0];
        (cfg.succs[p].len() == 1).then_some(p)
    }

    /// `Loop::getUniqueExitBlocks()`.
    pub fn exit_blocks(&self, cfg: &Cfg, l: usize) -> Vec<usize> {
        let lp = &self.loops[l];
        let mut out = Vec::new();
        for &b in &lp.blocks {
            for &s in &cfg.succs[b] {
                if !lp.contains(s) && !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        out
    }

    /// `Loop::getExitBlock()`: the unique exit block or none.
    pub fn exit_block(&self, cfg: &Cfg, l: usize) -> Option<usize> {
        let e = self.exit_blocks(cfg, l);
        let lp = &self.loops[l];
        // LLVM's getExitBlock requires a single exit edge target (may have multiple edges to it).
        if e.len() == 1 {
            let _ = lp;
            Some(e[0])
        } else {
            None
        }
    }

    /// `Loop::getExitingBlock()`: unique block with an edge leaving the loop.
    pub fn exiting_block(&self, cfg: &Cfg, l: usize) -> Option<usize> {
        let lp = &self.loops[l];
        let ex: Vec<usize> =
            lp.blocks.iter().copied().filter(|&b| cfg.succs[b].iter().any(|s| !lp.contains(*s))).collect();
        (ex.len() == 1).then(|| ex[0])
    }

    /// `Loop::hasDedicatedExits()`.
    pub fn has_dedicated_exits(&self, cfg: &Cfg, l: usize) -> bool {
        let lp = &self.loops[l];
        self.exit_blocks(cfg, l).into_iter().all(|e| cfg.preds[e].iter().all(|p| lp.contains(*p)))
    }

    /// Loops in the order `appendLoopsToWorklist` + `pop_back_val` processes them:
    /// innermost first (post-order over the loop forest).
    pub fn worklist_order(&self) -> Vec<usize> {
        let mut out = Vec::new();
        fn visit(li: &LoopInfo, l: usize, out: &mut Vec<usize>) {
            for &s in &li.loops[l].subloops {
                visit(li, s, out);
            }
            out.push(l);
        }
        for (i, l) in self.loops.iter().enumerate() {
            if l.parent.is_none() {
                visit(self, i, &mut out);
            }
        }
        out
    }
}

/// Blocks that belong to an irreducible cycle (or to a loop nested in one).
///
/// PORT-NOTE: approximates `CycleInfo` + `Cycle::isReducible()` walking parent
/// cycles: remove natural back edges, any remaining non-trivial SCC is an
/// irreducible cycle; blocks of loops whose header lies in such an SCC are
/// considered to be inside it too.
pub fn irreducible_blocks(cfg: &Cfg, dt: &DomTree, li: &LoopInfo) -> FxHashSet<usize> {
    let n = cfg.blocks.len();
    let succ: Vec<Vec<usize>> = (0..n)
        .map(|b| cfg.succs[b].iter().copied().filter(|&s| !(dt.is_reachable(b) && dt.dominates(s, b))).collect())
        .collect();
    // Tarjan
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on = vec![false; n];
    let mut stack = Vec::new();
    let mut counter = 0;
    let mut result = FxHashSet::default();
    for start in 0..n {
        if index[start] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(start, 0)];
        index[start] = counter;
        low[start] = counter;
        counter += 1;
        stack.push(start);
        on[start] = true;
        while let Some(&mut (v, ref mut k)) = call.last_mut() {
            if *k < succ[v].len() {
                let w = succ[v][*k];
                *k += 1;
                if index[w] == usize::MAX {
                    index[w] = counter;
                    low[w] = counter;
                    counter += 1;
                    stack.push(w);
                    on[w] = true;
                    call.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some(&(p, _)) = call.last() {
                    low[p] = low[p].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut comp = Vec::new();
                    loop {
                        let w = stack.pop().unwrap_or(v);
                        on[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    if comp.len() > 1 {
                        result.extend(comp);
                    }
                }
            }
        }
    }
    if !result.is_empty() {
        for l in &li.loops {
            if result.contains(&l.header) {
                result.extend(l.blocks.iter().copied());
            }
        }
    }
    result
}
