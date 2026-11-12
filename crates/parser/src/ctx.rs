//! Generic scoped identifier table `Context<T>`.
//!
//! Note: the reference implementation keeps `BindingCheckpoint`s in a thread-local map keyed by the context
//! pointer and lets each checkpoint call a `saveState` callback. Here checkpoints are a
//! stack owned by the context itself; the owner (e.g. `TypeContext`) snapshots its own
//! extra state when [`Context::prepare_checkpoint`] reports newly prepared checkpoints
//! (see `TypeContext::prepare_checkpoint`).

use std::any::Any;
use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use typhon_common::SrcInfo;

/// Stack of bindings for one identifier (front = top-most).
pub type BindingStack<I> = SmallVec<[I; 2]>;

/// One speculative-realization checkpoint (`Context::BindingCheckpoint`).
pub struct BindingCheckpoint<I> {
    /// Saved binding stacks (`None` = identifier did not exist).
    pub bindings: FxHashMap<String, Option<BindingStack<I>>>,
    /// Set once the first change was recorded.
    pub prepared: bool,
    /// Whether the owner wants a state snapshot on preparation.
    pub wants_state: bool,
    /// Owner-defined state snapshot taken at preparation time.
    pub state: Option<Box<dyn Any>>,
}

/// A variable table (transformation context): identifiers and their block hierarchy.
pub struct Context<I: Clone> {
    /// Identifier -> stack of objects sharing it (front = current block).
    pub map: FxHashMap<String, BindingStack<I>>,
    /// Stack of blocks and their identifiers (front = current block).
    pub stack: VecDeque<Vec<String>>,
    /// The absolute path of the current module.
    filename: String,
    /// Source-info stack (`nodeStack`), used for `getSrcInfo()`.
    node_stack: Vec<SrcInfo>,
    /// Active checkpoints (innermost last).
    checkpoints: Vec<BindingCheckpoint<I>>,
}

impl<I: Clone> Clone for Context<I> {
    /// Copy constructor (checkpoints are not copied: they are bound to the original).
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            stack: self.stack.clone(),
            filename: self.filename.clone(),
            node_stack: self.node_stack.clone(),
            checkpoints: Vec::new(),
        }
    }
}

impl<I: Clone> Context<I> {
    pub fn new(filename: impl Into<String>) -> Self {
        let mut stack = VecDeque::new();
        // Add a top-level block to the stack.
        stack.push_front(Vec::new());
        Self { map: FxHashMap::default(), stack, filename: filename.into(), node_stack: Vec::new(), checkpoints: Vec::new() }
    }

    // ------------------------------------------------------------------
    // checkpoints
    // ------------------------------------------------------------------

    /// Open a new checkpoint (`BindingCheckpoint` constructor). Returns its index.
    pub fn push_checkpoint(&mut self, wants_state: bool) -> usize {
        self.checkpoints.push(BindingCheckpoint {
            bindings: FxHashMap::default(),
            prepared: false,
            wants_state,
            state: None,
        });
        self.checkpoints.len() - 1
    }
    /// Close the innermost checkpoint (`~BindingCheckpoint`).
    pub fn pop_checkpoint(&mut self) -> Option<BindingCheckpoint<I>> {
        self.checkpoints.pop()
    }
    pub fn has_checkpoints(&self) -> bool {
        !self.checkpoints.is_empty()
    }
    pub fn checkpoint_mut(&mut self, idx: usize) -> Option<&mut BindingCheckpoint<I>> {
        self.checkpoints.get_mut(idx)
    }
    /// `prepareCheckpoint()`: mark all checkpoints prepared. Returns the indices of
    /// checkpoints that just became prepared and want a state snapshot (the owner must fill
    /// `state` *before* mutating anything).
    pub fn prepare_checkpoint(&mut self) -> Vec<usize> {
        let mut r = vec![];
        for (i, c) in self.checkpoints.iter_mut().enumerate() {
            if !c.prepared {
                c.prepared = true;
                if c.wants_state {
                    r.push(i);
                }
            }
        }
        r
    }
    /// `recordBinding(name)`: remember the binding stack of `name` in every checkpoint.
    /// Callers must have called `prepare_checkpoint` (directly or through the owner).
    pub fn record_binding(&mut self, name: &str) {
        if self.checkpoints.is_empty() {
            return;
        }
        let current = self.map.get(name).cloned();
        for c in self.checkpoints.iter_mut().rev() {
            c.prepared = true;
            if c.bindings.contains_key(name) {
                continue;
            }
            c.bindings.insert(name.to_string(), current.clone());
        }
    }
    /// `BindingCheckpoint::rollback(preserve)`.
    pub fn rollback(&mut self, cp: &BindingCheckpoint<I>, preserve: impl Fn(&I) -> bool) {
        for (name, saved) in &cp.bindings {
            match saved {
                Some(s) => {
                    self.map.insert(name.clone(), s.clone());
                }
                None => {
                    let keep = self.map.get(name).and_then(|c| c.first().filter(|i| preserve(i)).cloned());
                    match keep {
                        Some(item) => {
                            let mut v = BindingStack::new();
                            v.push(item);
                            self.map.insert(name.clone(), v);
                        }
                        None => {
                            self.map.remove(name);
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // bindings
    // ------------------------------------------------------------------

    /// `add(name, var)`: add an object to the top of the stack.
    pub fn add(&mut self, name: &str, var: I) {
        typhon_common::seqassert!(!name.is_empty(), "adding an empty identifier");
        self.record_binding(name);
        self.map.entry(name.to_string()).or_default().insert(0, var);
        self.stack.front_mut().unwrap().push(name.to_string());
    }
    /// `addToplevel`-style insertion without registering in the block.
    pub fn add_binding_only(&mut self, name: &str, var: I) {
        self.record_binding(name);
        self.map.entry(name.to_string()).or_default().insert(0, var);
    }
    /// `remove(name)`: remove the top-most object with a given identifier.
    pub fn remove(&mut self, name: &str) {
        self.remove_from_map(name);
        for s in self.stack.iter_mut() {
            if let Some(i) = s.iter().position(|x| x == name) {
                s.remove(i);
                return;
            }
        }
    }
    /// `find(name)`: top-most object with a given identifier.
    pub fn find(&self, name: &str) -> Option<I> {
        self.map.get(name).and_then(|s| s.first().cloned())
    }
    /// `find_all(name)`.
    pub fn find_all(&self, name: &str) -> Option<&BindingStack<I>> {
        self.map.get(name)
    }
    /// `addBlock()`.
    pub fn add_block(&mut self) {
        self.stack.push_front(Vec::new());
    }
    /// `popBlock()`: remove the top-most block and all variables it holds.
    pub fn pop_block(&mut self) {
        let names = self.stack.front().cloned().unwrap_or_default();
        for name in &names {
            self.remove_from_map(name);
        }
        self.stack.pop_front();
    }
    /// `getBlock()`.
    pub fn block(&self) -> &Vec<String> {
        self.stack.front().unwrap()
    }
    pub fn block_mut(&mut self) -> &mut Vec<String> {
        self.stack.front_mut().unwrap()
    }
    /// `removeFromTopStack(name)`.
    pub fn remove_from_top_stack(&mut self, name: &str) {
        let f = self.stack.front_mut().unwrap();
        if let Some(i) = f.iter().position(|x| x == name) {
            f.remove(i);
        }
    }
    /// `removeFromMap(name)`.
    pub fn remove_from_map(&mut self, name: &str) {
        if !self.map.contains_key(name) {
            return;
        }
        self.record_binding(name);
        let e = self.map.get_mut(name).unwrap();
        typhon_common::seqassert!(!e.is_empty(), "identifier {} not found in the map", name);
        if !e.is_empty() {
            e.remove(0);
        }
        if e.is_empty() {
            self.map.remove(name);
        }
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }
    pub fn set_filename(&mut self, f: impl Into<String>) {
        self.filename = f.into();
    }
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }

    /* SrcInfo helpers */
    pub fn push_node(&mut self, s: SrcInfo) {
        self.node_stack.push(s);
    }
    pub fn pop_node(&mut self) {
        self.node_stack.pop();
    }
    pub fn last_node(&self) -> Option<SrcInfo> {
        self.node_stack.last().copied()
    }
    pub fn parent_node(&self) -> Option<SrcInfo> {
        let n = self.node_stack.len();
        if n > 1 { Some(self.node_stack[n - 2]) } else { None }
    }
    /// `getSrcInfo()`.
    pub fn src_info(&self) -> SrcInfo {
        self.node_stack.last().copied().unwrap_or_default()
    }
}
