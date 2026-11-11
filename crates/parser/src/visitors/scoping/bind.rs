//! Port of `ScopingVisitor::visitName` / `findDominatingBinding` (scoping.cpp).

use typhon_common::ErrorKind;

use super::{Item, Node, ScopingVisitor};
use crate::ast::{Attr, Binding, CaptureType};

impl ScopingVisitor<'_> {
    /// `visitName(name, adding, root, src)`. Can error!
    pub(crate) fn visit_name(
        &mut self,
        name: &str,
        adding: bool,
        root: Option<Node>,
        src: &typhon_common::srcinfo::SrcInfo,
    ) -> bool {
        if adding && self.ctx.in_class {
            return false;
        }
        if adding {
            if let Some(p) = self.ctx.captures.get(name).copied() {
                if p == CaptureType::Read {
                    let fs = self.ctx.first_seen.entry(name.to_string()).or_default().clone();
                    self.add_error(ErrorKind::ASSIGN_LOCAL_REFERENCE, &fs, &[&name, src]);
                    return false;
                } else if root.is_some() {
                    // global, nonlocal
                    self.switch_to_update(root, name, false);
                }
            } else {
                if let Some(i) = self.ctx.child_captures.get(name).copied()
                    && i != CaptureType::Global
                    && self.ctx.function_scope.is_some()
                {
                    let new_scope = vec![self.ctx.scope[0].id];
                    let suite = self.ctx.scope[0].suite;
                    typhon_common::seqassert!(suite.is_some(), "invalid suite");
                    if let Some(suite) = suite {
                        self.node_bindings_mut(Node::Stmt(suite)).bindings.insert(
                            name.to_string(),
                            Binding { name: name.to_string(), count: 0, is_nonlocal: i == CaptureType::Nonlocal },
                        );
                    }
                    let item = Item::new(src.clone(), new_scope, None, vec![]);
                    self.ctx.map.entry(name.to_string()).or_default().push(item);
                }
                let scope = self.ctx.get_scope();
                self.ctx.map.entry(name.to_string()).or_default().insert(0, Item::new(src.clone(), scope, root, vec![]));
            }
        } else {
            self.ctx.first_seen.entry(name.to_string()).or_insert_with(|| src.clone());
            if !self.ctx.map.contains_key(name) {
                self.ctx.captures.insert(name.to_string(), CaptureType::Read);
            }
        }
        if let Some(vi) = self.find_dominating_binding(name, true) {
            // Track loop variables to dominate them later. Example:
            // x = 1
            // while True:
            //   if x > 10: break
            //   x = x + 1  # x must be dominated after the loop to ensure that it gets updated
            let val_scope = self.ctx.map[name][vi].scope.clone();
            let mut scope = self.ctx.get_scope();
            for li in (0..self.ctx.scope.len()).rev() {
                if self.ctx.scope[li].seen_vars.is_some() {
                    let inside = !scope.is_empty()
                        && val_scope.len() >= scope.len()
                        && val_scope[scope.len() - 1] == *scope.last().unwrap();
                    if !inside {
                        self.ctx.scope[li].seen_vars.as_mut().unwrap().insert(name.to_string());
                    } else {
                        break;
                    }
                }
                scope.pop();
            }

            // Variable binding check for variables that are defined within conditional blocks
            let val = &self.ctx.map[name][vi];
            if !val.access_checked.is_empty() {
                let checked = val.access_checked.iter().rev().any(|a| {
                    !a.is_empty() && a.len() <= self.ctx.scope.len() && a[a.len() - 1] == self.ctx.scope[a.len() - 1].id
                });
                if !checked {
                    typhon_common::seqassert!(!adding, "bad branch");
                    let binding = val.binding;
                    if !binding.is_some_and(|b| self.node_has(b, Attr::Bindings)) {
                        // If the expression is not conditional, we can just do the check once
                        let s = self.ctx.get_scope();
                        self.ctx.map.get_mut(name).unwrap()[vi].access_checked.push(s);
                    }
                    return true;
                }
            }
        }
        false
    }

    /// `findDominatingBinding`: get an item from the context. Perform domination analysis
    /// for accessing items defined in the conditional blocks (i.e., Python scoping).
    /// Returns the index of the dominating item in `ctx.map[name]`.
    pub(crate) fn find_dominating_binding(&mut self, name: &str, allow_shadow: bool) -> Option<usize> {
        let items = self.ctx.map.get(name)?;
        if items.is_empty() {
            return None;
        }
        let n = items.len();
        let mut last_good = items.iter().position(|i| !i.ignore).unwrap_or(n);
        let cs_len = self.ctx.scope.len();
        let mut common_scope = cs_len;
        // Iterate through all bindings with the given name and find the closest binding that
        // dominates the current scope.
        for (idx, i) in items.iter().enumerate() {
            if i.ignore {
                continue;
            }
            let complete = i.scope.len() <= cs_len && *i.scope.last().unwrap() == self.ctx.scope[i.scope.len() - 1].id;
            if complete {
                common_scope = i.scope.len();
                last_good = idx;
                break;
            } else {
                typhon_common::seqassert!(i.scope[0] == 0, "bad scoping");
                typhon_common::seqassert!(self.ctx.scope[0].id == 0, "bad scoping");
                // Find the longest block prefix between the binding and the current common scope.
                common_scope = common_scope.min(i.scope.len());
                while common_scope > 0 && i.scope[common_scope - 1] != self.ctx.scope[common_scope - 1].id {
                    common_scope -= 1;
                }
                last_good = idx;
            }
        }
        typhon_common::seqassert!(last_good != n, "corrupted scoping ({})", name);
        if last_good == n {
            return None;
        }
        if !allow_shadow {
            // go to the end
            last_good = n - 1;
            let lg = &items[last_good];
            let mut p = common_scope.min(lg.scope.len());
            // PORT-NOTE: C++ reads `scope[p - 1]` with p == 0 (UB); stop at 0 instead.
            while p > 0 && lg.scope[p - 1] != self.ctx.scope[p - 1].id {
                p -= 1;
            }
            common_scope = p;
        }

        let mut got_used_var = false;
        let lg_scope = items[last_good].scope.clone();
        let lg_binding = items[last_good].binding;
        if lg_scope.len() != common_scope {
            // The current scope is potentially reachable by multiple bindings that are
            // not dominated by a common binding. Create such binding in the scope that
            // dominates (covers) all of them.
            let new_scope: Vec<i32> = self.ctx.get_scope()[..common_scope].to_vec();
            // Make sure to prepend a binding declaration: `var` and `var__used__ = False`
            // to the dominating scope.
            // PORT-NOTE: the C++ loop `for (si = cs; si-- > 0; si--)` steps by two; kept.
            let mut si = common_scope as isize - 1;
            while si >= 0 {
                let Some(suite) = self.ctx.scope[si as usize].suite else {
                    si -= 2;
                    continue;
                };
                self.node_bindings_mut(Node::Stmt(suite))
                    .bindings
                    .insert(name.to_string(), Binding { name: name.to_string(), count: 1, is_nonlocal: false });
                let item = Item::new(self.src.clone(), new_scope, Some(Node::Stmt(suite)), vec![lg_scope.clone()]);
                self.ctx.map.get_mut(name).unwrap().insert(last_good + 1, item);
                last_good += 1;
                got_used_var = true;
                break;
            }
        } else if let Some(b) = lg_binding
            && self.node_has(b, Attr::Bindings)
        {
            let bs = self.node_bindings_mut(b);
            got_used_var = bs.bindings.entry(name.to_string()).or_default().count > 0;
        }
        // Remove all bindings after the dominant binding.
        let final_scope = self.ctx.map[name][last_good].scope.clone();
        for i in 0..last_good {
            let b = self.ctx.map[name][i].binding;
            self.switch_to_update(b, name, got_used_var);
            let it = &mut self.ctx.map.get_mut(name).unwrap()[i];
            it.scope = final_scope.clone();
            it.ignore = true;
        }
        let lgb = self.ctx.map[name][last_good].binding;
        if !got_used_var
            && let Some(b) = lgb
            && self.node_has(b, Attr::Bindings)
        {
            self.node_bindings_mut(b)
                .bindings
                .insert(name.to_string(), Binding { name: name.to_string(), count: 0, is_nonlocal: false });
        }
        Some(last_good)
    }
}
