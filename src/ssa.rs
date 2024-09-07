use std::collections::{HashMap, HashSet};

use crate::control_flow::{Cfg, NodeId};

pub struct Ssa<I> {
    cfg: Cfg<I>,
    dominates: HashMap<usize, HashSet<usize>>,
}

impl<I> Ssa<I> {
    pub fn new(cfg: Cfg<I>) -> Self {
        let mut me = Ssa {
            cfg,
            dominates: HashMap::new(),
        };
        me.compute_dominates();
        me
    }

    fn compute_dominates(&mut self) {
        let all_nodes = self.cfg.nodes.keys().copied().collect::<HashSet<_>>();

        for &node in &all_nodes {
            self.dominates.insert(node, all_nodes.clone());
        }

        loop {
            let mut changed = false;
            for &node in &all_nodes {
                let mut new_dom = HashSet::new();

                for (i, &pred) in self.cfg.nodes[&node]
                    .predecessors
                    .iter()
                    .filter_map(|id| {
                        if let NodeId::Block(id) = id {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .enumerate()
                {
                    if i == 0 {
                        new_dom = self.dominates[&pred].clone();
                    } else {
                        new_dom = new_dom
                            .intersection(&self.dominates[&pred])
                            .copied()
                            .collect();
                    }
                }

                new_dom.insert(node);

                if new_dom != self.dominates[&node] {
                    self.dominates.insert(node, new_dom);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
    }
}
