use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    control_flow::{Cfg, NodeId},
    tacky::Instruction,
};

pub trait MayHasDst {
    fn dst(&self) -> Option<EcoString>;
}

impl MayHasDst for Instruction {
    fn dst(&self) -> Option<EcoString> {
        match self {
            Instruction::Nop => None,
            Instruction::Return(_) => None,
            Instruction::Cast { dst, .. } => Some(dst.var().clone()),
            Instruction::Unary { dst, .. } => Some(dst.var().clone()),
            Instruction::Binary { dst, .. } => Some(dst.var().clone()),
            Instruction::Copy { dst, .. } => Some(dst.var().clone()),
            Instruction::GetAddress { dst, .. } => Some(dst.var().clone()),
            Instruction::Load { dst, .. } => Some(dst.var().clone()),
            Instruction::Store { dst, .. } => Some(dst.var().clone()),
            Instruction::Jump(_) => None,
            Instruction::JumpIfZero { .. } => None,
            Instruction::JumpIfNotZero { .. } => None,
            Instruction::Label(_) => None,
            Instruction::FunCall { dst, .. } => dst.clone().map(|dst| dst.var().clone()),
            Instruction::AddPtr { dst, .. } => Some(dst.var().clone()),
            Instruction::CopyToOffset { dst, .. } => Some(dst.var().clone()),
            Instruction::CopyFromOffset { dst, .. } => Some(dst.var().clone()),
        }
    }
}

pub struct Ssa<I> {
    cfg: Cfg<I>,
    dominates: HashMap<usize, HashSet<usize>>,
    dominate_frontiers: HashMap<usize, HashSet<usize>>,
    phi: HashMap<usize, HashMap<usize, EcoString>>,
}

impl<I: MayHasDst> Ssa<I> {
    pub fn new(cfg: Cfg<I>) -> Self {
        let mut me = Ssa {
            cfg,
            dominates: HashMap::new(),
            dominate_frontiers: HashMap::new(),
            phi: HashMap::new(),
        };
        me.compute_dominates();
        me.compute_dominate_frontiers();
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

    fn compute_dominate_frontiers(&mut self) {
        for (n, dom) in &self.dominates {
            let mut one_step = HashSet::new();

            for d in dom {
                for next in &self.cfg.nodes[d].successors {
                    if let NodeId::Block(next) = next {
                        one_step.insert(*next);
                    }
                }
            }

            let domf = one_step.difference(&one_step).cloned().collect();
            self.dominate_frontiers.insert(*n, domf);
        }
    }

    fn defs(&self) -> HashMap<EcoString, HashSet<usize>> {
        let mut defs: HashMap<EcoString, HashSet<usize>> = HashMap::new();

        for (node, block) in &self.cfg.nodes {
            for inst in &block.instructions {
                if let Some(dst) = inst.dst() {
                    defs.entry(dst).or_default().insert(*node);
                }
            }
        }

        defs
    }

    fn add_phi(&mut self) {
        let mut defs = self.defs();
        let vars = defs.keys().cloned().collect::<Vec<_>>();

        for v in vars {
            for d in defs[&v].clone() {
                for block in &self.dominate_frontiers[&d] {
                    self.phi.entry(*block).or_default().insert(d, v.clone());
                    defs.entry(v.clone()).or_default().insert(*block);
                }
            }
        }
    }
}
