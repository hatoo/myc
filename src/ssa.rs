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
            Instruction::Cast { src, dst } => Some(dst.var().clone()),
            Instruction::Unary { op, src, dst } => Some(dst.var().clone()),
            Instruction::Binary { op, lhs, rhs, dst } => Some(dst.var().clone()),
            Instruction::Copy { src, dst } => Some(dst.var().clone()),
            Instruction::GetAddress { src, dst } => Some(dst.var().clone()),
            Instruction::Load { src, dst } => Some(dst.var().clone()),
            Instruction::Store { src, dst } => Some(dst.var().clone()),
            Instruction::Jump(_) => None,
            Instruction::JumpIfZero { src, dst } => None,
            Instruction::JumpIfNotZero { src, dst } => None,
            Instruction::Label(_) => None,
            Instruction::FunCall { callee, args, dst } => dst.clone().map(|dst| dst.var().clone()),
            Instruction::AddPtr {
                ptr,
                index,
                scale,
                dst,
            } => Some(dst.var().clone()),
            Instruction::CopyToOffset { src, dst, offset } => Some(dst.var().clone()),
            Instruction::CopyFromOffset { src, offset, dst } => Some(dst.var().clone()),
        }
    }
}

pub struct Ssa<I> {
    cfg: Cfg<I>,
    dominates: HashMap<usize, HashSet<usize>>,
    dominate_frontiers: HashMap<usize, HashSet<usize>>,
    phi: HashMap<usize, HashMap<usize, EcoString>>,
}

impl<I> Ssa<I> {
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
        let defs: HashMap<EcoString, HashSet<usize>> = HashMap::new();

        /*
        for (node, block) in &self.cfg.nodes {
            for inst in &block.instructions {
                if let Instruction::Copy {
                    dst: Val::Var(dst), ..
                } = inst
                {
                    defs.entry(dst.clone()).or_default().insert(*node);
                }
            }
        }
        */

        defs
    }

    fn add_phi(&mut self) {}
}
