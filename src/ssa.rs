use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    control_flow::{Cfg, NodeId},
    tacky::{Instruction, Val},
};

pub trait SsaInstruction {
    fn dst(&mut self) -> Option<&mut EcoString>;
    fn map_args<F: FnMut(&mut EcoString)>(&mut self, f: F);
}

impl SsaInstruction for Instruction {
    fn dst(&mut self) -> Option<&mut EcoString> {
        fn f(dst: &mut Val) -> Option<&mut EcoString> {
            if let Val::Var(dst) = dst {
                Some(dst)
            } else {
                None
            }
        }

        match self {
            Instruction::Nop => None,
            Instruction::Return(_) => None,
            Instruction::Cast { dst, .. } => f(dst),
            Instruction::Unary { dst, .. } => f(dst),
            Instruction::Binary { dst, .. } => f(dst),
            Instruction::Copy { dst, .. } => f(dst),
            Instruction::GetAddress { dst, .. } => f(dst),
            Instruction::Load { dst, .. } => f(dst),
            Instruction::Store { dst, .. } => f(dst),
            Instruction::Jump(_) => None,
            Instruction::JumpIfZero { .. } => None,
            Instruction::JumpIfNotZero { .. } => None,
            Instruction::Label(_) => None,
            Instruction::FunCall { dst, .. } => dst.as_mut().and_then(f),
            Instruction::AddPtr { dst, .. } => f(dst),
            Instruction::CopyToOffset { dst, .. } => f(dst),
            Instruction::CopyFromOffset { dst, .. } => f(dst),
        }
    }

    fn map_args<F: FnMut(&mut EcoString)>(&mut self, mut f: F) {
        let mut apply = |var: &mut Val| {
            if let Val::Var(var) = var {
                f(var);
            }
        };

        match self {
            Instruction::Nop => {}
            Instruction::Return(val) => {
                if let Some(val) = val {
                    apply(val);
                }
            }
            Instruction::Cast { src, dst } => {
                apply(dst);
            }
            Instruction::Unary { op, src, dst } => {
                apply(dst);
            }
            Instruction::Binary { op, lhs, rhs, dst } => {
                apply(lhs);
                apply(rhs);
            }
            Instruction::Copy { src, dst } => {
                apply(src);
            }
            Instruction::GetAddress { src, dst } => {
                apply(src);
            }
            Instruction::Load { src, dst } => {
                apply(src);
            }
            Instruction::Store { src, dst } => {
                apply(src);
            }
            Instruction::Jump(_) => {}
            Instruction::JumpIfZero { src, dst } => {
                apply(src);
            }
            Instruction::JumpIfNotZero { src, dst } => {
                apply(src);
            }
            Instruction::Label(_) => {}
            Instruction::FunCall { callee, args, dst } => {
                apply(callee);
                for arg in args {
                    apply(arg);
                }
            }
            Instruction::AddPtr {
                ptr,
                index,
                scale,
                dst,
            } => {
                apply(ptr);
                apply(index);
            }
            Instruction::CopyToOffset { src, dst, offset } => {
                apply(src);
            }
            Instruction::CopyFromOffset { src, offset, dst } => {
                apply(src);
            }
        }
    }
}

#[derive(Debug)]
pub struct Ssa<I> {
    cfg: Cfg<I>,
    dominates: HashMap<usize, HashSet<usize>>,
    dominated: HashMap<usize, HashSet<usize>>,
    dominate_frontiers: HashMap<usize, HashSet<usize>>,
    phi: HashMap<usize, HashMap<EcoString, HashMap<usize, EcoString>>>,
}

impl<I: SsaInstruction> Ssa<I> {
    pub fn new(cfg: Cfg<I>) -> Self {
        let mut me = Ssa {
            cfg,
            dominates: HashMap::new(),
            dominated: HashMap::new(),
            dominate_frontiers: HashMap::new(),
            phi: HashMap::new(),
        };
        me.compute_dominators();
        me.compute_dominate_frontiers();
        me.add_phi();
        me
    }

    fn compute_dominators(&mut self) {
        let all_nodes = self.cfg.nodes.keys().copied().collect::<HashSet<_>>();

        let mut dominators = HashMap::new();
        for &node in &all_nodes {
            dominators.insert(node, all_nodes.clone());
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
                        new_dom = dominators[&pred].clone();
                    } else {
                        new_dom = new_dom.intersection(&dominators[&pred]).copied().collect();
                    }
                }

                new_dom.insert(node);

                if new_dom != dominators[&node] {
                    dominators.insert(node, new_dom);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
        let mut dominates: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (n, dom) in &dominators {
            for d in dom {
                dominates.entry(*d).or_default().insert(*n);
            }
        }

        self.dominated = dominators;
        self.dominates = dominates;
    }

    fn compute_dominate_frontiers(&mut self) {
        for (n, dom) in &self.dominates {
            let mut one_step = HashSet::new();

            for d in dom {
                for next in &self.cfg.nodes[&d].successors {
                    if let NodeId::Block(next) = next {
                        one_step.insert(*next);
                    }
                }
            }

            let domf = one_step.difference(&dom).cloned().collect();
            self.dominate_frontiers.insert(*n, domf);
        }
    }

    fn defs(&mut self) -> HashMap<EcoString, HashSet<usize>> {
        let mut defs: HashMap<EcoString, HashSet<usize>> = HashMap::new();

        for (node, block) in &mut self.cfg.nodes {
            for inst in &mut block.instructions {
                if let Some(dst) = inst.dst() {
                    defs.entry(dst.clone()).or_default().insert(*node);
                }
            }
        }

        defs
    }

    fn add_phi(&mut self) {
        let mut defs = self.defs();
        let vars = defs.keys().cloned().collect::<Vec<_>>();

        for v in &vars {
            for d in defs[v].clone() {
                for block in &self.dominate_frontiers[&d] {
                    self.phi
                        .entry(*block)
                        .or_default()
                        .entry(v.clone())
                        .or_default()
                        .insert(d, v.clone());

                    defs.entry(v.clone()).or_default().insert(*block);
                }
            }
        }

        let mut stack = HashMap::new();

        let mut counter = 0;
        for s in self.cfg.entry.successors.clone() {
            if let NodeId::Block(s) = s {
                self.rename(s, &mut stack, &mut counter);
            }
        }
    }

    fn rename(
        &mut self,
        block: usize,
        stack: &mut HashMap<EcoString, Vec<EcoString>>,
        counter: &mut usize,
    ) {
        let mut new_name = |old: &EcoString| -> EcoString {
            let n = format!("ssa.{}.{}", old, counter).into();
            *counter += 1;
            n
        };

        let mut pushed = Vec::new();

        for (name, phi) in self.phi.entry(block).or_default().clone() {
            let new_name = new_name(&name);
            stack
                .entry(name.clone())
                .or_default()
                .push(new_name.clone());
            pushed.push(name.clone());

            self.phi.get_mut(&block).unwrap().remove(&name);
            self.phi
                .get_mut(&block)
                .unwrap()
                .insert(new_name.clone(), phi.clone());
        }

        for inst in self
            .cfg
            .nodes
            .get_mut(&block)
            .unwrap()
            .instructions
            .iter_mut()
        {
            inst.map_args(|arg| {
                let new_name = stack
                    .entry(arg.clone())
                    .or_default()
                    .last()
                    .unwrap_or_else(|| arg);
                *arg = new_name.clone();
            });
            if let Some(dst) = inst.dst() {
                let new_name = new_name(dst);
                stack.entry(dst.clone()).or_default().push(new_name.clone());
                pushed.push(dst.clone());
                *dst = new_name;
            }
        }

        for s in self.cfg.nodes[&block].successors.iter().filter_map(|id| {
            if let NodeId::Block(id) = id {
                Some(id)
            } else {
                None
            }
        }) {
            for (old_name, phi) in self.phi.entry(*s).or_default() {
                let new_name = stack
                    .entry(old_name.clone())
                    .or_default()
                    .last()
                    .unwrap_or_else(|| old_name)
                    .clone();

                phi.insert(block, new_name.clone());
            }
        }

        dbg!(block, self.immediate_dominates(block));
        for d in self.immediate_dominates(block) {
            self.rename(d, stack, counter);
        }

        for var in pushed {
            stack.get_mut(&var).unwrap().pop();
        }
    }

    fn strictly_dominates(&self, a: usize, b: usize) -> bool {
        self.dominates[&a].contains(&b) && a != b
    }

    fn immediate_dominates(&self, a: usize) -> HashSet<usize> {
        dbg!(&self.dominates);
        self.dominates[&a]
            .iter()
            .filter(|&&b| {
                a != b
                    && !self.dominated[&b]
                        .iter()
                        .filter(|&&x| x != b)
                        .any(|&x| self.strictly_dominates(a, x))
            })
            .copied()
            .collect()
    }
}
