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
        let defs = self.defs();
        let vars = defs.keys().cloned().collect::<Vec<_>>();

        let mut phis: HashMap<usize, HashMap<EcoString, HashMap<usize, EcoString>>> = self
            .cfg
            .nodes
            .keys()
            .map(|&k| (k, HashMap::new()))
            .collect();

        for v in &vars {
            let mut stack: Vec<usize> = defs[v].iter().copied().collect();
            let mut visited = HashSet::new();
            while let Some(d) = stack.pop() {
                if visited.insert(d) {
                    for block in &self.dominate_frontiers[&d] {
                        phis.entry(*block)
                            .or_default()
                            .insert(v.clone(), HashMap::new());

                        stack.push(*block);
                    }
                }
            }
        }

        for (block, phi) in &mut phis {
            for p in &self.cfg.nodes[block].predecessors {
                if let NodeId::Block(p) = p {
                    for (v, map) in phi.iter_mut() {
                        map.insert(p.clone(), v.clone());
                    }
                }
            }
        }

        self.phi = phis;

        let mut stack = HashMap::new();
        for s in self.cfg.entry.successors.clone() {
            if let NodeId::Block(s) = s {
                self.rename(s, &mut stack);
            }
        }
    }

    fn rename(&mut self, block: usize, stack: &mut HashMap<EcoString, Vec<EcoString>>) {
        let mut counter = 0;
        let mut new_name = |old: &EcoString| -> EcoString {
            let n = format!("ssa.{}.{}_{}", old, block, counter).into();
            n
        };

        let mut pushed = Vec::new();

        let new_phi = self.phi[&block]
            .clone()
            .into_iter()
            .map(|(k, v)| {
                let new_name = new_name(&k);
                stack.entry(k.clone()).or_default().push(new_name.clone());
                (new_name, v)
            })
            .collect();
        self.phi.insert(block, new_phi);

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
            for (_, map) in self.phi.entry(*s).or_default() {
                let old_name = map[&block].clone();

                let new_name = stack
                    .entry(old_name.clone())
                    .or_default()
                    .last()
                    .unwrap_or_else(|| &old_name)
                    .clone();

                map.insert(block, new_name);
            }
        }

        for d in self.immediate_dominates(block) {
            self.rename(d, stack);
        }

        for var in pushed {
            stack.get_mut(&var).unwrap().pop();
        }
    }

    fn strictly_dominates(&self, a: usize, b: usize) -> bool {
        self.dominates[&a].contains(&b) && a != b
    }

    fn immediate_dominates(&self, a: usize) -> HashSet<usize> {
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
