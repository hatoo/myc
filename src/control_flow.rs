use std::collections::{BTreeMap, HashMap, HashSet};

use ecow::EcoString;

use crate::{codegen, tacky};

pub enum GeneralizedInstruction {
    Return,
    Jump(EcoString),
    MayJump(EcoString),
    Label(EcoString),
    Others,
}

impl<'a> From<&'a tacky::Instruction> for GeneralizedInstruction {
    fn from(val: &'a tacky::Instruction) -> Self {
        match val {
            tacky::Instruction::Return(..) => GeneralizedInstruction::Return,
            tacky::Instruction::Jump(label) => GeneralizedInstruction::Jump(label.clone()),
            tacky::Instruction::JumpIfNotZero { dst, .. }
            | tacky::Instruction::JumpIfZero { dst, .. } => {
                GeneralizedInstruction::MayJump(dst.clone())
            }
            tacky::Instruction::Label(label) => GeneralizedInstruction::Label(label.clone()),
            _ => GeneralizedInstruction::Others,
        }
    }
}

impl<'a> From<&'a codegen::Instruction> for GeneralizedInstruction {
    fn from(val: &'a codegen::Instruction) -> Self {
        match val {
            codegen::Instruction::Ret => GeneralizedInstruction::Return,
            codegen::Instruction::Jmp(label) => GeneralizedInstruction::Jump(label.clone()),
            codegen::Instruction::JmpCc(_, label) => GeneralizedInstruction::MayJump(label.clone()),
            codegen::Instruction::Label(label) => GeneralizedInstruction::Label(label.clone()),
            _ => GeneralizedInstruction::Others,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum NodeId {
    Entry,
    Exit,
    Block(usize),
}

#[derive(Debug)]
pub struct Node<I> {
    pub id: usize,
    pub instructions: Vec<I>,
    pub predecessors: HashSet<NodeId>,
    pub successors: HashSet<NodeId>,
}

#[derive(Debug)]
pub struct Entry {
    pub successors: HashSet<NodeId>,
}

#[derive(Debug)]
struct Exit {
    predecessors: HashSet<NodeId>,
}

#[derive(Debug)]
pub struct Cfg<I> {
    pub entry: Entry,
    exit: Exit,
    pub nodes: BTreeMap<usize, Node<I>>,
    label_map: HashMap<EcoString, NodeId>,
}

impl<I> Cfg<I>
where
    I: Clone,
    for<'a> &'a I: Into<GeneralizedInstruction>,
{
    pub fn new(program: &[I]) -> Self {
        let blocks = Self::partition_into_basic_blocks(program);
        let mut graph = Self::put_node_id(blocks);
        graph.add_all_edges();
        graph
    }

    pub fn program(&self) -> Vec<I> {
        let mut program = Vec::new();
        for node in self.nodes.values() {
            program.extend(node.instructions.iter().cloned());
        }

        program
    }

    pub fn eliminate_unreachable_code(&mut self) {
        self.remove_unreachable_nodes();
        self.remove_redundant_jumps();
        self.remove_useless_label();
        self.remove_empty_nodes();
    }

    fn add_edge(&mut self, from: NodeId, to: NodeId) {
        match from {
            NodeId::Entry => {
                self.entry.successors.insert(to);
            }
            NodeId::Block(id) => {
                self.nodes.get_mut(&id).unwrap().successors.insert(to);
            }
            NodeId::Exit => {
                panic!()
            }
        }

        match to {
            NodeId::Entry => {
                panic!()
            }
            NodeId::Block(id) => {
                self.nodes.get_mut(&id).unwrap().predecessors.insert(from);
            }
            NodeId::Exit => {
                self.exit.predecessors.insert(from);
            }
        }
    }

    fn remove_unreachable_nodes(&mut self) {
        let mut reachable = HashSet::new();

        let mut stack = self.entry.successors.iter().cloned().collect::<Vec<_>>();

        while let Some(node) = stack.pop() {
            if let NodeId::Block(id) = node {
                if reachable.insert(node) {
                    stack.extend(self.nodes[&id].successors.iter().cloned());
                }
            }
        }

        self.nodes
            .retain(|k, _| reachable.contains(&NodeId::Block(*k)));

        for node in self.nodes.values_mut() {
            node.predecessors.retain(|p| reachable.contains(p));
        }
        self.exit.predecessors.retain(|p| reachable.contains(p));
    }

    fn remove_redundant_jumps(&mut self) {
        let next_ids = self
            .nodes
            .keys()
            .copied()
            .collect::<Vec<usize>>()
            .windows(2)
            .map(|w| w[1])
            .collect::<Vec<_>>();
        for (next, node) in next_ids.into_iter().zip(self.nodes.values_mut()) {
            if let Some(GeneralizedInstruction::Jump(_) | GeneralizedInstruction::MayJump(..)) =
                node.instructions.last().map(Into::into)
            {
                let default_succ = NodeId::Block(next);
                if node.successors.iter().all(|s| s == &default_succ) {
                    node.instructions.pop();
                }
            }
        }
    }

    fn remove_useless_label(&mut self) {
        for (prev, next) in self
            .nodes
            .keys()
            .copied()
            .zip(self.nodes.keys().copied().skip(1))
            .collect::<Vec<_>>()
            .into_iter()
        {
            if let Some(GeneralizedInstruction::Label(_)) =
                self.nodes[&next].instructions.first().map(Into::into)
            {
                if self.nodes[&next]
                    .predecessors
                    .iter()
                    .all(|p| *p == NodeId::Block(prev))
                {
                    self.nodes.get_mut(&next).unwrap().instructions.remove(0);
                }
            }
        }

        if let Some(mut node) = self.nodes.first_entry() {
            if let Some(GeneralizedInstruction::Label(_)) =
                node.get().instructions.first().map(Into::into)
            {
                if node.get().predecessors.is_empty() {
                    node.get_mut().instructions.remove(0);
                }
            }
        }
    }

    fn remove_empty_nodes(&mut self) {
        for k in self.nodes.keys().copied().collect::<Vec<_>>() {
            let node = self.nodes.get(&k).unwrap();
            if self.nodes[&k].instructions.is_empty() {
                let preds = node.predecessors.clone();
                let succs = node.successors.clone();

                for &pred in &preds {
                    for &succ in &succs {
                        self.add_edge(pred, succ);
                    }
                }

                for pred in &preds {
                    if let NodeId::Block(pred) = pred {
                        self.nodes
                            .get_mut(pred)
                            .unwrap()
                            .successors
                            .remove(&NodeId::Block(k));
                    }
                }

                for succ in &succs {
                    if let NodeId::Block(succ) = succ {
                        self.nodes
                            .get_mut(succ)
                            .unwrap()
                            .predecessors
                            .remove(&NodeId::Block(k));
                    }
                }
            }
        }
    }

    fn add_all_edges(&mut self) {
        for (id, node) in &self.nodes {
            if let Some(GeneralizedInstruction::Label(label)) =
                node.instructions.first().map(Into::into)
            {
                self.label_map.insert(label.clone(), NodeId::Block(*id));
            }
        }

        self.add_edge(NodeId::Entry, NodeId::Block(0));

        let max_id = *self.nodes.keys().max().unwrap();

        for id in 0..=max_id {
            let next_id = if id == max_id {
                NodeId::Exit
            } else {
                NodeId::Block(id + 1)
            };

            let last_inst = self.nodes[&id].instructions.last().unwrap().into();
            match last_inst {
                GeneralizedInstruction::Return => {
                    self.add_edge(NodeId::Block(id), NodeId::Exit);
                }
                GeneralizedInstruction::Jump(label) => {
                    self.add_edge(NodeId::Block(id), self.label_map[&label]);
                }
                GeneralizedInstruction::MayJump(dst) => {
                    self.add_edge(NodeId::Block(id), self.label_map[&dst]);
                    self.add_edge(NodeId::Block(id), next_id);
                }
                _ => {
                    self.add_edge(NodeId::Block(id), next_id);
                }
            }
        }
    }

    fn partition_into_basic_blocks(program: &[I]) -> Vec<Vec<I>> {
        let mut blocks = Vec::new();
        let mut current_block = Vec::new();
        for inst in program {
            match inst.into() {
                GeneralizedInstruction::Label(_) => {
                    if !current_block.is_empty() {
                        blocks.push(std::mem::take(&mut current_block));
                    }
                    current_block.push(inst.clone());
                }
                GeneralizedInstruction::Jump(..)
                | GeneralizedInstruction::MayJump(..)
                | GeneralizedInstruction::Return => {
                    current_block.push(inst.clone());
                    blocks.push(std::mem::take(&mut current_block));
                }
                _ => {
                    current_block.push(inst.clone());
                }
            }
        }

        if !current_block.is_empty() {
            blocks.push(current_block);
        }

        blocks
    }

    fn put_node_id(blocks: Vec<Vec<I>>) -> Cfg<I> {
        let mut basic_blocks = BTreeMap::new();

        for (id, block) in blocks.into_iter().enumerate() {
            basic_blocks.insert(
                id,
                Node {
                    id,
                    instructions: block,
                    predecessors: Default::default(),
                    successors: Default::default(),
                },
            );
        }

        Cfg {
            entry: Entry {
                successors: Default::default(),
            },
            exit: Exit {
                predecessors: Default::default(),
            },
            nodes: basic_blocks,
            label_map: HashMap::new(),
        }
    }

    pub fn all_instructions(&self) -> impl Iterator<Item = &I> {
        self.nodes
            .values()
            .flat_map(|node| node.instructions.iter())
    }
}
