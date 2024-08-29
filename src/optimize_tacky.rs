use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::{Add, Div, Mul, Rem, Sub},
};

use ecow::EcoString;

use crate::{
    ast::{Const, VarType},
    semantics::type_check::SymbolTable,
    tacky::{BinaryOp, Instruction, Program, TopLevelItem, UnaryOp, Val},
};

macro_rules! fold_binary {
    ($arg:expr; $($op:pat => ($f:ident, $fd:ident)),*) => {
        match $arg {
            $(
                Instruction::Binary {
                    op: $op,
                    lhs: Val::Constant(lhs),
                    rhs: Val::Constant(rhs),
                    dst,
                } => match (lhs, rhs) {
                    (Const::Char(lhs), Const::Char(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Char((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::UChar(lhs), Const::UChar(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::UChar((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Int(lhs), Const::Int(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Uint(lhs), Const::Uint(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Uint((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Long(lhs), Const::Long(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Long((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Ulong(lhs), Const::Ulong(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Ulong((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Double(lhs), Const::Double(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Double((*lhs).$fd(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    _ => {}
                },
            )*
            _ => {}
        }
    };
}

macro_rules! fold_binary_cmp {
    ($arg:expr; $($op:pat => $f:ident),*) => {
        match $arg {
            $(
                Instruction::Binary {
                    op: $op,
                    lhs: Val::Constant(lhs),
                    rhs: Val::Constant(rhs),
                    dst,
                } => match (lhs, rhs) {
                    (Const::Char(lhs), Const::Char(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::UChar(lhs), Const::UChar(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Int(lhs), Const::Int(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Uint(lhs), Const::Uint(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Long(lhs), Const::Long(rhs)) => {
                        *$arg = Instruction::Copy {
                             src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Ulong(lhs), Const::Ulong(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Double(lhs), Const::Double(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    _ => {}
                },
            )*
            _ => {}
        }
    };
}

pub fn constant_folding(program: &mut Vec<Instruction>, symbol_table: &SymbolTable) {
    for inst in program.iter_mut() {
        if let Instruction::Binary {
            op: BinaryOp::Divide | BinaryOp::Remainder,
            lhs: _,
            rhs: Val::Constant(c),
            dst: _,
        } = inst
        {
            if matches!(
                c,
                Const::Int(0) | Const::Uint(0) | Const::Long(0) | Const::Ulong(0)
            ) {
                continue;
            }
        }
        fold_binary!(inst; BinaryOp::Add => (wrapping_add, add), BinaryOp::Subtract => (wrapping_sub, sub), BinaryOp::Multiply => (wrapping_mul, mul), BinaryOp::Divide => (div, div), BinaryOp::Remainder => (rem, rem));
        fold_binary_cmp!(inst; BinaryOp::Equal => eq, BinaryOp::NotEqual => ne, BinaryOp::LessThan => lt, BinaryOp::LessOrEqual => le, BinaryOp::GreaterThan => gt, BinaryOp::GreaterOrEqual => ge);

        if let Instruction::Unary {
            op,
            src: Val::Constant(c),
            dst,
        } = inst
        {
            let val = match op {
                UnaryOp::Negate => match c {
                    Const::Char(c) => Const::Char(-*c),
                    Const::UChar(c) => Const::UChar(!*c + 1),
                    Const::Int(c) => Const::Int(-*c),
                    Const::Uint(c) => Const::Uint(!*c + 1),
                    Const::Long(c) => Const::Long(-*c),
                    Const::Ulong(c) => Const::Ulong(!*c + 1),
                    Const::Double(c) => Const::Double(-*c),
                },
                UnaryOp::Complement => match c {
                    Const::Char(c) => Const::Char(!*c),
                    Const::UChar(c) => Const::UChar(!*c),
                    Const::Int(c) => Const::Int(!*c),
                    Const::Uint(c) => Const::Uint(!*c),
                    Const::Long(c) => Const::Long(!*c),
                    Const::Ulong(c) => Const::Ulong(!*c),
                    _ => panic!(),
                },
                UnaryOp::Not => Const::Int(if c.get_ulong() == 0 { 1 } else { 0 }),
            };

            *inst = Instruction::Copy {
                src: Val::Constant(val),
                dst: dst.clone(),
            };
        }

        if let Instruction::DoubleToInt {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::DoubleToUint {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::SignExtend {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::ZeroExtend {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::Truncate {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            let src = match dst.ty(symbol_table) {
                VarType::Base(base) => match base {
                    crate::ast::BaseType::Char => Const::Char(c.get_char()),
                    crate::ast::BaseType::SChar => Const::Char(c.get_char()),
                    crate::ast::BaseType::UChar => Const::UChar(c.get_uchar()),
                    crate::ast::BaseType::Int => Const::Int(c.get_int()),
                    crate::ast::BaseType::Long => Const::Long(c.get_long()),
                    crate::ast::BaseType::Uint => Const::Uint(c.get_uint()),
                    crate::ast::BaseType::Ulong => Const::Ulong(c.get_ulong()),
                    crate::ast::BaseType::Double => Const::Double(c.get_double()),
                },
                VarType::Pointer(_) => Const::Ulong(c.get_ulong()),
                _ => panic!(),
            };
            *inst = Instruction::Copy {
                src: Val::Constant(src),
                dst: dst.clone(),
            };
        }

        if let Instruction::IntToDouble {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::UintToDouble {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            let src = Val::Constant(Const::Double(c.get_double()));
            *inst = Instruction::Copy {
                src,
                dst: dst.clone(),
            };
        }

        if let Instruction::JumpIfNotZero {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            if c.is_zero() {
                *inst = Instruction::Nop;
            } else {
                *inst = Instruction::Jump(dst.clone());
            }
        }

        if let Instruction::JumpIfZero {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            if c.is_zero() {
                *inst = Instruction::Jump(dst.clone());
            } else {
                *inst = Instruction::Nop;
            }
        }
    }

    program.retain(|inst| !matches!(inst, Instruction::Nop));
}

pub enum OptimizeOption {
    ConstantFolding,
    DeadCodeElimination,
}

pub fn optimize(program: &mut Program, symbol_table: &SymbolTable, options: &[OptimizeOption]) {
    for top in &mut program.top_levels {
        if let TopLevelItem::Function(f) = top {
            loop {
                let snapshot = f.body.clone();

                for opt in options {
                    match opt {
                        OptimizeOption::ConstantFolding => {
                            constant_folding(&mut f.body, symbol_table);
                        }
                        OptimizeOption::DeadCodeElimination => {
                            let mut graph = Graph::new(&f.body);
                            graph.eliminate_unreachable_code();
                            f.body = graph.program();
                        }
                    }
                }

                if snapshot == f.body {
                    break;
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum NodeId {
    Entry,
    Exit,
    Block(usize),
}

impl NodeId {
    fn next(self) -> NodeId {
        match self {
            NodeId::Entry => NodeId::Block(0),
            NodeId::Block(n) => NodeId::Block(n + 1),
            NodeId::Exit => panic!(),
        }
    }
}

struct Node {
    id: usize,
    instructions: Vec<Instruction>,
    predecessors: HashSet<NodeId>,
    successors: HashSet<NodeId>,
}

struct Entry {
    successors: HashSet<NodeId>,
}

struct Exit {
    predecessors: HashSet<NodeId>,
}

struct Graph {
    entry: Entry,
    exit: Exit,
    nodes: BTreeMap<usize, Node>,
    label_map: HashMap<EcoString, NodeId>,
}

impl Graph {
    fn new(program: &[Instruction]) -> Self {
        let blocks = Self::partition_into_basic_blocks(program);
        let mut graph = Self::put_node_id(blocks);
        graph.add_all_edges();
        graph
    }

    fn program(&self) -> Vec<Instruction> {
        let mut program = Vec::new();
        for node in self.nodes.values() {
            program.extend(node.instructions.iter().cloned());
        }

        program
    }

    fn eliminate_unreachable_code(&mut self) {
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
        for (next, (i, node)) in next_ids.into_iter().zip(self.nodes.iter_mut()) {
            if let Some(
                Instruction::Jump(_)
                | Instruction::JumpIfZero { .. }
                | Instruction::JumpIfNotZero { .. },
            ) = node.instructions.last()
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
            if let Some(Instruction::Label(_)) = self.nodes[&next].instructions.first() {
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
            if let Some(Instruction::Label(_)) = node.get().instructions.first() {
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
            if let Some(Instruction::Label(label)) = node.instructions.get(0) {
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

            let last_inst = self.nodes[&id].instructions.last().cloned().unwrap();
            match last_inst {
                Instruction::Return(_) => {
                    self.add_edge(NodeId::Block(id), NodeId::Exit);
                }
                Instruction::Jump(label) => {
                    self.add_edge(NodeId::Block(id), self.label_map[&label]);
                }
                Instruction::JumpIfZero { dst, .. } | Instruction::JumpIfNotZero { dst, .. } => {
                    self.add_edge(NodeId::Block(id), self.label_map[&dst]);
                    self.add_edge(NodeId::Block(id), next_id);
                }
                _ => {
                    self.add_edge(NodeId::Block(id), next_id);
                }
            }
        }
    }

    fn partition_into_basic_blocks(program: &[Instruction]) -> Vec<Vec<Instruction>> {
        let mut blocks = Vec::new();
        let mut current_block = Vec::new();
        for inst in program {
            match inst {
                Instruction::Label(_) => {
                    if !current_block.is_empty() {
                        blocks.push(std::mem::take(&mut current_block));
                    }
                    current_block.push(inst.clone());
                }
                Instruction::Jump(..)
                | Instruction::JumpIfNotZero { .. }
                | Instruction::JumpIfZero { .. }
                | Instruction::Return { .. } => {
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

    fn put_node_id(blocks: Vec<Vec<Instruction>>) -> Graph {
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

        Graph {
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
}
