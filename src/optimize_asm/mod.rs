use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    ast::{BaseType, VarType},
    codegen::{Instruction, Operand, Pseudo, Register},
    control_flow::Cfg,
    semantics::type_check::{Attr, SymbolTable},
};

mod liveness_analysis;

// R10 and R11 are used as temporary registers in the code generator
const FREE_REGISTERS: [Register; 11] = [
    Register::Ax,
    Register::Bx,
    Register::Cx,
    Register::Dx,
    Register::Di,
    Register::Si,
    Register::R8,
    Register::R9,
    Register::R13,
    Register::R14,
    Register::R15,
];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NodeId {
    Register(Register),
    Pseudo(EcoString),
}

impl TryInto<NodeId> for &Operand {
    type Error = ();

    fn try_into(self) -> Result<NodeId, Self::Error> {
        match self {
            Operand::Reg(reg) => Ok(NodeId::Register(*reg)),
            Operand::Pseudo(Pseudo::Mem { name, .. }) => Ok(NodeId::Pseudo(name.clone())),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    id: NodeId,
    neighbors: Vec<NodeId>,
    spill_cost: f32,
    color: Option<usize>,
    pruned: bool,
}

struct ColoringGraph<'a> {
    map: HashMap<NodeId, Node>,
    symbol_table: &'a SymbolTable,
}

impl Node {
    fn new(id: NodeId) -> Self {
        Self {
            id,
            neighbors: Vec::new(),
            spill_cost: 0.0,
            color: None,
            pruned: false,
        }
    }
}

fn is_int_scalar(n: &NodeId, symbol_table: &SymbolTable) -> bool {
    match n {
        NodeId::Pseudo(name) => {
            if let Attr::Local(ty) = &symbol_table[name] {
                if ty.is_scalar() && *ty != VarType::Base(BaseType::Double) {
                    return true;
                }
            }
        }
        NodeId::Register(reg) => {
            if FREE_REGISTERS.contains(reg) {
                return true;
            }
        }
    }

    false
}

impl<'a> ColoringGraph<'a> {
    fn new(program: &[Instruction], symbol_table: &'a SymbolTable) -> Self {
        let mut me = Self {
            map: HashMap::new(),
            symbol_table,
        };
        me.add_base_registers(&FREE_REGISTERS);

        let cfg = Cfg::new(program);
        me.collect_pseudo_vars(program);
        me.add_edges(&cfg);
        me
    }

    fn add_base_registers(&mut self, registers: &[Register]) {
        for &reg in &FREE_REGISTERS {
            self.add_var(NodeId::Register(reg));
        }

        for &reg in registers {
            for &neighbor in registers {
                if reg != neighbor {
                    self.map
                        .get_mut(&NodeId::Register(reg))
                        .unwrap()
                        .neighbors
                        .push(NodeId::Register(neighbor));
                }
            }
        }
    }

    fn check_node_id(&self, n: &NodeId) -> bool {
        match n {
            NodeId::Register(r) => FREE_REGISTERS.contains(r),
            NodeId::Pseudo(name) => match &self.symbol_table[name] {
                Attr::Local(ty) => ty.is_scalar() && *ty != VarType::Base(BaseType::Double),
                _ => false,
            },
        }
    }

    fn add_var(&mut self, n: NodeId) {
        if is_int_scalar(&n, &self.symbol_table) {
            self.map.insert(n.clone(), Node::new(n));
        }
    }

    fn add_edge(&mut self, a: NodeId, b: NodeId) {
        if a != b && self.map.contains_key(&a) && self.map.contains_key(&b) {
            self.map.get_mut(&a).unwrap().neighbors.push(b.clone());
            self.map.get_mut(&b).unwrap().neighbors.push(a);
        }
    }

    fn add_edges(&mut self, cfg: &Cfg<Instruction>) {
        let mut annotation = liveness_analysis::Annotation::default();

        annotation.iterate(cfg, &self.symbol_table);

        for node in cfg.nodes.values() {
            let annotation = annotation.instruction_annotation.get(&node.id).unwrap();

            for (inst, live) in node.instructions.iter().zip(annotation.iter()) {
                let (_used, updated) =
                    liveness_analysis::find_used_and_updated(inst, &self.symbol_table);

                for l in live {
                    if let Instruction::Mov { src, .. } = inst {
                        if src.try_into().as_ref() == Ok(l) {
                            continue;
                        }
                    }

                    for &u in &updated {
                        if let Ok(u) = u.try_into() {
                            self.add_edge(l.clone(), u);
                        }
                    }
                }
            }
        }
    }

    fn collect_pseudo_vars(&mut self, insts: &[Instruction]) {
        let mut add_op = |op: &Operand| {
            if let Ok(id) = op.try_into() {
                self.add_var(id);
            }
        };

        for inst in insts {
            match inst {
                Instruction::Binary { lhs, rhs, .. } => {
                    add_op(lhs);
                    add_op(rhs);
                }
                Instruction::Call(op) => {
                    add_op(op);
                }
                Instruction::Cdq(_) => {}
                Instruction::Cmp(_, op1, op2) => {
                    add_op(op1);
                    add_op(op2);
                }
                Instruction::Cvtsi2sd { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Cvttsd2si { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Div(_, op) => {
                    add_op(op);
                }
                Instruction::Idiv(_, op) => {
                    add_op(op);
                }
                Instruction::Jmp(_) => {}
                Instruction::JmpCc(..) => {}
                Instruction::Label(_) => {}
                Instruction::Lea { src: _, dst } => {
                    // add_op(src);
                    add_op(dst);
                }
                Instruction::Mov { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::MovZeroExtend { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Movsx { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Pop(_) => {}
                Instruction::Push(op) => {
                    add_op(op);
                }
                Instruction::Ret => {}
                Instruction::SetCc(_, op) => {
                    add_op(op);
                }
                Instruction::Unary { src, .. } => {
                    add_op(src);
                }
            }
        }
    }
}
