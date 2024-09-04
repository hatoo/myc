use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    ast::{BaseType, FunType, VarType},
    codegen::{
        asm_type, classify_struct, is_return_in_memory, Class, Instruction, Operand, Pseudo,
        Register,
    },
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
    neighbors: HashSet<NodeId>,
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
            neighbors: HashSet::new(),
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
        me.add_spill_costs(program);
        me.add_edges(&cfg);
        me
    }

    fn add_base_registers(&mut self, registers: &[Register]) {
        for &reg in &FREE_REGISTERS {
            self.map.insert(
                NodeId::Register(reg),
                Node {
                    id: NodeId::Register(reg),
                    spill_cost: 1e9,
                    neighbors: HashSet::new(),
                    color: None,
                    pruned: false,
                },
            );
        }

        for &reg in registers {
            for &neighbor in registers {
                if reg != neighbor {
                    self.map
                        .get_mut(&NodeId::Register(reg))
                        .unwrap()
                        .neighbors
                        .insert(NodeId::Register(neighbor));
                }
            }
        }
    }

    fn color_graph(&mut self) {
        // TODO: optimize
        let k = FREE_REGISTERS.len();

        if self.map.values().all(|n| n.pruned) {
            return;
        }

        let chosen_id = if let Some(chosen_node) = self
            .map
            .values()
            .filter(|n| !n.pruned)
            .find(|n| n.neighbors.iter().filter(|&n| !self.map[n].pruned).count() < k)
        {
            chosen_node.id.clone()
        } else {
            self.map
                .values()
                .filter(|n| !n.pruned)
                .min_by(|n1, n2| {
                    (n1.spill_cost
                        / (n1.neighbors.iter().filter(|n| !self.map[n].pruned).count() + 1) as f32)
                        .total_cmp(
                            &(n2.spill_cost
                                / (n2.neighbors.iter().filter(|n| !self.map[n].pruned).count() + 1)
                                    as f32),
                        )
                })
                .unwrap()
                .id
                .clone()
        };

        let node = self.map.get_mut(&chosen_id).unwrap();
        node.pruned = true;

        self.color_graph();

        let mut colors = vec![false; k];

        let node = self.map.get(&chosen_id).unwrap();
        for neighbor in &node.neighbors {
            if let Some(neighbor) = self.map.get(neighbor) {
                if let Some(color) = neighbor.color {
                    colors[color] = true;
                }
            }
        }

        if colors.iter().any(|&c| !c) {
            match chosen_id {
                NodeId::Register(r) if r.is_callee_saved() => {
                    let color = colors
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, &c)| !c)
                        .unwrap()
                        .0;
                    self.map.get_mut(&chosen_id).unwrap().color = Some(color);
                }
                _ => {
                    let color = colors.iter().enumerate().find(|(_, &c)| !c).unwrap().0;
                    self.map.get_mut(&chosen_id).unwrap().color = Some(color);
                }
            }

            self.map.get_mut(&chosen_id).unwrap().pruned = false;
        }
    }

    fn create_register_map(&self) -> (HashMap<EcoString, Register>, HashSet<Register>) {
        let mut color_map = HashMap::new();

        for node in self.map.values() {
            if let NodeId::Register(r) = &node.id {
                if let Some(color) = node.color {
                    color_map.insert(color, r.clone());
                }
            }
        }

        let mut register_map = HashMap::new();
        let mut callee_saved = HashSet::new();

        for node in self.map.values() {
            match &node.id {
                NodeId::Pseudo(name) => {
                    if let Some(color) = node.color {
                        let hardreg = color_map[&color];
                        register_map.insert(name.clone(), hardreg);
                        if hardreg.is_callee_saved() {
                            callee_saved.insert(hardreg);
                        }
                    }
                }
                _ => {}
            }
        }

        (register_map, callee_saved)
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
            self.map.get_mut(&a).unwrap().neighbors.insert(b.clone());
            self.map.get_mut(&b).unwrap().neighbors.insert(a);
        }
    }

    fn increment_spill_cost(&mut self, n: &NodeId) {
        if let Some(node) = self.map.get_mut(n) {
            node.spill_cost += 1.0;
        }
    }

    fn add_edges(&mut self, cfg: &Cfg<Instruction>) {
        let mut annotation = liveness_analysis::Annotation::default();

        annotation.iterate(cfg, &self.symbol_table);

        for node in cfg.nodes.values() {
            let annotation = annotation.instruction_annotation.get(&node.id).unwrap();

            for (inst, live) in node.instructions.iter().zip(annotation.iter()) {
                let (_used, updated) = find_used_and_updated(inst, &self.symbol_table);

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
        for inst in insts {
            let (used, updated) = find_used_and_updated(inst, &self.symbol_table);
            for &op in used.iter().chain(updated.iter()) {
                if let Ok(id) = op.try_into() {
                    self.add_var(id);
                }
            }
        }
    }

    fn add_spill_costs(&mut self, insts: &[Instruction]) {
        for inst in insts {
            let (used, updated) = find_used_and_updated(inst, &self.symbol_table);
            for &op in used.iter().chain(updated.iter()) {
                if let Ok(id) = op.try_into() {
                    self.increment_spill_cost(&id);
                }
            }
        }
    }
}

fn find_used_and_updated<'a>(
    inst: &'a Instruction,
    symbol_table: &SymbolTable,
) -> (Vec<&'a Operand>, Vec<&'a Operand>) {
    match inst {
        Instruction::Mov { src, dst, .. }
        | Instruction::MovZeroExtend { src, dst, .. }
        | Instruction::Movsx { src, dst, .. } => (vec![src], vec![dst]),
        Instruction::Binary { lhs, rhs, .. } => (vec![lhs, rhs], vec![rhs]),
        Instruction::Unary { src, .. } => (vec![src], vec![src]),
        Instruction::Cmp(_, v1, v2) => (vec![v1, v2], vec![]),
        Instruction::SetCc(_, dst) => (vec![], vec![dst]),
        Instruction::Push(op) => (vec![op], vec![]),
        Instruction::Idiv(_, divisor) => (
            vec![
                divisor,
                &Operand::Reg(Register::Ax),
                &Operand::Reg(Register::Dx),
            ],
            vec![&Operand::Reg(Register::Ax), &Operand::Reg(Register::Dx)],
        ),
        Instruction::Cdq(_) => (
            vec![&Operand::Reg(Register::Ax)],
            vec![&Operand::Reg(Register::Dx)],
        ),
        Instruction::Call(op) => {
            let Operand::Pseudo(Pseudo::Mem { name, .. }) = op else {
                panic!("Do this before pseudo to stack phase")
            };

            let Attr::Fun { ty, .. } = &symbol_table[name] else {
                unreachable!()
            };

            let (i, d, _) = fun_use_registers(ty, symbol_table);

            const INT_REGS: [Operand; 6] = [
                Operand::Reg(Register::Di),
                Operand::Reg(Register::Si),
                Operand::Reg(Register::Dx),
                Operand::Reg(Register::Cx),
                Operand::Reg(Register::R8),
                Operand::Reg(Register::R9),
            ];

            const DOUBLE_REGS: [Operand; 8] = [
                Operand::Reg(Register::Xmm(0)),
                Operand::Reg(Register::Xmm(1)),
                Operand::Reg(Register::Xmm(2)),
                Operand::Reg(Register::Xmm(3)),
                Operand::Reg(Register::Xmm(4)),
                Operand::Reg(Register::Xmm(5)),
                Operand::Reg(Register::Xmm(6)),
                Operand::Reg(Register::Xmm(7)),
            ];

            let used = INT_REGS[..i]
                .iter()
                .chain(DOUBLE_REGS[..d].iter())
                .collect();

            (
                used,
                vec![
                    &Operand::Reg(Register::Di),
                    &Operand::Reg(Register::Si),
                    &Operand::Reg(Register::Dx),
                    &Operand::Reg(Register::Cx),
                    &Operand::Reg(Register::R8),
                    &Operand::Reg(Register::R9),
                    &Operand::Reg(Register::Ax),
                ],
            )
        }
        _ => (vec![], vec![]),
    }
}

fn fun_use_registers(ty: &FunType, symbol_table: &SymbolTable) -> (usize, usize, usize) {
    let mut int_regs = 0;
    let mut double_regs = 0;
    let mut stack_args = 0;

    let return_in_memory = is_return_in_memory(&ty.ret, symbol_table);

    if return_in_memory {
        int_regs += 1;
    }

    let int_regs_available = 6;

    for ty in &ty.params {
        let asm_ty = asm_type(ty, symbol_table);
        match &ty {
            VarType::Base(BaseType::Double) => {
                if double_regs < 8 {
                    double_regs += 1;
                } else {
                    stack_args += 1;
                }
            }
            VarType::Struct(name) => {
                let structure = symbol_table.struct_def(name);
                let classes = classify_struct(structure, &symbol_table);
                let mut use_stack = true;
                let struct_size = structure.size;

                if classes[0] != Class::Memory {
                    let mut tentative_ints = 0;
                    let mut tentative_doubles = 0;
                    let mut offset = 0;
                    for &class in &classes {
                        if class == Class::Sse {
                            tentative_doubles += 1;
                        } else {
                            tentative_ints += 1;
                        }

                        offset += 8;
                    }

                    if (tentative_doubles + double_regs) <= 8
                        && (tentative_ints + int_regs) <= int_regs_available
                    {
                        double_regs += tentative_doubles;
                        int_regs += tentative_ints;
                        use_stack = false;
                    }
                }
                if use_stack {
                    stack_args += classes.len();
                }
            }
            _ => {
                if int_regs < int_regs_available {
                    int_regs += 1;
                } else {
                    stack_args += 1;
                }
            }
        }
    }

    (int_regs, double_regs, stack_args)
}
