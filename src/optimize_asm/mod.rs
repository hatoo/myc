use std::collections::{BTreeSet, HashMap, HashSet};

use ecow::EcoString;

use crate::{
    ast::{BaseType, FunType, VarType},
    codegen::{
        classify_struct, is_return_in_memory, Class, Instruction, Operand, Pseudo, Register,
    },
    control_flow::Cfg,
    semantics::type_check::{Attr, SymbolTable},
};

mod dsu;
mod liveness_analysis;

// R10 and R11 are used as temporary registers in the code generator
const FREE_INT_REGISTERS: [Register; 12] = [
    Register::Ax,
    Register::Bx,
    Register::Cx,
    Register::Dx,
    Register::Di,
    Register::Si,
    Register::R8,
    Register::R9,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

const FREE_DOUBLE_REGISTERS: [Register; 14] = [
    Register::Xmm(0),
    Register::Xmm(1),
    Register::Xmm(2),
    Register::Xmm(3),
    Register::Xmm(4),
    Register::Xmm(5),
    Register::Xmm(6),
    Register::Xmm(7),
    Register::Xmm(8),
    Register::Xmm(9),
    Register::Xmm(10),
    Register::Xmm(11),
    Register::Xmm(12),
    Register::Xmm(13),
];

pub fn register_allocation(
    program: &mut [Instruction],
    return_registers: &[Register],
    symbol_table: &SymbolTable,
    aliased_vals: &HashSet<EcoString>,
    mode: ColoringMode,
) -> HashSet<Register> {
    let mut graph = loop {
        let mut graph =
            ColoringGraph::new(program, symbol_table, aliased_vals, mode, return_registers);
        if graph.coalesce(program) {
            break graph;
        }
    };

    graph.color_graph();
    let (register_map, callee_saved) = graph.create_register_map();

    let replace = |op: &mut Operand| {
        if let Operand::Pseudo(Pseudo::Mem { name, offset: _ }) = op {
            if let Some(reg) = register_map.get(name) {
                *op = Operand::Reg(*reg);
            }
        }
    };

    for inst in program {
        match inst {
            Instruction::Mov { src, dst, .. } => {
                replace(src);
                replace(dst);

                if let (Operand::Reg(a), Operand::Reg(b)) = (src, dst) {
                    if a == b {
                        *inst = Instruction::Nop;
                    }
                }
            }
            Instruction::MovZeroExtend { src, dst, .. } => {
                replace(src);
                replace(dst);
            }
            Instruction::Movsx { src, dst, .. } => {
                replace(src);
                replace(dst);
            }
            Instruction::Binary { lhs, rhs, .. } => {
                replace(lhs);
                replace(rhs);
            }
            Instruction::Unary { src, .. } => {
                replace(src);
            }
            Instruction::Cmp(_, v1, v2) => {
                replace(v1);
                replace(v2);
            }
            Instruction::SetCc(_, dst) => {
                replace(dst);
            }
            Instruction::Push(op) => {
                replace(op);
            }
            Instruction::Idiv(_, divisor) => {
                replace(divisor);
            }
            Instruction::Cdq(_) => {}
            Instruction::Call(op) => {
                replace(op);
            }
            Instruction::Lea { src, dst } => {
                replace(src);
                replace(dst);
            }
            Instruction::Div(_, op) => {
                replace(op);
            }
            Instruction::Jmp(_) => {}
            Instruction::JmpCc(_, _) => {}
            Instruction::Label(_) => {}
            Instruction::Ret => {}
            Instruction::Pop(_) => {}
            Instruction::Cvttsd2si { src, dst, .. } => {
                replace(src);
                replace(dst);
            }
            Instruction::Cvtsi2sd { src, dst, .. } => {
                replace(src);
                replace(dst);
            }
            Instruction::Nop => {}
        }
    }

    callee_saved
}

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
            Operand::Pseudo(Pseudo::Mem { name, offset: 0 }) => Ok(NodeId::Pseudo(name.clone())),
            _ => Err(()),
        }
    }
}

impl Into<Operand> for NodeId {
    fn into(self) -> Operand {
        match self {
            NodeId::Register(reg) => Operand::Reg(reg),
            NodeId::Pseudo(name) => Operand::Pseudo(Pseudo::Mem { name, offset: 0 }),
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
    mode: ColoringMode,
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
            if FREE_INT_REGISTERS.contains(reg) {
                return true;
            }
        }
    }

    false
}

fn is_double_scalar(n: &NodeId, symbol_table: &SymbolTable) -> bool {
    match n {
        NodeId::Pseudo(name) => {
            if let Attr::Local(ty) = &symbol_table[name] {
                if *ty == VarType::Base(BaseType::Double) {
                    return true;
                }
            }
        }
        NodeId::Register(reg) => {
            if FREE_DOUBLE_REGISTERS.contains(reg) {
                return true;
            }
        }
    }

    false
}

#[derive(Debug, Clone, Copy)]
pub enum ColoringMode {
    Int,
    Double,
}

impl<'a> ColoringGraph<'a> {
    fn new(
        program: &[Instruction],
        symbol_table: &'a SymbolTable,
        aliased_vals: &HashSet<EcoString>,
        mode: ColoringMode,
        return_registers: &[Register],
    ) -> Self {
        let mut me = Self {
            map: HashMap::new(),
            symbol_table,
            mode,
        };
        me.add_base_registers();

        let cfg = Cfg::new(program);
        me.collect_pseudo_vars(program);

        for name in aliased_vals {
            me.map.remove(&NodeId::Pseudo(name.clone()));
        }

        me.add_spill_costs(program);
        me.add_edges(&cfg, return_registers);
        me
    }

    fn free_registers(&self) -> &'static [Register] {
        match self.mode {
            ColoringMode::Int => &FREE_INT_REGISTERS,
            ColoringMode::Double => &FREE_DOUBLE_REGISTERS,
        }
    }

    fn add_base_registers(&mut self) {
        for &reg in self.free_registers() {
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

        for &reg in self.free_registers() {
            for &neighbor in self.free_registers() {
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

    fn spill_cost(&self, node_id: &NodeId) -> f32 {
        let node = &self.map[node_id];

        node.spill_cost
            / node
                .neighbors
                .iter()
                .filter(|n| !self.map[n].pruned)
                .count()
                .max(1) as f32
    }

    fn color_graph(&mut self) {
        // TODO: optimize
        let k = self.free_registers().len();

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
                .min_by(|n1, n2| self.spill_cost(&n1.id).total_cmp(&self.spill_cost(&n2.id)))
                .unwrap()
                .id
                .clone()
        };

        let node = self.map.get_mut(&chosen_id).unwrap();
        node.pruned = true;

        self.color_graph();

        let mut colors = (0..k).collect::<BTreeSet<_>>();

        let node = self.map.get(&chosen_id).unwrap();
        for neighbor in &node.neighbors {
            if let Some(neighbor) = self.map.get(neighbor) {
                if let Some(color) = neighbor.color {
                    colors.remove(&color);
                }
            }
        }

        if !colors.is_empty() {
            match chosen_id {
                NodeId::Register(r) if r.is_callee_saved() => {
                    let color = colors.first().unwrap().clone();
                    self.map.get_mut(&chosen_id).unwrap().color = Some(color);
                }
                _ => {
                    let color = colors.last().unwrap().clone();
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
                    color_map.insert(color, *r);
                }
            }
        }

        let mut register_map = HashMap::new();
        let mut callee_saved = HashSet::new();

        for node in self.map.values() {
            if let NodeId::Pseudo(name) = &node.id {
                if let Some(color) = node.color {
                    let hardreg = color_map[&color];
                    register_map.insert(name.clone(), hardreg);
                    if hardreg.is_callee_saved() {
                        callee_saved.insert(hardreg);
                    }
                }
            }
        }

        (register_map, callee_saved)
    }

    fn add_var(&mut self, n: NodeId) {
        match self.mode {
            ColoringMode::Int => {
                if is_int_scalar(&n, self.symbol_table) {
                    self.map.entry(n.clone()).or_insert(Node::new(n));
                }
            }
            ColoringMode::Double => {
                if is_double_scalar(&n, self.symbol_table) {
                    self.map.entry(n.clone()).or_insert(Node::new(n));
                }
            }
        }
    }

    fn check_node_id(&self, n: &NodeId) -> bool {
        match self.mode {
            ColoringMode::Int => is_int_scalar(n, self.symbol_table),
            ColoringMode::Double => is_double_scalar(n, self.symbol_table),
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

    fn add_edges(&mut self, cfg: &Cfg<Instruction>, return_registers: &[Register]) {
        let mut annotation = liveness_analysis::Annotation::new(return_registers);

        annotation.iterate(cfg, self.symbol_table);

        for node in cfg.nodes.values() {
            let annotation = annotation.instruction_annotation.get(&node.id).unwrap();

            for (inst, live) in node.instructions.iter().zip(annotation.iter()) {
                let (_used, updated) = find_used_and_updated(inst, self.symbol_table);

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
            let (used, updated) = find_used_and_updated(inst, self.symbol_table);
            for &op in used.iter().chain(updated.iter()) {
                if let Ok(id) = op.try_into() {
                    self.add_var(id);
                }
            }
        }
    }

    fn add_spill_costs(&mut self, insts: &[Instruction]) {
        for inst in insts {
            let (used, updated) = find_used_and_updated(inst, self.symbol_table);
            for &op in used.iter().chain(updated.iter()) {
                if let Ok(id) = op.try_into() {
                    self.increment_spill_cost(&id);
                }
            }
        }
    }

    fn coalesce(&mut self, insts: &mut [Instruction]) -> bool {
        let mut dsu = dsu::DisjointSet::default();

        for i in insts.iter() {
            match i {
                Instruction::Mov { src, dst, .. } => {
                    if let (Ok(src), Ok(dst)) = (src.try_into(), dst.try_into()) {
                        let src = dsu.find(&src);
                        let dst = dsu.find(&dst);

                        if self.check_node_id(&src) && self.check_node_id(&dst) {
                            if self.map.contains_key(&src)
                                && self.map.contains_key(&dst)
                                && src != dst
                                && !self.are_neighbors(&src, &dst)
                                && self.conservative_coaleasceble(&src, &dst)
                            {
                                let (to_keep, to_merge) = if let NodeId::Register(_) = src {
                                    (src, dst)
                                } else {
                                    (dst, src)
                                };

                                dsu.merge(&to_merge, &to_keep);
                                self.update_graph(&to_merge, &to_keep);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let mut replace = |op: &mut Operand| {
            if let Ok(node_id) = (&*op).try_into() {
                *op = dsu.find(&node_id).into();
            }
        };
        for inst in insts {
            match inst {
                Instruction::Mov { src, dst, .. } => {
                    replace(src);
                    replace(dst);

                    if let (Operand::Reg(a), Operand::Reg(b)) = (src, dst) {
                        if a == b {
                            *inst = Instruction::Nop;
                        }
                    }
                }
                Instruction::MovZeroExtend { src, dst, .. } => {
                    replace(src);
                    replace(dst);
                }
                Instruction::Movsx { src, dst, .. } => {
                    replace(src);
                    replace(dst);
                }
                Instruction::Binary { lhs, rhs, .. } => {
                    replace(lhs);
                    replace(rhs);
                }
                Instruction::Unary { src, .. } => {
                    replace(src);
                }
                Instruction::Cmp(_, v1, v2) => {
                    replace(v1);
                    replace(v2);
                }
                Instruction::SetCc(_, dst) => {
                    replace(dst);
                }
                Instruction::Push(op) => {
                    replace(op);
                }
                Instruction::Idiv(_, divisor) => {
                    replace(divisor);
                }
                Instruction::Cdq(_) => {}
                Instruction::Call(op) => {
                    replace(op);
                }
                Instruction::Lea { src, dst } => {
                    replace(src);
                    replace(dst);
                }
                Instruction::Div(_, op) => {
                    replace(op);
                }
                Instruction::Jmp(_) => {}
                Instruction::JmpCc(_, _) => {}
                Instruction::Label(_) => {}
                Instruction::Ret => {}
                Instruction::Pop(_) => {}
                Instruction::Cvttsd2si { src, dst, .. } => {
                    replace(src);
                    replace(dst);
                }
                Instruction::Cvtsi2sd { src, dst, .. } => {
                    replace(src);
                    replace(dst);
                }
                Instruction::Nop => {}
            }
        }

        dsu.is_empty()
    }

    fn update_graph(&mut self, x: &NodeId, y: &NodeId) {
        let to_remove = self.map.remove(x).unwrap();
        for neighbor in &to_remove.neighbors {
            self.add_edge(y.clone(), neighbor.clone());
            self.map.get_mut(neighbor).unwrap().neighbors.remove(x);
        }
    }

    fn are_neighbors(&self, x: &NodeId, y: &NodeId) -> bool {
        self.map[x].neighbors.contains(y)
    }

    fn conservative_coaleasceble(&self, src: &NodeId, dst: &NodeId) -> bool {
        if self.briggs_test(src, dst) {
            return true;
        }
        match (&src, &dst) {
            (NodeId::Register(src), NodeId::Pseudo(dst)) => self.george_test(*src, dst.clone()),
            (NodeId::Pseudo(src), NodeId::Register(dst)) => self.george_test(*dst, src.clone()),
            _ => false,
        }
    }

    fn briggs_test(&self, x: &NodeId, y: &NodeId) -> bool {
        let mut significant_neighbors = 0;

        let x_node = &self.map[x];
        let y_node = &self.map[y];

        let combined_neighbors = x_node
            .neighbors
            .iter()
            .chain(y_node.neighbors.iter())
            .cloned()
            .collect::<HashSet<_>>();

        for n in combined_neighbors {
            let node = &self.map[&n];
            let mut degree = node.neighbors.len();
            if self.are_neighbors(&n, x) && self.are_neighbors(&n, y) {
                degree -= 1;
            }
            if degree >= self.free_registers().len() {
                significant_neighbors += 1;
            }
        }

        significant_neighbors < self.free_registers().len()
    }

    fn george_test(&self, hardreg: Register, pseudoreg: EcoString) -> bool {
        let pseudo_node = &self.map[&NodeId::Pseudo(pseudoreg)];

        for n in &pseudo_node.neighbors {
            if self.are_neighbors(n, &NodeId::Register(hardreg)) {
                continue;
            }
            let node = &self.map[n];
            if node.neighbors.len() < self.free_registers().len() {
                continue;
            }
            return false;
        }

        true
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

        Instruction::Div(_, divisor) | Instruction::Idiv(_, divisor) => (
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
            let name = match op {
                Operand::Pseudo(Pseudo::Mem { name, .. }) => name,
                Operand::Plt(name) => name,
                Operand::GotPcrel(name) => name,
                _ => panic!("Invalid operand {:?}", op),
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
                    &Operand::Reg(Register::Xmm(0)),
                    &Operand::Reg(Register::Xmm(1)),
                    &Operand::Reg(Register::Xmm(2)),
                    &Operand::Reg(Register::Xmm(3)),
                    &Operand::Reg(Register::Xmm(4)),
                    &Operand::Reg(Register::Xmm(5)),
                    &Operand::Reg(Register::Xmm(6)),
                    &Operand::Reg(Register::Xmm(7)),
                    &Operand::Reg(Register::Xmm(8)),
                    &Operand::Reg(Register::Xmm(9)),
                    &Operand::Reg(Register::Xmm(10)),
                    &Operand::Reg(Register::Xmm(11)),
                    &Operand::Reg(Register::Xmm(12)),
                    &Operand::Reg(Register::Xmm(13)),
                    &Operand::Reg(Register::Xmm(14)),
                ],
            )
        }
        Instruction::Nop
        | Instruction::Pop(_)
        | Instruction::Jmp(_)
        | Instruction::JmpCc(_, _)
        | Instruction::Ret
        | Instruction::Label(_) => (vec![], vec![]),

        Instruction::Lea { src, dst } => (vec![src], vec![dst]),
        Instruction::Cvttsd2si { src, dst, .. } | Instruction::Cvtsi2sd { src, dst, .. } => {
            (vec![src], vec![dst])
        }
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
                let classes = classify_struct(structure, symbol_table);
                let mut use_stack = true;

                if classes[0] != Class::Memory {
                    let mut tentative_ints = 0;
                    let mut tentative_doubles = 0;
                    for &class in &classes {
                        if class == Class::Sse {
                            tentative_doubles += 1;
                        } else {
                            tentative_ints += 1;
                        }
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
