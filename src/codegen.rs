use core::panic;
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Display,
    hash::Hash,
};

use ecow::EcoString;

use crate::{
    ast::{self, BaseType, Const, VarType},
    math::round_up,
    optimize_asm::register_allocation,
    semantics::{
        self,
        type_check::{self, Attr, StructDef, SymbolTable},
    },
    tacky::{self, Val},
};

#[derive(Debug)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyType {
    Byte,
    LongWord,
    QuadWord,
    Double,
    ByteArray { size: usize, alignment: usize },
}

impl AssemblyType {
    pub fn suffix(&self) -> &'static str {
        match self {
            AssemblyType::Byte => "b",
            AssemblyType::LongWord => "l",
            AssemblyType::QuadWord => "q",
            AssemblyType::Double => "sd",
            AssemblyType::ByteArray { .. } => {
                panic!("This variant must not be appeared in assembly emit stage");
            }
        }
    }
}

#[derive(Debug)]
pub enum TopLevel {
    StaticConstant(StaticConstant),
    StaticVariable(StaticVariable),
    Function(Function),
}

#[derive(Debug)]
pub struct StaticConstant {
    pub name: EcoString,
    pub alignment: usize,
    pub init: semantics::type_check::StaticInit,
}

#[derive(Debug)]
pub struct StaticVariable {
    pub global: bool,
    pub name: EcoString,
    pub alignment: usize,
    pub init: Vec<semantics::type_check::StaticInit>,
}

#[derive(Debug)]
pub struct Function {
    pub global: bool,
    pub name: EcoString,
    pub body: Vec<Instruction>,
    pub callee_saved: Vec<Register>,
    pub stack_size: usize,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Nop,
    Mov {
        ty: AssemblyType,
        src: Operand,
        dst: Operand,
    },
    Movsx {
        src_type: AssemblyType,
        dst_type: AssemblyType,
        src: Operand,
        dst: Operand,
    },
    MovZeroExtend {
        src_type: AssemblyType,
        dst_type: AssemblyType,
        src: Operand,
        dst: Operand,
    },
    Lea {
        src: Operand,
        dst: Operand,
    },
    Unary {
        op: UnaryOp,
        ty: AssemblyType,
        src: Operand,
    },
    Binary {
        op: BinaryOp,
        ty: AssemblyType,
        lhs: Operand,
        rhs: Operand,
    },
    Cmp(AssemblyType, Operand, Operand),
    Idiv(AssemblyType, Operand),
    Div(AssemblyType, Operand),
    Cdq(AssemblyType),
    Jmp(EcoString),
    JmpCc(CondCode, EcoString),
    SetCc(CondCode, Operand),
    Label(EcoString),
    Ret,
    Push(Operand),
    Pop(Register),
    Call(Operand),
    Cvttsd2si {
        ty: AssemblyType,
        src: Operand,
        dst: Operand,
    },
    Cvtsi2sd {
        ty: AssemblyType,
        src: Operand,
        dst: Operand,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
    Shr,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mult,
    DivDouble,
    And,
    Or,
    Xor,
    Shl,
    ShrTwo,
}

#[derive(Debug, Clone)]
pub enum Pseudo {
    // Must be placed in read only section
    Double { value: f64, alignment: usize },
    Mem { name: EcoString, offset: usize },
}

impl Pseudo {
    fn var(name: EcoString) -> Self {
        Pseudo::Mem { name, offset: 0 }
    }
}

#[derive(Debug, Clone)]
pub enum Operand {
    Imm(u64),
    Reg(Register),
    Pseudo(Pseudo),
    Memory(Register, i32),
    Data(EcoString, i32),
    Plt(EcoString),
    GotPcrel(EcoString),
    Indexed {
        base: Register,
        index: Register,
        scale: usize,
    },
}

impl Operand {
    fn stack(offset: i32) -> Self {
        Operand::Memory(Register::BP, offset)
    }

    fn sized(&self, size: AssemblyType) -> SizedOperand {
        SizedOperand { ty: size, op: self }
    }

    fn offset(&self, offset: i32) -> Self {
        match self {
            Operand::Pseudo(Pseudo::Mem { name, offset: off }) => Operand::Pseudo(Pseudo::Mem {
                name: name.clone(),
                offset: *off + offset as usize,
            }),
            Operand::Memory(base, off) => Operand::Memory(*base, *off + offset),
            _ => panic!(
                "You can't offset this operand {:?}. This is your responsibility.",
                self
            ),
        }
    }
}

impl From<tacky::Val> for Operand {
    fn from(val: tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(Const::Double(d)) => Operand::Pseudo(Pseudo::Double {
                value: d,
                alignment: 8,
            }),
            tacky::Val::Constant(imm) => Operand::Imm(imm.get_ulong()),
            tacky::Val::Var(var) => Operand::Pseudo(Pseudo::var(var)),
        }
    }
}

impl<'a> From<&'a tacky::Val> for Operand {
    fn from(val: &'a tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(Const::Double(d)) => Operand::Pseudo(Pseudo::Double {
                value: *d,
                alignment: 8,
            }),
            tacky::Val::Constant(imm) => Operand::Imm(imm.get_ulong()),
            tacky::Val::Var(var) => Operand::Pseudo(Pseudo::var(var.clone())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Register {
    Ax,
    Bx,
    Cx,
    Dx,
    Di,
    Si,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    SP,
    BP,
    Xmm(u8),
}

impl Register {
    pub fn is_callee_saved(&self) -> bool {
        matches!(
            self,
            Register::Bx
                | Register::BP
                | Register::R12
                | Register::R13
                | Register::R14
                | Register::R15
        )
    }
}

pub enum RegisterSize<'a> {
    Byte(&'a Register),
    Dword(&'a Register),
    Qword(&'a Register),
}

#[derive(Debug, Clone)]
pub enum CondCode {
    E,
    Ne,
    G,
    Ge,
    L,
    Le,
    A,
    Ae,
    B,
    Be,
}

/*
pub enum AsmEntry {
    Obj { ty: AssemblyType, is_static: bool },
    Fun { is_defined: bool },
}
*/

#[derive(Debug, Default)]
pub struct ConstTable {
    counter: usize,
    table: HashMap<(u64, usize), EcoString>,
}

impl ConstTable {
    fn label(&mut self, double: f64, alignment: usize) -> EcoString {
        let key = (double.to_bits(), alignment);

        match self.table.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let label = EcoString::from(format!("double{}", self.counter));
                self.counter += 1;
                entry.insert(label.clone());
                label
            }
        }
    }
}

fn divide_into_assembly_sizes(size: usize) -> impl Iterator<Item = (AssemblyType, usize)> {
    let mut size = size;
    let mut offset = 0;
    std::iter::from_fn(move || {
        if size == 0 {
            None
        } else if size >= 8 {
            size -= 8;
            let ret = Some((AssemblyType::QuadWord, offset));
            offset += 8;
            ret
        } else if size >= 4 {
            size -= 4;
            let ret = Some((AssemblyType::LongWord, offset));
            offset += 4;
            ret
        } else {
            size -= 1;
            let ret = Some((AssemblyType::Byte, offset));
            offset += 1;
            ret
        }
    })
}

#[derive(Debug)]
pub struct CodeGen<'a> {
    const_table: ConstTable,
    label_counter: usize,
    symbol_table: &'a SymbolTable,
}

impl<'a> CodeGen<'a> {
    pub fn new(symbol_table: &'a SymbolTable) -> Self {
        Self {
            const_table: ConstTable::default(),
            label_counter: 0,
            symbol_table,
        }
    }

    fn gen_label(&mut self, prefix: &str) -> EcoString {
        let label = format!("codegen.{}.{}", prefix, self.label_counter);
        self.label_counter += 1;
        EcoString::from(label)
    }

    fn val_asm_type(&self, val: &Val) -> AssemblyType {
        asm_type(&val.ty(self.symbol_table), self.symbol_table)
    }

    pub fn gen_program(
        &mut self,
        program: &tacky::Program,
        enable_register_relocation: bool,
    ) -> Program {
        let top_levels: Vec<_> = program
            .top_levels
            .iter()
            .map(|item| match item {
                tacky::TopLevelItem::StaticVariable(tacky::StaticVariable {
                    global,
                    name,
                    alignment,
                    init,
                }) => TopLevel::StaticVariable(StaticVariable {
                    global: *global,
                    name: name.clone(),
                    alignment: *alignment,
                    init: init.clone(),
                }),
                tacky::TopLevelItem::Function(function) => {
                    TopLevel::Function(self.gen_function(function, enable_register_relocation))
                }
                tacky::TopLevelItem::StaticConstant(tacky::StaticConstant { name, ty, init }) => {
                    TopLevel::StaticConstant(StaticConstant {
                        name: name.clone(),
                        alignment: self.symbol_table.alignment(ty),
                        init: init.clone(),
                    })
                }
            })
            .collect();

        Program {
            top_levels: self
                .const_table
                .table
                .iter()
                .map(|((value, align), v)| {
                    TopLevel::StaticConstant(StaticConstant {
                        name: v.clone(),
                        alignment: *align,
                        init: semantics::type_check::StaticInit::Double(f64::from_bits(*value)),
                    })
                })
                .chain(top_levels)
                .collect(),
        }
    }
    fn copy_bytes_to_reg(
        &self,
        op: &Operand,
        dst_reg: Register,
        byte_count: usize,
        body: &mut Vec<Instruction>,
    ) {
        let mut offset = byte_count as i32 - 1;
        while offset >= 0 {
            let src_byte = op.offset(offset);
            body.push(Instruction::Mov {
                ty: AssemblyType::Byte,
                src: src_byte,
                dst: Operand::Reg(dst_reg),
            });
            if offset > 0 {
                body.push(Instruction::Binary {
                    op: BinaryOp::Shl,
                    ty: AssemblyType::QuadWord,
                    lhs: Operand::Imm(8),
                    rhs: Operand::Reg(dst_reg),
                });
            }
            offset -= 1;
        }
    }
    fn copy_bytes_from_reg(
        &self,
        op: &Operand,
        src_reg: Register,
        byte_count: usize,
        body: &mut Vec<Instruction>,
    ) {
        let mut offset = 0;
        while offset < byte_count {
            let dst_byte = op.offset(offset as i32);
            body.push(Instruction::Mov {
                ty: AssemblyType::Byte,
                src: Operand::Reg(src_reg),
                dst: dst_byte,
            });
            if offset < byte_count - 1 {
                body.push(Instruction::Binary {
                    op: BinaryOp::ShrTwo,
                    ty: AssemblyType::QuadWord,
                    lhs: Operand::Imm(8),
                    rhs: Operand::Reg(src_reg),
                });
            }
            offset += 1;
        }
    }

    fn gen_function(
        &mut self,
        function: &tacky::Function,
        enable_register_relocation: bool,
    ) -> Function {
        let aliased_vals = function
            .body
            .iter()
            .flat_map(|inst| {
                if let tacky::Instruction::GetAddress { src, .. } = inst {
                    Some(src.var().clone())
                } else {
                    None
                }
            })
            .collect();
        let mut body = Vec::new();

        let semantics::type_check::Attr::Fun { ty, .. } = &self.symbol_table[&function.name] else {
            unreachable!()
        };

        let return_in_memory = is_return_in_memory(&ty.ret, self.symbol_table);

        if return_in_memory {
            body.push(Instruction::Mov {
                ty: AssemblyType::QuadWord,
                src: Operand::Reg(Register::Di),
                dst: Operand::Memory(Register::BP, -8),
            });
        }

        let (int_reg_args, double_reg_args, stack_args) = self.classify_parameters(
            function.params.iter().map(|name| Val::Var(name.clone())),
            return_in_memory,
        );

        let int_regs = if return_in_memory {
            &PARAM_REGISTERS[1..]
        } else {
            &PARAM_REGISTERS
        };

        for (i, (asm_ty, op)) in int_reg_args.into_iter().enumerate() {
            if let AssemblyType::ByteArray { size, .. } = asm_ty {
                self.copy_bytes_from_reg(&op, int_regs[i], size, &mut body);
            } else {
                body.push(Instruction::Mov {
                    ty: asm_ty,
                    src: Operand::Reg(int_regs[i]),
                    dst: op,
                });
            }
        }

        for (i, (_asm_ty, op)) in double_reg_args.into_iter().enumerate() {
            body.push(Instruction::Mov {
                ty: AssemblyType::Double,
                src: Operand::Reg(Register::Xmm(i as _)),
                dst: op,
            });
        }

        for (i, (asm_ty, op)) in stack_args.into_iter().enumerate() {
            if let AssemblyType::ByteArray { size, .. } = asm_ty {
                for (asm_ty, offset) in divide_into_assembly_sizes(size) {
                    body.push(Instruction::Mov {
                        ty: asm_ty,
                        src: Operand::Memory(Register::BP, (16 + i * 8 + offset) as i32),
                        dst: op.offset(offset as i32),
                    });
                }
            } else {
                body.push(Instruction::Mov {
                    ty: asm_ty,
                    src: Operand::stack((16 + i * 8) as i32),
                    dst: op,
                });
            }
        }

        for inst in &function.body {
            match inst {
                tacky::Instruction::Nop => {}
                tacky::Instruction::Return(val) => {
                    if let Some(val) = val {
                        let (int_retvals, double_retvals, return_in_memory) =
                            self.classify_return_value(val);

                        if return_in_memory {
                            body.push(Instruction::Mov {
                                ty: AssemblyType::QuadWord,
                                src: Operand::Memory(Register::BP, -8),
                                dst: Operand::Reg(Register::Ax),
                            });
                            let return_storage = Operand::Memory(Register::Ax, 0);
                            let ret_operand: Operand = val.into();
                            let size = self.symbol_table.size(&val.ty(self.symbol_table));
                            for (asm_ty, offset) in divide_into_assembly_sizes(size) {
                                body.push(Instruction::Mov {
                                    ty: asm_ty,
                                    src: ret_operand.offset(offset as i32),
                                    dst: return_storage.offset(offset as i32),
                                });
                            }
                        } else {
                            let int_ret_regs = [Register::Ax, Register::Dx];

                            for (i, (asm_ty, op)) in int_retvals.into_iter().enumerate() {
                                if let AssemblyType::ByteArray { size, .. } = asm_ty {
                                    self.copy_bytes_to_reg(&op, int_ret_regs[i], size, &mut body);
                                } else {
                                    body.push(Instruction::Mov {
                                        ty: asm_ty,
                                        src: op,
                                        dst: Operand::Reg(int_ret_regs[i]),
                                    });
                                }
                            }

                            for (i, op) in double_retvals.into_iter().enumerate() {
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Double,
                                    src: op,
                                    dst: Operand::Reg(Register::Xmm(i as _)),
                                });
                            }
                        }
                    }
                    body.push(Instruction::Ret);
                }
                tacky::Instruction::Unary { op, src, dst } => {
                    let src_ty = src.ty(self.symbol_table);
                    let dst_ty = dst.ty(self.symbol_table);

                    match (src_ty, op) {
                        (VarType::Base(BaseType::Double), tacky::UnaryOp::Not) => {
                            body.push(Instruction::Binary {
                                op: BinaryOp::Xor,
                                ty: AssemblyType::Double,
                                lhs: Operand::Reg(Register::Xmm(0)),
                                rhs: Operand::Reg(Register::Xmm(0)),
                            });
                            body.push(Instruction::Cmp(
                                AssemblyType::Double,
                                src.into(),
                                Operand::Reg(Register::Xmm(0)),
                            ));
                            body.push(Instruction::Mov {
                                ty: asm_type(&dst_ty, self.symbol_table),
                                src: Operand::Imm(0),
                                dst: dst.into(),
                            });
                            body.push(Instruction::SetCc(CondCode::E, dst.into()));
                        }
                        (VarType::Base(BaseType::Double), tacky::UnaryOp::Negate) => {
                            body.push(Instruction::Mov {
                                ty: AssemblyType::Double,
                                src: src.into(),
                                dst: dst.into(),
                            });
                            body.push(Instruction::Binary {
                                op: BinaryOp::Xor,
                                ty: AssemblyType::Double,
                                lhs: Operand::Pseudo(Pseudo::Double {
                                    value: -0.0,
                                    alignment: 16,
                                }),
                                rhs: dst.into(),
                            });
                        }
                        _ => {
                            enum Unary {
                                Simple(UnaryOp),
                                Not,
                            }

                            let op = match op {
                                tacky::UnaryOp::Negate => Unary::Simple(UnaryOp::Neg),
                                tacky::UnaryOp::Complement => Unary::Simple(UnaryOp::Not),
                                tacky::UnaryOp::Not => Unary::Not,
                            };

                            match op {
                                Unary::Simple(op) => {
                                    body.push(Instruction::Mov {
                                        ty: self.val_asm_type(src),
                                        src: src.into(),
                                        dst: dst.into(),
                                    });
                                    body.push(Instruction::Unary {
                                        ty: self.val_asm_type(src),
                                        op,
                                        src: dst.into(),
                                    });
                                }
                                Unary::Not => {
                                    body.push(Instruction::Cmp(
                                        self.val_asm_type(src),
                                        Operand::Imm(0),
                                        src.into(),
                                    ));
                                    body.push(Instruction::Mov {
                                        ty: asm_type(&dst_ty, self.symbol_table),
                                        src: Operand::Imm(0),
                                        dst: dst.into(),
                                    });
                                    body.push(Instruction::SetCc(CondCode::E, dst.into()));
                                }
                            }
                        }
                    }
                }
                tacky::Instruction::Binary { op, lhs, rhs, dst } => {
                    enum Binary {
                        Simple(BinaryOp),
                        Divide,
                        Remainder,
                        Compare(CondCode),
                    }

                    let ty = lhs.ty(self.symbol_table);

                    let op = match op {
                        tacky::BinaryOp::Add => Binary::Simple(BinaryOp::Add),
                        tacky::BinaryOp::Subtract => Binary::Simple(BinaryOp::Sub),
                        tacky::BinaryOp::Multiply => Binary::Simple(BinaryOp::Mult),
                        tacky::BinaryOp::Divide => Binary::Divide,
                        tacky::BinaryOp::Remainder => Binary::Remainder,
                        tacky::BinaryOp::Equal => Binary::Compare(CondCode::E),
                        tacky::BinaryOp::NotEqual => Binary::Compare(CondCode::Ne),
                        tacky::BinaryOp::LessThan => Binary::Compare(if ty.is_signed() {
                            CondCode::L
                        } else {
                            CondCode::B
                        }),
                        tacky::BinaryOp::LessOrEqual => Binary::Compare(if ty.is_signed() {
                            CondCode::Le
                        } else {
                            CondCode::Be
                        }),
                        tacky::BinaryOp::GreaterThan => Binary::Compare(if ty.is_signed() {
                            CondCode::G
                        } else {
                            CondCode::A
                        }),
                        tacky::BinaryOp::GreaterOrEqual => Binary::Compare(if ty.is_signed() {
                            CondCode::Ge
                        } else {
                            CondCode::Ae
                        }),
                    };

                    match op {
                        Binary::Simple(op) => {
                            body.push(Instruction::Mov {
                                ty: self.val_asm_type(lhs),
                                src: lhs.into(),
                                dst: dst.into(),
                            });
                            body.push(Instruction::Binary {
                                ty: self.val_asm_type(lhs),
                                op,
                                lhs: rhs.into(),
                                rhs: dst.into(),
                            });
                        }
                        Binary::Divide => {
                            let ty = lhs.ty(self.symbol_table);
                            if ty == VarType::Base(BaseType::Double) {
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Double,
                                    src: lhs.into(),
                                    dst: dst.into(),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::DivDouble,
                                    ty: AssemblyType::Double,
                                    lhs: rhs.into(),
                                    rhs: dst.into(),
                                });
                                /*
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Double,
                                    src: rhs.into(),
                                    dst: Operand::Reg(Register::Xmm(14)),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::DivDouble,
                                    ty: AssemblyType::Double,
                                    lhs: lhs.into(),
                                    rhs: Operand::Reg(Register::Xmm(14)),
                                });
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Double,
                                    src: Operand::Reg(Register::Xmm(14)),
                                    dst: dst.into(),
                                });
                                */
                            } else if ty.is_signed() {
                                let ty = asm_type(&ty, self.symbol_table);
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Cdq(ty));
                                body.push(Instruction::Idiv(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Ax),
                                    dst: dst.into(),
                                });
                            } else {
                                let ty = asm_type(&ty, self.symbol_table);
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Imm(0),
                                    dst: Operand::Reg(Register::Dx),
                                });
                                body.push(Instruction::Div(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Ax),
                                    dst: dst.into(),
                                });
                            }
                        }
                        Binary::Remainder => {
                            let ty = lhs.ty(self.symbol_table);
                            if ty.is_signed() {
                                let ty = asm_type(&ty, self.symbol_table);
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Cdq(ty));
                                body.push(Instruction::Idiv(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Dx),
                                    dst: dst.into(),
                                });
                            } else {
                                let ty = asm_type(&ty, self.symbol_table);
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Imm(0),
                                    dst: Operand::Reg(Register::Dx),
                                });
                                body.push(Instruction::Div(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Dx),
                                    dst: dst.into(),
                                });
                            }
                        }
                        Binary::Compare(cond) => {
                            body.push(Instruction::Cmp(
                                self.val_asm_type(lhs),
                                rhs.into(),
                                lhs.into(),
                            ));
                            body.push(Instruction::Mov {
                                ty: self.val_asm_type(dst),
                                src: Operand::Imm(0),
                                dst: dst.into(),
                            });
                            body.push(Instruction::SetCc(cond, dst.into()));
                        }
                    }
                }
                tacky::Instruction::Copy { src, dst } => match src {
                    Val::Constant(_) => {
                        body.push(Instruction::Mov {
                            ty: self.val_asm_type(src),
                            src: src.into(),
                            dst: dst.into(),
                        });
                    }
                    Val::Var(src_name) => {
                        let size = self.symbol_table.size(&src.ty(self.symbol_table));
                        let dst = dst.var();
                        for (asm, offset) in divide_into_assembly_sizes(size) {
                            body.push(Instruction::Mov {
                                ty: asm,
                                src: Operand::Pseudo(Pseudo::Mem {
                                    name: src_name.clone(),
                                    offset,
                                }),
                                dst: Operand::Pseudo(Pseudo::Mem {
                                    name: dst.clone(),
                                    offset,
                                }),
                            });
                        }
                    }
                },
                tacky::Instruction::Jump(label) => {
                    body.push(Instruction::Jmp(label.clone()));
                }
                tacky::Instruction::JumpIfZero { src, dst } => {
                    if src.ty(self.symbol_table) == VarType::Base(BaseType::Double) {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Xor,
                            ty: AssemblyType::Double,
                            lhs: Operand::Reg(Register::Xmm(0)),
                            rhs: Operand::Reg(Register::Xmm(0)),
                        });
                        body.push(Instruction::Cmp(
                            AssemblyType::Double,
                            src.into(),
                            Operand::Reg(Register::Xmm(0)),
                        ));
                        body.push(Instruction::JmpCc(CondCode::E, dst.clone()));
                    } else {
                        body.push(Instruction::Cmp(
                            self.val_asm_type(src),
                            Operand::Imm(0),
                            src.into(),
                        ));
                        body.push(Instruction::JmpCc(CondCode::E, dst.clone()));
                    }
                }
                tacky::Instruction::JumpIfNotZero { src, dst } => {
                    if src.ty(self.symbol_table) == VarType::Base(BaseType::Double) {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Xor,
                            ty: AssemblyType::Double,
                            lhs: Operand::Reg(Register::Xmm(0)),
                            rhs: Operand::Reg(Register::Xmm(0)),
                        });
                        body.push(Instruction::Cmp(
                            AssemblyType::Double,
                            src.into(),
                            Operand::Reg(Register::Xmm(0)),
                        ));
                        body.push(Instruction::JmpCc(CondCode::Ne, dst.clone()));
                    } else {
                        body.push(Instruction::Cmp(
                            self.val_asm_type(src),
                            Operand::Imm(0),
                            src.into(),
                        ));
                        body.push(Instruction::JmpCc(CondCode::Ne, dst.clone()));
                    }
                }
                tacky::Instruction::Label(label) => {
                    body.push(Instruction::Label(label.clone()));
                }
                tacky::Instruction::FunCall { callee, args, dst } => {
                    let (int_dests, double_dests, return_in_memory) = if let Some(retval) = dst {
                        self.classify_return_value(retval)
                    } else {
                        let Attr::Fun { ty, .. } = &self.symbol_table[callee.var()] else {
                            panic!()
                        };
                        (
                            Vec::new(),
                            Vec::new(),
                            is_return_in_memory(&ty.ret, self.symbol_table),
                        )
                    };

                    let param_regs = if return_in_memory {
                        if let Some(dst) = dst {
                            body.push(Instruction::Lea {
                                src: dst.into(),
                                dst: Operand::Reg(Register::Di),
                            });
                        }
                        &PARAM_REGISTERS[1..]
                    } else {
                        &PARAM_REGISTERS
                    };

                    let (int_reg_args, double_reg_args, stack_args) =
                        self.classify_parameters(args.iter().cloned(), return_in_memory);

                    let stack_padding = 8 * (stack_args.len() % 2);

                    if stack_padding > 0 {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Sub,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Imm(stack_padding as _),
                            rhs: Operand::Reg(Register::SP),
                        });
                    }

                    for (i, (asm_ty, op)) in int_reg_args.into_iter().enumerate() {
                        if let AssemblyType::ByteArray { size, .. } = asm_ty {
                            self.copy_bytes_to_reg(&op, param_regs[i], size, &mut body);
                        } else {
                            body.push(Instruction::Mov {
                                ty: asm_ty,
                                src: op,
                                dst: Operand::Reg(param_regs[i]),
                            });
                        }
                    }

                    for (i, (_, op)) in double_reg_args.into_iter().enumerate() {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::Double,
                            src: op,
                            dst: Operand::Reg(Register::Xmm(i as _)),
                        });
                    }

                    let stack_len = stack_args.len();
                    for (asm_ty, op) in stack_args.into_iter().rev() {
                        if let AssemblyType::ByteArray { size, .. } = asm_ty {
                            body.push(Instruction::Binary {
                                op: BinaryOp::Sub,
                                ty: AssemblyType::QuadWord,
                                lhs: Operand::Imm(8),
                                rhs: Operand::Reg(Register::SP),
                            });
                            debug_assert!(size <= 8);
                            divide_into_assembly_sizes(size).for_each(|(asm_ty, offset)| {
                                body.push(Instruction::Mov {
                                    ty: asm_ty,
                                    src: op.offset(offset as _),
                                    dst: Operand::Memory(Register::SP, offset as _),
                                });
                            });
                        } else if matches!(op, Operand::Reg(_) | Operand::Imm(_))
                            || asm_ty == AssemblyType::QuadWord
                            || asm_ty == AssemblyType::Double
                        {
                            body.push(Instruction::Push(op));
                        } else {
                            body.push(Instruction::Mov {
                                ty: asm_ty,
                                src: op,
                                dst: Operand::Reg(Register::Ax),
                            });
                            body.push(Instruction::Push(Operand::Reg(Register::Ax)));
                        }
                    }

                    body.push(Instruction::Call(
                        if let Attr::Fun { .. } = self.symbol_table[callee.var()] {
                            Operand::Plt(callee.var().clone())
                        } else {
                            callee.into()
                        },
                    ));

                    let bytes_to_remove = 8 * stack_len + stack_padding;

                    if bytes_to_remove > 0 {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Add,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Imm(bytes_to_remove as _),
                            rhs: Operand::Reg(Register::SP),
                        });
                    }

                    if dst.is_some() && !return_in_memory {
                        let int_return_regs = [Register::Ax, Register::Dx];

                        for (i, (asm_ty, op)) in int_dests.into_iter().enumerate() {
                            match asm_ty {
                                AssemblyType::ByteArray { size, .. } => {
                                    self.copy_bytes_from_reg(
                                        &op,
                                        int_return_regs[i],
                                        size,
                                        &mut body,
                                    );
                                }
                                _ => {
                                    body.push(Instruction::Mov {
                                        ty: asm_ty,
                                        src: Operand::Reg(int_return_regs[i]),
                                        dst: op,
                                    });
                                }
                            }
                        }

                        for (i, op) in double_dests.into_iter().enumerate() {
                            body.push(Instruction::Mov {
                                ty: AssemblyType::Double,
                                src: Operand::Reg(Register::Xmm(i as _)),
                                dst: op,
                            });
                        }
                    }
                }
                tacky::Instruction::Cast { src, dst } => {
                    let src_ty = src.ty(self.symbol_table);
                    let dst_ty = dst.ty(self.symbol_table);

                    match (src_ty, dst_ty) {
                        (
                            VarType::Base(BaseType::Double),
                            VarType::Base(
                                BaseType::Char | BaseType::SChar | BaseType::Int | BaseType::Long,
                            ),
                        ) => {
                            // Double to signed
                            if self.symbol_table.size(&dst.ty(self.symbol_table)) == 1 {
                                body.push(Instruction::Cvttsd2si {
                                    ty: AssemblyType::LongWord,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Byte,
                                    src: Operand::Reg(Register::R10),
                                    dst: dst.into(),
                                });
                            } else {
                                body.push(Instruction::Cvttsd2si {
                                    ty: self.val_asm_type(dst),
                                    src: src.into(),
                                    dst: dst.into(),
                                });
                            }
                        }
                        (
                            VarType::Base(BaseType::Double),
                            VarType::Base(BaseType::UChar | BaseType::Uint | BaseType::Ulong),
                        ) => {
                            if dst.ty(self.symbol_table) == ast::VarType::Base(BaseType::UChar) {
                                body.push(Instruction::Cvttsd2si {
                                    ty: AssemblyType::LongWord,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Byte,
                                    src: Operand::Reg(Register::R10),
                                    dst: dst.into(),
                                });
                            } else if dst.ty(self.symbol_table)
                                == ast::VarType::Base(BaseType::Uint)
                            {
                                body.push(Instruction::Cvttsd2si {
                                    ty: AssemblyType::QuadWord,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::LongWord,
                                    src: Operand::Reg(Register::R10),
                                    dst: dst.into(),
                                });
                            } else {
                                let upper_bound = Operand::Pseudo(Pseudo::Double {
                                    value: 9223372036854775808.0,
                                    alignment: 8,
                                });

                                let ae_upper = self.gen_label("ae_upper");
                                let end = self.gen_label("end");

                                body.push(Instruction::Cmp(
                                    AssemblyType::Double,
                                    upper_bound.clone(),
                                    src.into(),
                                ));
                                body.push(Instruction::JmpCc(CondCode::Ae, ae_upper.clone()));
                                body.push(Instruction::Cvttsd2si {
                                    ty: AssemblyType::QuadWord,
                                    src: src.into(),
                                    dst: dst.into(),
                                });
                                body.push(Instruction::Jmp(end.clone()));
                                body.push(Instruction::Label(ae_upper));
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Double,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::Xmm(0)),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::Sub,
                                    ty: AssemblyType::Double,
                                    lhs: upper_bound.clone(),
                                    rhs: Operand::Reg(Register::Xmm(0)),
                                });
                                body.push(Instruction::Cvttsd2si {
                                    ty: AssemblyType::QuadWord,
                                    src: Operand::Reg(Register::Xmm(0)),
                                    dst: dst.into(),
                                });
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::QuadWord,
                                    src: Operand::Imm(9223372036854775808),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::Add,
                                    ty: AssemblyType::QuadWord,
                                    lhs: Operand::Reg(Register::R10),
                                    rhs: dst.into(),
                                });
                                body.push(Instruction::Label(end));
                            }
                        }
                        (
                            VarType::Base(
                                BaseType::Char | BaseType::SChar | BaseType::Int | BaseType::Long,
                            ),
                            VarType::Base(BaseType::Double),
                        ) => {
                            if self.symbol_table.size(&src.ty(self.symbol_table)) == 1 {
                                body.push(Instruction::Movsx {
                                    src_type: AssemblyType::Byte,
                                    dst_type: AssemblyType::LongWord,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Cvtsi2sd {
                                    ty: AssemblyType::LongWord,
                                    src: Operand::Reg(Register::R10),
                                    dst: dst.into(),
                                });
                            } else {
                                body.push(Instruction::Cvtsi2sd {
                                    ty: self.val_asm_type(src),
                                    src: src.into(),
                                    dst: dst.into(),
                                });
                            }
                        }
                        (
                            VarType::Base(BaseType::Uint | BaseType::Ulong | BaseType::UChar),
                            VarType::Base(BaseType::Double),
                        ) => match src.ty(self.symbol_table) {
                            ast::VarType::Base(BaseType::UChar) => {
                                body.push(Instruction::MovZeroExtend {
                                    src_type: AssemblyType::Byte,
                                    dst_type: AssemblyType::LongWord,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Cvtsi2sd {
                                    ty: AssemblyType::LongWord,
                                    src: Operand::Reg(Register::R10),
                                    dst: dst.into(),
                                });
                            }
                            ast::VarType::Base(BaseType::Uint) => {
                                body.push(Instruction::MovZeroExtend {
                                    src_type: AssemblyType::LongWord,
                                    dst_type: AssemblyType::QuadWord,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Cvtsi2sd {
                                    ty: AssemblyType::QuadWord,
                                    src: Operand::Reg(Register::R10),
                                    dst: dst.into(),
                                });
                            }
                            ast::VarType::Base(BaseType::Ulong) => {
                                let l1 = self.gen_label("l1");
                                let end = self.gen_label("end");
                                body.push(Instruction::Cmp(
                                    AssemblyType::QuadWord,
                                    Operand::Imm(0),
                                    src.into(),
                                ));
                                body.push(Instruction::JmpCc(CondCode::L, l1.clone()));
                                body.push(Instruction::Cvtsi2sd {
                                    ty: AssemblyType::QuadWord,
                                    src: src.into(),
                                    dst: dst.into(),
                                });
                                body.push(Instruction::Jmp(end.clone()));
                                body.push(Instruction::Label(l1));
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::QuadWord,
                                    src: src.into(),
                                    dst: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::QuadWord,
                                    src: Operand::Reg(Register::R10),
                                    dst: Operand::Reg(Register::R11),
                                });
                                body.push(Instruction::Unary {
                                    op: UnaryOp::Shr,
                                    ty: AssemblyType::QuadWord,
                                    src: Operand::Reg(Register::R11),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::And,
                                    ty: AssemblyType::QuadWord,
                                    lhs: Operand::Imm(1),
                                    rhs: Operand::Reg(Register::R10),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::Or,
                                    ty: AssemblyType::QuadWord,
                                    lhs: Operand::Reg(Register::R10),
                                    rhs: Operand::Reg(Register::R11),
                                });
                                body.push(Instruction::Cvtsi2sd {
                                    ty: AssemblyType::QuadWord,
                                    src: Operand::Reg(Register::R11),
                                    dst: dst.into(),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::Add,
                                    ty: AssemblyType::Double,
                                    lhs: dst.into(),
                                    rhs: dst.into(),
                                });
                                body.push(Instruction::Label(end));
                            }
                            _ => unreachable!(),
                        },
                        (from, to)
                            if self.symbol_table.size(&from) == self.symbol_table.size(&to) =>
                        {
                            body.push(Instruction::Mov {
                                ty: self.val_asm_type(src),
                                src: src.into(),
                                dst: dst.into(),
                            });
                        }
                        (from, to)
                            if self.symbol_table.size(&from) > self.symbol_table.size(&to) =>
                        {
                            body.push(Instruction::Mov {
                                ty: self.val_asm_type(dst),
                                src: src.into(),
                                dst: dst.into(),
                            });
                        }
                        (from, _to) if from.is_signed() => {
                            body.push(Instruction::Movsx {
                                src_type: self.val_asm_type(src),
                                dst_type: self.val_asm_type(dst),
                                src: src.into(),
                                dst: dst.into(),
                            });
                        }
                        (_from, _to) => {
                            body.push(Instruction::MovZeroExtend {
                                src_type: self.val_asm_type(src),
                                dst_type: self.val_asm_type(dst),
                                src: src.into(),
                                dst: dst.into(),
                            });
                        }
                    }
                }
                tacky::Instruction::Load { src, dst } => {
                    let size = self.symbol_table.size(&dst.ty(self.symbol_table));
                    let Val::Var(dst) = dst else { unreachable!() };
                    body.push(Instruction::Mov {
                        ty: AssemblyType::QuadWord,
                        src: src.into(),
                        dst: Operand::Reg(Register::Ax),
                    });
                    for (asm, offset) in divide_into_assembly_sizes(size) {
                        body.push(Instruction::Mov {
                            ty: asm,
                            src: Operand::Memory(Register::Ax, offset as i32),
                            dst: Operand::Pseudo(Pseudo::Mem {
                                name: dst.clone(),
                                offset,
                            }),
                        });
                    }
                }
                tacky::Instruction::Store { src, dst } => {
                    let size = self.symbol_table.size(&src.ty(self.symbol_table));
                    if let Val::Var(src) = src {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: dst.into(),
                            dst: Operand::Reg(Register::Ax),
                        });
                        for (asm, offset) in divide_into_assembly_sizes(size) {
                            body.push(Instruction::Mov {
                                ty: asm,
                                src: Operand::Pseudo(Pseudo::Mem {
                                    name: src.clone(),
                                    offset,
                                }),
                                dst: Operand::Memory(Register::Ax, offset as i32),
                            });
                        }
                    } else {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: dst.into(),
                            dst: Operand::Reg(Register::Ax),
                        });
                        body.push(Instruction::Mov {
                            ty: self.val_asm_type(src),
                            src: src.into(),
                            dst: Operand::Memory(Register::Ax, 0),
                        });
                    }
                }
                tacky::Instruction::GetAddress { src, dst } => {
                    let var = src.var();
                    if let Attr::Fun { .. } = self.symbol_table[var] {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: Operand::GotPcrel(var.clone()),
                            dst: dst.into(),
                        });
                    } else {
                        body.push(Instruction::Lea {
                            src: Operand::Pseudo(Pseudo::Mem {
                                name: var.clone(),
                                offset: 0,
                            }),
                            dst: dst.into(),
                        });
                    }
                }
                tacky::Instruction::CopyToOffset { src, dst, offset } => match src {
                    Val::Constant(_) => {
                        body.push(Instruction::Mov {
                            ty: self.val_asm_type(src),
                            src: src.into(),
                            dst: Operand::Pseudo(Pseudo::Mem {
                                name: dst.clone(),
                                offset: *offset,
                            }),
                        });
                    }
                    Val::Var(src_name) => {
                        let size = self.symbol_table.size(&src.ty(self.symbol_table));

                        for (asm, offset2) in divide_into_assembly_sizes(size) {
                            body.push(Instruction::Mov {
                                ty: asm,
                                src: Operand::Pseudo(Pseudo::Mem {
                                    name: src_name.clone(),
                                    offset: offset2,
                                }),
                                dst: Operand::Pseudo(Pseudo::Mem {
                                    name: dst.clone(),
                                    offset: offset + offset2,
                                }),
                            });
                        }
                    }
                },
                tacky::Instruction::CopyFromOffset { src, offset, dst } => {
                    let size = self.symbol_table.size(&dst.ty(self.symbol_table));
                    let Val::Var(dst) = dst else { unreachable!() };

                    for (asm, offset2) in divide_into_assembly_sizes(size) {
                        body.push(Instruction::Mov {
                            ty: asm,
                            src: Operand::Pseudo(Pseudo::Mem {
                                name: src.clone(),
                                offset: offset + offset2,
                            }),
                            dst: Operand::Pseudo(Pseudo::Mem {
                                name: dst.clone(),
                                offset: offset2,
                            }),
                        });
                    }
                }
                tacky::Instruction::AddPtr {
                    ptr,
                    index,
                    scale,
                    dst,
                } => match *scale {
                    1 | 2 | 4 | 8 => {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: ptr.into(),
                            dst: Operand::Reg(Register::Ax),
                        });
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: index.into(),
                            dst: Operand::Reg(Register::Dx),
                        });
                        body.push(Instruction::Lea {
                            src: Operand::Indexed {
                                base: Register::Ax,
                                index: Register::Dx,
                                scale: *scale,
                            },
                            dst: dst.into(),
                        });
                    }
                    _ => {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: ptr.into(),
                            dst: Operand::Reg(Register::Ax),
                        });
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: index.into(),
                            dst: Operand::Reg(Register::Dx),
                        });
                        body.push(Instruction::Binary {
                            op: BinaryOp::Mult,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Imm(*scale as _),
                            rhs: Operand::Reg(Register::Dx),
                        });
                        body.push(Instruction::Lea {
                            src: Operand::Indexed {
                                base: Register::Ax,
                                index: Register::Dx,
                                scale: 1,
                            },
                            dst: dst.into(),
                        });
                    }
                },
            }
        }

        let callee_saved = if enable_register_relocation {
            let return_regs = return_registers(&function.return_ty, self.symbol_table);
            let callee_saved_int = register_allocation(
                &mut body,
                &return_regs,
                self.symbol_table,
                &aliased_vals,
                crate::optimize_asm::ColoringMode::Int,
            );
            let callee_saved_double = register_allocation(
                &mut body,
                &return_regs,
                self.symbol_table,
                &aliased_vals,
                crate::optimize_asm::ColoringMode::Double,
            );
            let mut v: Vec<_> = callee_saved_int
                .into_iter()
                .chain(callee_saved_double)
                .collect();
            v.sort();
            v
        } else {
            Vec::new()
        };

        let stack_size = pseudo_to_stack(
            &mut body,
            self.symbol_table,
            &mut self.const_table,
            if return_in_memory { 8 } else { 0 },
        );
        let total_stack_size = stack_size + 8 * callee_saved.len();
        let adjusted_stack_size = round_up(total_stack_size, 16);
        let stack_size = adjusted_stack_size - 8 * callee_saved.len();

        /*
        body.insert(
            0,
            Instruction::Binary {
                op: BinaryOp::Sub,
                ty: AssemblyType::QuadWord,
                lhs: Operand::Imm(stack_size as _),
                rhs: Operand::Reg(Register::SP),
            },
        );
        */

        body = avoid_mov_mem_mem(body);

        Function {
            global: function.global,
            name: function.name.clone(),
            body,
            callee_saved,
            stack_size,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn classify_parameters(
        &self,
        iter: impl Iterator<Item = Val>,
        return_in_memory: bool,
    ) -> (
        Vec<(AssemblyType, Operand)>,
        Vec<(AssemblyType, Operand)>,
        Vec<(AssemblyType, Operand)>,
    ) {
        let mut int_reg_args = Vec::new();
        let mut double_reg_args = Vec::new();
        let mut stack_args = Vec::new();

        let int_regs_available = if return_in_memory { 5 } else { 6 };

        for val in iter {
            let ty = val.ty(self.symbol_table);
            let asm_ty = asm_type(&ty, self.symbol_table);
            match &ty {
                VarType::Base(BaseType::Double) => {
                    if double_reg_args.len() < 8 {
                        double_reg_args.push((asm_ty, val.into()));
                    } else {
                        stack_args.push((asm_ty, val.into()));
                    }
                }
                VarType::Struct(name) => {
                    let structure = self.symbol_table.struct_def(name);
                    let classes = classify_struct(structure, self.symbol_table);
                    let mut use_stack = true;
                    let struct_size = structure.size;
                    let Val::Var(val_name) = val else {
                        unreachable!()
                    };

                    if classes[0] != Class::Memory {
                        let mut tentative_ints = Vec::new();
                        let mut tentative_doubles = Vec::new();
                        let mut offset = 0;
                        for &class in &classes {
                            let operand = Operand::Pseudo(Pseudo::Mem {
                                name: val_name.clone(),
                                offset,
                            });

                            if class == Class::Sse {
                                tentative_doubles.push((asm_ty, operand));
                            } else {
                                let eightbyte_type = get_eightbyte_type(offset, struct_size);
                                tentative_ints.push((eightbyte_type, operand));
                            }

                            offset += 8;
                        }

                        if (tentative_doubles.len() + double_reg_args.len()) <= 8
                            && (tentative_ints.len() + int_reg_args.len()) <= int_regs_available
                        {
                            double_reg_args.extend(tentative_doubles);
                            int_reg_args.extend(tentative_ints);
                            use_stack = false;
                        }
                    }
                    if use_stack {
                        let mut offset = 0;
                        for _ in classes {
                            let operand = Operand::Pseudo(Pseudo::Mem {
                                name: val_name.clone(),
                                offset,
                            });
                            let eightbyte_type = get_eightbyte_type(offset, struct_size);
                            stack_args.push((eightbyte_type, operand));
                            offset += 8;
                        }
                    }
                }
                _ => {
                    if int_reg_args.len() < int_regs_available {
                        int_reg_args.push((asm_ty, val.into()));
                    } else {
                        stack_args.push((asm_ty, val.into()));
                    }
                }
            }
        }

        (int_reg_args, double_reg_args, stack_args)
    }

    fn classify_return_value(
        &self,
        retval: &Val,
    ) -> (Vec<(AssemblyType, Operand)>, Vec<Operand>, bool) {
        let ty = retval.ty(self.symbol_table);
        let asm_ty: AssemblyType = asm_type(&ty, self.symbol_table);

        match asm_ty {
            AssemblyType::Double => (Vec::new(), vec![retval.into()], false),
            AssemblyType::ByteArray { .. } => {
                let Val::Var(name) = retval else {
                    unreachable!()
                };
                let VarType::Struct(struct_name) = &ty else {
                    unreachable!()
                };
                let struct_def = self.symbol_table.struct_def(struct_name);
                let classes = classify_struct(struct_def, self.symbol_table);
                let struct_size = struct_def.size;

                if classes[0] == Class::Memory {
                    (Vec::new(), Vec::new(), true)
                } else {
                    let mut int_retvals = Vec::new();
                    let mut double_ret_vals = Vec::new();
                    let mut offset = 0;

                    for class in classes {
                        let operand = Operand::Pseudo(Pseudo::Mem {
                            name: name.clone(),
                            offset,
                        });
                        match class {
                            Class::Sse => {
                                double_ret_vals.push(operand);
                            }
                            Class::Integer => {
                                let eightbyte_type = get_eightbyte_type(offset, struct_size);
                                int_retvals.push((eightbyte_type, operand));
                            }
                            Class::Memory => unreachable!(),
                        }
                        offset += 8;
                    }
                    (int_retvals, double_ret_vals, false)
                }
            }
            // scalar
            _ => (vec![(asm_ty, retval.into())], Vec::new(), false),
        }
    }
}

pub fn return_registers(ret_type: &VarType, symbol_table: &SymbolTable) -> Vec<Register> {
    if *ret_type == VarType::Void {
        return Vec::new();
    }

    let asm_ty: AssemblyType = asm_type(ret_type, symbol_table);

    match asm_ty {
        AssemblyType::Double => vec![Register::Xmm(0)],
        AssemblyType::ByteArray { .. } => {
            let VarType::Struct(struct_name) = ret_type else {
                unreachable!()
            };
            let struct_def = symbol_table.struct_def(struct_name);
            let classes = classify_struct(struct_def, symbol_table);

            if classes[0] == Class::Memory {
                vec![]
            } else {
                let mut int_retvals = 0;
                let mut double_ret_vals = 0;

                for class in classes {
                    match class {
                        Class::Sse => {
                            double_ret_vals += 1;
                        }
                        Class::Integer => {
                            int_retvals += 1;
                        }
                        Class::Memory => unreachable!(),
                    }
                }

                PARAM_REGISTERS[..int_retvals]
                    .iter()
                    .copied()
                    .chain((0..double_ret_vals).map(Register::Xmm))
                    .collect()
            }
        }
        // scalar
        _ => vec![Register::Ax],
    }
}

fn get_eightbyte_type(offset: usize, struct_size: usize) -> AssemblyType {
    let bytes_from_end = struct_size - offset;
    if bytes_from_end >= 8 {
        AssemblyType::QuadWord
    } else if bytes_from_end == 4 {
        AssemblyType::LongWord
    } else if bytes_from_end == 1 {
        AssemblyType::Byte
    } else {
        AssemblyType::ByteArray {
            size: bytes_from_end,
            alignment: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Memory,
    Sse,
    Integer,
}

const PARAM_REGISTERS: [Register; 6] = [
    Register::Di,
    Register::Si,
    Register::Dx,
    Register::Cx,
    Register::R8,
    Register::R9,
];

fn pseudo_to_stack(
    insts: &mut [Instruction],
    symbol_table: &SymbolTable,
    const_table: &mut ConstTable,
    offset: usize,
) -> usize {
    let mut total = offset as i32;
    let mut known_vars = HashMap::new();

    let mut remove_pseudo = |operand: &mut Operand| {
        if let Operand::Pseudo(var) = operand {
            match var {
                Pseudo::Double {
                    value: d,
                    alignment,
                } => {
                    *operand = Operand::Data(const_table.label(*d, *alignment), 0);
                }
                Pseudo::Mem { name, offset } => match &symbol_table[name] {
                    semantics::type_check::Attr::Static { .. } => {
                        *operand = Operand::Data(name.clone(), *offset as _)
                    }
                    semantics::type_check::Attr::Local(ty) => {
                        match known_vars.entry(name.clone()) {
                            Entry::Occupied(entry) => {
                                let addr = *entry.get();
                                *operand = Operand::stack(addr + (*offset as i32));
                            }
                            Entry::Vacant(entry) => {
                                let size = symbol_table.size(ty) as i32;
                                let align = symbol_table.alignment(ty) as i32;
                                total += size;
                                total = round_up(total as usize, align as usize) as i32;
                                entry.insert(-total);
                                *operand = Operand::stack(-total + (*offset as i32));
                            }
                        }
                    }
                    semantics::type_check::Attr::Constant { .. } => {
                        *operand = Operand::Data(name.clone(), *offset as _)
                    }
                    semantics::type_check::Attr::Fun { defined, .. } => {
                        if *defined {
                            *operand = Operand::Data(name.clone(), 0);
                        } else {
                            *operand = Operand::Plt(name.clone());
                        }
                    }
                    semantics::type_check::Attr::Struct(StructDef {
                        alignment, size, ..
                    }) => match known_vars.entry(name.clone()) {
                        Entry::Occupied(entry) => {
                            let addr = *entry.get();
                            *operand = Operand::stack(addr + (*offset as i32));
                        }
                        Entry::Vacant(entry) => {
                            let align = *alignment as i32;
                            total += *size as i32;
                            total = round_up(total as usize, align as usize) as i32;
                            entry.insert(-total);
                            *operand = Operand::stack(-total + (*offset as i32));
                        }
                    },
                },
            }
        }
    };

    for inst in insts {
        match inst {
            Instruction::Nop => {}
            Instruction::Mov { ty: _, src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Unary { src, .. } => {
                remove_pseudo(src);
            }
            Instruction::Binary { lhs, rhs, .. } => {
                remove_pseudo(lhs);
                remove_pseudo(rhs);
            }
            Instruction::Cdq(_) => {}
            Instruction::Idiv(_, op) => {
                remove_pseudo(op);
            }
            Instruction::Ret => {}
            Instruction::Cmp(_, lhs, rhs) => {
                remove_pseudo(lhs);
                remove_pseudo(rhs);
            }
            Instruction::Jmp(_) => {}
            Instruction::JmpCc(_, _) => {}
            Instruction::SetCc(_, dst) => {
                remove_pseudo(dst);
            }
            Instruction::Label(_) => {}
            Instruction::Push(op) => {
                remove_pseudo(op);
            }
            Instruction::Call(op) => {
                remove_pseudo(op);
            }
            Instruction::Movsx { src, dst, .. } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::MovZeroExtend { src, dst, .. } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Div(_, op) => {
                remove_pseudo(op);
            }
            Instruction::Cvttsd2si { ty: _, src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Cvtsi2sd { ty: _, src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Lea { src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Pop(_) => {}
        }
    }

    total as _
}

fn avoid_mov_mem_mem(insts: Vec<Instruction>) -> Vec<Instruction> {
    let mut new_insts = Vec::new();

    for inst in insts {
        match inst {
            Instruction::Mov {
                ty,
                src: src @ (Operand::Memory(..) | Operand::Data(..) | Operand::GotPcrel(..)),
                dst: dst @ (Operand::Memory(..) | Operand::Data(..)),
            } => {
                let tmp_reg = Operand::Reg(if ty == AssemblyType::Double {
                    Register::Xmm(14)
                } else {
                    Register::R10
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src,
                    dst: tmp_reg.clone(),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: tmp_reg.clone(),
                    dst,
                });
            }
            Instruction::Mov {
                ty: AssemblyType::QuadWord,
                src: src @ Operand::Imm(_),
                dst: dst @ (Operand::Memory(..) | Operand::Data(..)),
            } => {
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Movsx {
                src_type,
                dst_type,
                src: src @ Operand::Imm(_),
                dst: dst @ (Operand::Memory(..) | Operand::Data(..)),
            } => {
                new_insts.push(Instruction::Mov {
                    ty: src_type,
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Movsx {
                    src_type,
                    dst_type,
                    src: Operand::Reg(Register::R10),
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Mov {
                    ty: dst_type,
                    src: Operand::Reg(Register::R11),
                    dst,
                });
            }
            Instruction::Movsx {
                src_type,
                dst_type,
                src: src @ Operand::Imm(_),
                dst,
            } => {
                new_insts.push(Instruction::Mov {
                    ty: src_type,
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Movsx {
                    src_type,
                    dst_type,
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Movsx {
                src_type,
                dst_type,
                src,
                dst: dst @ (Operand::Memory(..) | Operand::Data(..)),
            } => {
                new_insts.push(Instruction::Movsx {
                    src_type,
                    dst_type,
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    ty: dst_type,
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Idiv(ty, op @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: op,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Idiv(ty, Operand::Reg(Register::R10)));
            }
            Instruction::Div(ty, op @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: op,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Div(ty, Operand::Reg(Register::R10)));
            }
            Instruction::Binary {
                ty,
                op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::And | BinaryOp::Or),
                lhs: lhs @ (Operand::Memory(..) | Operand::Data(..)),
                rhs: rhs @ (Operand::Memory(..) | Operand::Data(..)),
            } if !matches!(ty, AssemblyType::Double) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: lhs,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Binary {
                    ty,
                    op,
                    lhs: Operand::Reg(Register::R10),
                    rhs,
                });
            }
            Instruction::Binary {
                ty,
                op: BinaryOp::Mult,
                lhs,
                rhs: rhs @ (Operand::Memory(..) | Operand::Data(..)),
            } => {
                let lhs = if ty == AssemblyType::QuadWord && matches!(lhs, Operand::Imm(_)) {
                    new_insts.push(Instruction::Mov {
                        ty,
                        src: lhs,
                        dst: Operand::Reg(Register::R10),
                    });
                    Operand::Reg(Register::R10)
                } else {
                    lhs
                };
                let tmp = if ty == AssemblyType::Double {
                    Operand::Reg(Register::Xmm(15))
                } else {
                    Operand::Reg(Register::R11)
                };
                new_insts.push(Instruction::Mov {
                    ty,
                    src: rhs.clone(),
                    dst: tmp.clone(),
                });
                new_insts.push(Instruction::Binary {
                    ty,
                    op: BinaryOp::Mult,
                    lhs,
                    rhs: tmp.clone(),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: tmp.clone(),
                    dst: rhs,
                });
            }
            Instruction::Binary {
                op,
                ty: AssemblyType::QuadWord,
                lhs,
                rhs,
            } => {
                let lhs = if let Operand::Imm(x) = lhs {
                    if i32::try_from(x).is_err() {
                        new_insts.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: lhs,
                            dst: Operand::Reg(Register::R10),
                        });
                        Operand::Reg(Register::R10)
                    } else {
                        lhs
                    }
                } else {
                    lhs
                };

                let rhs = if let Operand::Imm(x) = rhs {
                    if i32::try_from(x).is_err() {
                        new_insts.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: rhs,
                            dst: Operand::Reg(Register::R11),
                        });
                        Operand::Reg(Register::R11)
                    } else {
                        rhs
                    }
                } else {
                    rhs
                };

                new_insts.push(Instruction::Binary {
                    op,
                    ty: AssemblyType::QuadWord,
                    lhs,
                    rhs,
                });
            }
            Instruction::Binary {
                op,
                ty: AssemblyType::Double,
                lhs,
                rhs,
            } => {
                if !matches!(rhs, Operand::Reg(_)) {
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::Double,
                        src: rhs.clone(),
                        dst: Operand::Reg(Register::Xmm(15)),
                    });
                    new_insts.push(Instruction::Binary {
                        op,
                        ty: AssemblyType::Double,
                        lhs,
                        rhs: Operand::Reg(Register::Xmm(15)),
                    });
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::Double,
                        src: Operand::Reg(Register::Xmm(15)),
                        dst: rhs,
                    });
                } else {
                    new_insts.push(Instruction::Binary {
                        op,
                        ty: AssemblyType::Double,
                        lhs,
                        rhs,
                    });
                };
            }
            Instruction::Cmp(
                ty,
                lhs @ (Operand::Memory(..) | Operand::Data(..)),
                rhs @ (Operand::Memory(..) | Operand::Data(..)),
            ) if !matches!(ty, AssemblyType::Double) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: lhs,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Cmp(ty, Operand::Reg(Register::R10), rhs));
            }
            Instruction::Cmp(ty, lhs, rhs) => match ty {
                AssemblyType::LongWord | AssemblyType::Byte => {
                    if matches!(rhs, Operand::Imm(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: rhs,
                            dst: Operand::Reg(Register::R11),
                        });
                        new_insts.push(Instruction::Cmp(ty, lhs, Operand::Reg(Register::R11)));
                    } else {
                        new_insts.push(Instruction::Cmp(ty, lhs, rhs));
                    }
                }
                AssemblyType::QuadWord => {
                    let lhs = if matches!(lhs, Operand::Imm(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: lhs,
                            dst: Operand::Reg(Register::R10),
                        });
                        Operand::Reg(Register::R10)
                    } else {
                        lhs
                    };

                    let rhs = if matches!(rhs, Operand::Imm(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: rhs,
                            dst: Operand::Reg(Register::R11),
                        });
                        Operand::Reg(Register::R11)
                    } else {
                        rhs
                    };

                    new_insts.push(Instruction::Cmp(ty, lhs, rhs));
                }
                AssemblyType::Double => {
                    let rhs = if !matches!(rhs, Operand::Reg(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: rhs,
                            dst: Operand::Reg(Register::Xmm(15)),
                        });
                        Operand::Reg(Register::Xmm(15))
                    } else {
                        rhs
                    };

                    new_insts.push(Instruction::Cmp(ty, lhs, rhs));
                }
                AssemblyType::ByteArray { .. } => unreachable!(),
            },
            Instruction::Push(op @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: op,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Push(Operand::Reg(Register::R10)));
            }
            Instruction::Push(op @ Operand::Reg(Register::Xmm(_))) => {
                new_insts.push(Instruction::Binary {
                    op: BinaryOp::Sub,
                    ty: AssemblyType::QuadWord,
                    lhs: Operand::Imm(8),
                    rhs: Operand::Reg(Register::SP),
                });
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: op,
                    dst: Operand::Memory(Register::SP, 0),
                });
            }
            Instruction::MovZeroExtend {
                src_type,
                dst_type,
                src,
                dst,
            } => {
                if src_type == AssemblyType::LongWord {
                    if let Operand::Reg(_) = dst {
                        new_insts.push(Instruction::Mov {
                            ty: AssemblyType::LongWord,
                            src,
                            dst,
                        });
                    } else {
                        new_insts.push(Instruction::Mov {
                            ty: AssemblyType::LongWord,
                            src,
                            dst: Operand::Reg(Register::R11),
                        });
                        new_insts.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: Operand::Reg(Register::R11),
                            dst,
                        });
                    }
                } else if let Operand::Reg(_) = dst {
                    new_insts.push(Instruction::MovZeroExtend {
                        src_type,
                        dst_type,
                        src,
                        dst,
                    });
                } else {
                    new_insts.push(Instruction::Mov {
                        ty: src_type,
                        src,
                        dst: Operand::Reg(Register::R10),
                    });
                    new_insts.push(Instruction::MovZeroExtend {
                        src_type,
                        dst_type,
                        src: Operand::Reg(Register::R10),
                        dst: Operand::Reg(Register::R11),
                    });
                    new_insts.push(Instruction::Mov {
                        ty: dst_type,
                        src: Operand::Reg(Register::R11),
                        dst,
                    });
                }
            }
            Instruction::Cvttsd2si { ty, src, dst } if !matches!(dst, Operand::Reg(_)) => {
                new_insts.push(Instruction::Cvttsd2si {
                    ty,
                    src,
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: Operand::Reg(Register::R11),
                    dst,
                });
            }
            Instruction::Cvtsi2sd { ty, src, dst } => {
                let src = if matches!(src, Operand::Imm(_)) {
                    new_insts.push(Instruction::Mov {
                        ty,
                        src,
                        dst: Operand::Reg(Register::R10),
                    });
                    Operand::Reg(Register::R10)
                } else {
                    src
                };

                if !matches!(dst, Operand::Reg(_)) {
                    new_insts.push(Instruction::Cvtsi2sd {
                        ty,
                        src,
                        dst: Operand::Reg(Register::Xmm(15)),
                    });
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::Double,
                        src: Operand::Reg(Register::Xmm(15)),
                        dst,
                    });
                } else {
                    new_insts.push(Instruction::Cvtsi2sd { ty, src, dst });
                }
            }
            Instruction::Lea { src, dst } if !matches!(dst, Operand::Reg(_)) => {
                new_insts.push(Instruction::Lea {
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            _ => new_insts.push(inst),
        }
    }
    new_insts
}

struct SizedOperand<'a> {
    ty: AssemblyType,
    op: &'a Operand,
}

impl<'a> Display for SizedOperand<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.op {
            Operand::Imm(imm) => match self.ty {
                // trucated anyway
                AssemblyType::Byte => write!(f, "${}", *imm as u8)?,
                AssemblyType::LongWord => write!(f, "${}", *imm as u32)?,
                AssemblyType::QuadWord | AssemblyType::Double => write!(f, "${}", imm)?,
                _ => unreachable!(),
            },
            Operand::Reg(reg) => match self.ty {
                AssemblyType::Byte => write!(f, "{}", RegisterSize::Byte(reg))?,
                AssemblyType::LongWord => write!(f, "{}", RegisterSize::Dword(reg))?,
                AssemblyType::QuadWord | AssemblyType::Double => {
                    write!(f, "{}", RegisterSize::Qword(reg))?
                }
                _ => unreachable!(),
            },
            Operand::Pseudo(_) => panic!("Pseudo operand should have been removed"),
            Operand::Data(name, offset) => {
                if *offset == 0 {
                    write!(f, "{}(%rip)", name)?
                } else {
                    write!(f, "{}+{}(%rip)", name, offset)?
                }
            }
            Operand::Memory(reg, offset) => write!(f, "{}({})", offset, RegisterSize::Qword(reg))?,
            Operand::Plt(name) => write!(f, "{}", name)?, // write!(f, "{}@PLT", name)?,
            Operand::GotPcrel(name) => write!(f, "{}@GOTPCREL(%rip)", name)?,
            Operand::Indexed { base, index, scale } => write!(
                f,
                "({},{},{})",
                RegisterSize::Qword(base),
                RegisterSize::Qword(index),
                scale
            )?,
        }

        Ok(())
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for top in &self.top_levels {
            writeln!(f, "{}", top)?;
        }
        writeln!(f, ".section .note.GNU-stack,\"\",@progbits")?;
        Ok(())
    }
}

impl Display for TopLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopLevel::StaticConstant(constant) => write!(f, "{}", constant)?,
            TopLevel::StaticVariable(var) => write!(f, "{}", var)?,
            TopLevel::Function(func) => write!(f, "{}", func)?,
        }
        Ok(())
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.global {
            writeln!(f, ".globl {}", self.name)?;
        }
        writeln!(f, ".text")?;
        writeln!(f, "{}:", self.name)?;

        writeln!(f, "pushq %rbp")?;
        writeln!(f, "movq %rsp, %rbp")?;
        if self.stack_size != 0 {
            writeln!(
                f,
                "{}",
                Instruction::Binary {
                    op: BinaryOp::Sub,
                    ty: AssemblyType::QuadWord,
                    lhs: Operand::Imm(self.stack_size as _),
                    rhs: Operand::Reg(Register::SP),
                }
            )?;
        }
        for r in &self.callee_saved {
            writeln!(f, "pushq {}", RegisterSize::Qword(r))?;
        }

        for inst in &self.body {
            if let Instruction::Ret = inst {
                for r in self.callee_saved.iter().rev() {
                    writeln!(f, "popq {}", RegisterSize::Qword(r))?;
                }
            }
            write!(f, "{inst}")?;
        }
        Ok(())
    }
}

impl Display for StaticConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, ".section .rodata")?;
        writeln!(f, ".align {}", self.alignment)?;
        writeln!(f, "{}:", self.name)?;
        writeln!(f, "{}", self.init)?;
        Ok(())
    }
}

impl Display for StaticVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.global {
            writeln!(f, ".globl {}", self.name)?;
        }

        let is_zero = self.init.iter().all(|i| i.is_zero());

        if is_zero {
            writeln!(f, ".bss")?;
            writeln!(f, ".align {}", self.alignment)?;
            writeln!(f, "{}:", self.name)?;
            writeln!(
                f,
                ".zero {}",
                self.init.iter().map(|i| i.size()).sum::<usize>()
            )?;
        } else {
            writeln!(f, ".data")?;
            writeln!(f, ".align {}", self.alignment)?;
            writeln!(f, "{}:", self.name)?;
            for init in &self.init {
                writeln!(f, "{}", init)?;
            }
        }
        Ok(())
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Nop => {}
            Instruction::Mov { ty, src, dst } => {
                writeln!(
                    f,
                    "mov{} {}, {}",
                    ty.suffix(),
                    src.sized(*ty),
                    dst.sized(*ty)
                )?;
            }
            Instruction::Unary { ty, op, src } => {
                writeln!(f, "{op}{} {}", ty.suffix(), src.sized(*ty))?;
            }
            Instruction::Ret => {
                writeln!(f, "movq %rbp, %rsp")?;
                writeln!(f, "popq %rbp")?;
                writeln!(f, "ret")?;
            }
            Instruction::Binary { ty, op, lhs, rhs } => {
                if *ty == AssemblyType::Double {
                    match op {
                        BinaryOp::Mult => {
                            writeln!(
                                f,
                                "mulsd {}, {}",
                                lhs.sized(AssemblyType::Double),
                                rhs.sized(AssemblyType::Double)
                            )?;
                            return Ok(());
                        }
                        BinaryOp::Xor => {
                            writeln!(
                                f,
                                "xorpd {}, {}",
                                lhs.sized(AssemblyType::Double),
                                rhs.sized(AssemblyType::Double)
                            )?;
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                writeln!(
                    f,
                    "{op}{} {}, {}",
                    ty.suffix(),
                    lhs.sized(*ty),
                    rhs.sized(*ty)
                )?;
            }
            Instruction::Cdq(ty) => match ty {
                AssemblyType::LongWord => {
                    writeln!(f, "cdq")?;
                }
                AssemblyType::QuadWord | AssemblyType::Double => {
                    writeln!(f, "cqo")?;
                }
                AssemblyType::ByteArray { .. } | AssemblyType::Byte => unimplemented!(),
            },
            Instruction::Idiv(ty, op) => {
                writeln!(f, "idiv{} {}", ty.suffix(), op.sized(*ty))?;
            }
            Instruction::Cmp(ty, lhs, rhs) => {
                if *ty == AssemblyType::Double {
                    writeln!(
                        f,
                        "comisd {}, {}",
                        lhs.sized(AssemblyType::Double),
                        rhs.sized(AssemblyType::Double)
                    )?;
                    return Ok(());
                } else {
                    writeln!(
                        f,
                        "cmp{} {}, {}",
                        ty.suffix(),
                        lhs.sized(*ty),
                        rhs.sized(*ty)
                    )?;
                }
            }
            Instruction::Jmp(l) => {
                writeln!(f, "jmp .L{}", l)?;
            }
            Instruction::JmpCc(cond, l) => {
                writeln!(f, "j{} .L{}", cond, l)?;
            }
            Instruction::SetCc(cond, Operand::Reg(reg)) => {
                writeln!(f, "set{} {}", cond, RegisterSize::Byte(reg))?;
            }
            Instruction::SetCc(cond, dst) => {
                writeln!(f, "set{} {}", cond, dst.sized(AssemblyType::LongWord))?;
            }
            Instruction::Label(l) => {
                writeln!(f, ".L{}:", l)?;
            }
            Instruction::Push(op) => {
                writeln!(f, "pushq {}", op.sized(AssemblyType::QuadWord))?;
            }
            Instruction::Pop(reg) => {
                writeln!(
                    f,
                    "popq {}",
                    Operand::Reg(*reg).sized(AssemblyType::QuadWord)
                )?;
            }
            Instruction::Call(op) => {
                if let Operand::Plt(_) = op {
                    writeln!(f, "call {}", op.sized(AssemblyType::QuadWord))?;
                } else {
                    writeln!(f, "call *{}", op.sized(AssemblyType::QuadWord))?;
                }
            }
            Instruction::Movsx {
                src_type,
                dst_type,
                src,
                dst,
            } => {
                writeln!(
                    f,
                    "movs{}{} {}, {}",
                    src_type.suffix(),
                    dst_type.suffix(),
                    src.sized(*src_type),
                    dst.sized(*dst_type)
                )?;
            }
            Instruction::MovZeroExtend {
                src_type,
                dst_type,
                src,
                dst,
            } => {
                writeln!(
                    f,
                    "movz{}{} {}, {}",
                    src_type.suffix(),
                    dst_type.suffix(),
                    src.sized(*src_type),
                    dst.sized(*dst_type)
                )?;
            }
            Instruction::Div(ty, op) => {
                writeln!(f, "div{} {}", ty.suffix(), op.sized(*ty))?;
            }
            Instruction::Cvttsd2si { ty, src, dst } => {
                writeln!(
                    f,
                    "cvttsd2si{} {}, {}",
                    ty.suffix(),
                    src.sized(AssemblyType::Double),
                    dst.sized(*ty)
                )?;
            }
            Instruction::Cvtsi2sd { ty, src, dst } => {
                writeln!(
                    f,
                    "cvtsi2sd{} {}, {}",
                    ty.suffix(),
                    src.sized(*ty),
                    dst.sized(AssemblyType::Double)
                )?;
            }
            Instruction::Lea { src, dst } => {
                writeln!(
                    f,
                    "leaq {}, {}",
                    src.sized(AssemblyType::QuadWord),
                    dst.sized(AssemblyType::QuadWord)
                )?;
            }
        }
        Ok(())
    }
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "neg")?,
            UnaryOp::Not => write!(f, "not")?,
            UnaryOp::Shr => write!(f, "shr")?,
        }
        Ok(())
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "add")?,
            BinaryOp::Sub => write!(f, "sub")?,
            BinaryOp::Mult => write!(f, "imul")?,
            BinaryOp::And => write!(f, "and")?,
            BinaryOp::Or => write!(f, "or")?,
            BinaryOp::DivDouble => write!(f, "div")?,
            BinaryOp::Xor => write!(f, "xor")?,
            BinaryOp::Shl => write!(f, "shl")?,
            BinaryOp::ShrTwo => write!(f, "shr")?,
        }
        Ok(())
    }
}

impl<'a> Display for RegisterSize<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterSize::Byte(reg) => match reg {
                Register::Ax => write!(f, "%al")?,
                Register::Bx => write!(f, "%bl")?,
                Register::Cx => write!(f, "%cl")?,
                Register::Dx => write!(f, "%dl")?,
                Register::Di => write!(f, "%dil")?,
                Register::Si => write!(f, "%sil")?,
                Register::R8 => write!(f, "%r8b")?,
                Register::R9 => write!(f, "%r9b")?,
                Register::R10 => write!(f, "%r10b")?,
                Register::R11 => write!(f, "%r11b")?,
                Register::R12 => write!(f, "%r12b")?,
                Register::R13 => write!(f, "%r13b")?,
                Register::R14 => write!(f, "%r14b")?,
                Register::R15 => write!(f, "%r15b")?,
                Register::SP => write!(f, "%spl")?,
                Register::BP => write!(f, "%bpl")?,
                Register::Xmm(_) => panic!("XMM registers are only used for QWord"),
            },
            RegisterSize::Dword(reg) => match reg {
                Register::Ax => write!(f, "%eax")?,
                Register::Bx => write!(f, "%ebx")?,
                Register::Cx => write!(f, "%ecx")?,
                Register::Dx => write!(f, "%edx")?,
                Register::Di => write!(f, "%edi")?,
                Register::Si => write!(f, "%esi")?,
                Register::R8 => write!(f, "%r8d")?,
                Register::R9 => write!(f, "%r9d")?,
                Register::R10 => write!(f, "%r10d")?,
                Register::R11 => write!(f, "%r11d")?,
                Register::R12 => write!(f, "%r12d")?,
                Register::R13 => write!(f, "%r13d")?,
                Register::R14 => write!(f, "%r14d")?,
                Register::R15 => write!(f, "%r15d")?,
                Register::SP => write!(f, "%esp")?,
                Register::BP => write!(f, "%ebp")?,
                Register::Xmm(_) => panic!("XMM registers are only used for QWord"),
            },
            RegisterSize::Qword(reg) => match reg {
                Register::Ax => write!(f, "%rax")?,
                Register::Bx => write!(f, "%rbx")?,
                Register::Cx => write!(f, "%rcx")?,
                Register::Dx => write!(f, "%rdx")?,
                Register::Di => write!(f, "%rdi")?,
                Register::Si => write!(f, "%rsi")?,
                Register::R8 => write!(f, "%r8")?,
                Register::R9 => write!(f, "%r9")?,
                Register::R10 => write!(f, "%r10")?,
                Register::R11 => write!(f, "%r11")?,
                Register::R12 => write!(f, "%r12")?,
                Register::R13 => write!(f, "%r13")?,
                Register::R14 => write!(f, "%r14")?,
                Register::R15 => write!(f, "%r15")?,
                Register::SP => write!(f, "%rsp")?,
                Register::BP => write!(f, "%rbp")?,
                Register::Xmm(x) => write!(f, "%xmm{}", x)?,
            },
        }
        Ok(())
    }
}

impl Display for CondCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CondCode::E => write!(f, "e")?,
            CondCode::Ne => write!(f, "ne")?,
            CondCode::G => write!(f, "g")?,
            CondCode::Ge => write!(f, "ge")?,
            CondCode::L => write!(f, "l")?,
            CondCode::Le => write!(f, "le")?,
            CondCode::A => write!(f, "a")?,
            CondCode::Ae => write!(f, "ae")?,
            CondCode::B => write!(f, "b")?,
            CondCode::Be => write!(f, "be")?,
        }
        Ok(())
    }
}

pub fn is_return_in_memory(ty: &VarType, symbol_table: &SymbolTable) -> bool {
    if let VarType::Struct(name) = ty {
        let struct_def = symbol_table.struct_def(name);
        let classes = classify_struct(struct_def, symbol_table);
        classes[0] == Class::Memory
    } else {
        false
    }
}

pub fn classify_struct(
    structure: &type_check::StructDef,
    symbol_table: &SymbolTable,
) -> Vec<Class> {
    if structure.size > 16 {
        let mut ret = Vec::new();
        let mut size = structure.size;
        while size > 0 {
            if size >= 8 {
                size -= 8;
            } else {
                size = 0;
            }
            ret.push(Class::Memory);
        }
        ret
    } else {
        let mut scalar_types = Vec::new();
        for member in &structure.members {
            scalar_types.extend(symbol_table.flatten(&member.ty));
        }

        if structure.size > 8 {
            if scalar_types.first() == Some(&BaseType::Double)
                && scalar_types.last() == Some(&BaseType::Double)
            {
                vec![Class::Sse, Class::Sse]
            } else if scalar_types.first() == Some(&BaseType::Double) {
                vec![Class::Sse, Class::Integer]
            } else if scalar_types.last() == Some(&BaseType::Double) {
                vec![Class::Integer, Class::Sse]
            } else {
                vec![Class::Integer, Class::Integer]
            }
        } else if scalar_types.first() == Some(&BaseType::Double) {
            vec![Class::Sse]
        } else {
            vec![Class::Integer]
        }
    }
}

pub fn asm_type(ty: &VarType, symbol_table: &SymbolTable) -> AssemblyType {
    match ty {
        VarType::Void => panic!("Void type must be eliminated in type checking stage"),
        VarType::Base(base) => match base {
            BaseType::Char | BaseType::SChar | BaseType::UChar => AssemblyType::Byte,
            BaseType::Int | BaseType::Uint => AssemblyType::LongWord,
            BaseType::Long | BaseType::Ulong => AssemblyType::QuadWord,
            BaseType::Double => AssemblyType::Double,
        },
        VarType::Pointer(_) => AssemblyType::QuadWord,
        VarType::Array { .. } => AssemblyType::ByteArray {
            size: symbol_table.size(ty),
            alignment: symbol_table.alignment(ty),
        },
        VarType::Struct(name) => {
            let struct_def = symbol_table.struct_def(name);

            AssemblyType::ByteArray {
                size: struct_def.size,
                alignment: struct_def.alignment,
            }
        }
    }
}
