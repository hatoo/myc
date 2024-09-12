use std::collections::{BTreeMap, HashMap, HashSet};

use ecow::EcoString;
use egglog::{
    ast::{Expr, Literal, Symbol},
    EGraph, Term, TermDag, Value,
};
use ordered_float::OrderedFloat;

use crate::{
    ast::Const,
    control_flow::NodeId,
    ssa::Ssa,
    tacky::{BinaryOp, Instruction, UnaryOp, Val},
};

pub trait ToEgglogExpr {
    fn to_egglog_expr(&self) -> Expr;
}

impl ToEgglogExpr for UnaryOp {
    fn to_egglog_expr(&self) -> Expr {
        match self {
            UnaryOp::Negate => Expr::call("Negate", None),
            UnaryOp::Complement => Expr::call("Complement", None),
            UnaryOp::Not => Expr::call("Not", None),
        }
    }
}

impl ToEgglogExpr for BinaryOp {
    fn to_egglog_expr(&self) -> Expr {
        match self {
            BinaryOp::Add => Expr::call("Add", None),
            BinaryOp::Subtract => Expr::call("Subtract", None),
            BinaryOp::Multiply => Expr::call("Multiply", None),
            BinaryOp::Divide => Expr::call("Divide", None),
            BinaryOp::Remainder => Expr::call("Remainder", None),
            BinaryOp::Equal => Expr::call("Equal", None),
            BinaryOp::NotEqual => Expr::call("NotEqual", None),
            BinaryOp::LessThan => Expr::call("LessThan", None),
            BinaryOp::LessOrEqual => Expr::call("LessOrEqual", None),
            BinaryOp::GreaterThan => Expr::call("GreaterThan", None),
            BinaryOp::GreaterOrEqual => Expr::call("GreaterOrEqual", None),
            BinaryOp::BitAnd => Expr::call("BitAnd", None),
            BinaryOp::BitOr => Expr::call("BitOr", None),
        }
    }
}

impl ToEgglogExpr for Const {
    fn to_egglog_expr(&self) -> Expr {
        match self {
            Const::Char(i) => Expr::call("Integer", Some(Expr::lit(*i as i64))),
            Const::UChar(i) => Expr::call("Integer", Some(Expr::lit(*i as i64))),
            Const::Int(i) => Expr::call("Integer", Some(Expr::lit(*i as i64))),
            Const::Long(i) => Expr::call("Integer", Some(Expr::lit(*i))),
            Const::Uint(i) => Expr::call("Integer", Some(Expr::lit(*i as i64))),
            Const::Ulong(i) => Expr::call("Integer", Some(Expr::lit(*i as i64))),
            Const::Double(d) => Expr::call("Double", Some(Expr::lit(OrderedFloat::from(*d)))),
        }
    }
}

impl ToEgglogExpr for Val {
    fn to_egglog_expr(&self) -> Expr {
        match self {
            Val::Constant(c) => Expr::call("Constant", Some(c.to_egglog_expr())),
            // TODO: Look up symbol table to check if it's a static variable
            Val::Var(var) => Expr::call("Var", Some(Expr::lit(Symbol::from(var.as_str())))),
        }
    }
}

impl ToEgglogExpr for Instruction {
    fn to_egglog_expr(&self) -> Expr {
        match self {
            Instruction::Nop => Expr::call("Nop", None),
            Instruction::Return(val) => {
                if let Some(val) = val {
                    Expr::call(
                        "Return",
                        Some(Expr::call("Some", Some(val.to_egglog_expr()))),
                    )
                } else {
                    Expr::call("Return", Some(Expr::call("None", None)))
                }
            }
            Instruction::Cast { src, dst } => {
                Expr::call("Cast", vec![src.to_egglog_expr(), dst.to_egglog_expr()])
            }
            Instruction::Unary { op, src, dst } => Expr::call(
                "Unary",
                vec![
                    op.to_egglog_expr(),
                    src.to_egglog_expr(),
                    dst.to_egglog_expr(),
                ],
            ),
            Instruction::Binary { op, lhs, rhs, dst } => Expr::call(
                "Binary",
                vec![
                    op.to_egglog_expr(),
                    lhs.to_egglog_expr(),
                    rhs.to_egglog_expr(),
                    dst.to_egglog_expr(),
                ],
            ),
            Instruction::Copy { src, dst } => {
                Expr::call("Copy", vec![src.to_egglog_expr(), dst.to_egglog_expr()])
            }
            Instruction::GetAddress { src, dst } => Expr::call(
                "GetAddress",
                vec![src.to_egglog_expr(), dst.to_egglog_expr()],
            ),
            Instruction::Load { src, dst } => {
                Expr::call("Load", vec![src.to_egglog_expr(), dst.to_egglog_expr()])
            }
            Instruction::Store { src, dst } => {
                Expr::call("Store", vec![src.to_egglog_expr(), dst.to_egglog_expr()])
            }
            Instruction::Jump(l) => Expr::call("Jump", vec![Expr::lit(Symbol::from(l.as_str()))]),
            Instruction::JumpIfZero { src, dst } => Expr::call(
                "JumpIfZero",
                vec![src.to_egglog_expr(), Expr::lit(Symbol::from(dst.as_str()))],
            ),
            Instruction::JumpIfNotZero { src, dst } => Expr::call(
                "JumpIfNotZero",
                vec![src.to_egglog_expr(), Expr::lit(Symbol::from(dst.as_str()))],
            ),
            Instruction::Label(l) => Expr::call("Label", vec![Expr::lit(Symbol::from(l.as_str()))]),
            Instruction::FunCall { callee, args, dst } => {
                let dst = if let Some(dst) = dst {
                    Expr::call("Some", Some(dst.to_egglog_expr()))
                } else {
                    Expr::call("None", None)
                };

                Expr::call(
                    "FunCall",
                    vec![
                        callee.to_egglog_expr(),
                        Expr::call(
                            "vec-of",
                            args.iter()
                                .map(|arg| arg.to_egglog_expr())
                                .collect::<Vec<_>>(),
                        ),
                        dst,
                    ],
                )
            }
            Instruction::AddPtr {
                ptr,
                index,
                scale,
                dst,
            } => Expr::call(
                "AddPtr",
                vec![
                    ptr.to_egglog_expr(),
                    index.to_egglog_expr(),
                    Expr::lit(*scale as i64),
                    dst.to_egglog_expr(),
                ],
            ),
            Instruction::CopyToOffset { src, dst, offset } => Expr::call(
                "CopyToOffset",
                vec![
                    src.to_egglog_expr(),
                    dst.to_egglog_expr(),
                    Expr::lit(*offset as i64),
                ],
            ),
            Instruction::CopyFromOffset { src, offset, dst } => Expr::call(
                "CopyFromOffset",
                vec![
                    src.to_egglog_expr(),
                    Expr::lit(*offset as i64),
                    dst.to_egglog_expr(),
                ],
            ),
        }
    }
}

impl<'a> ToEgglogExpr for Ssa<'a, Instruction> {
    fn to_egglog_expr(&self) -> Expr {
        let mut exprs = vec![];
        for (id, node) in &self.cfg.nodes {
            let phis = self.phi[id].iter().map(|(name, table)| {
                let table = table
                    .iter()
                    .map(|(pred, val)| {
                        Expr::call(
                            "P",
                            vec![
                                Expr::call("Var", Some(Expr::lit(Symbol::from(val.as_str())))),
                                Expr::lit(*pred as i64),
                            ],
                        )
                    })
                    .collect::<Vec<_>>();

                Expr::call(
                    "Phi",
                    vec![
                        Expr::call("Var", Some(Expr::lit(Symbol::from(name.as_str())))),
                        Expr::call("vec-of", table),
                    ],
                )
            });

            if let Some(i @ Instruction::Label(_)) = node.instructions.first() {
                exprs.push(i.to_egglog_expr());
                exprs.extend(phis);
                exprs.extend(node.instructions.iter().skip(1).map(|i| i.to_egglog_expr()));
            } else {
                exprs.extend(phis);
                exprs.extend(node.instructions.iter().map(|i| i.to_egglog_expr()));
            }
        }

        let v = Expr::call("vec-of", exprs);
        Expr::call("Block", vec![v])
    }
}

fn node_egglog<'a>(ssa: &Ssa<'a, Instruction>, id: usize) -> Expr {
    let mut exprs = vec![];
    let node = &ssa.cfg.nodes[&id];
    let phis = ssa.phi[&id].iter().map(|(name, table)| {
        let table = table
            .iter()
            .map(|(pred, val)| {
                Expr::call(
                    "P",
                    vec![
                        Expr::call("Var", Some(Expr::lit(Symbol::from(val.as_str())))),
                        Expr::lit(*pred as i64),
                    ],
                )
            })
            .collect::<Vec<_>>();

        Expr::call(
            "Phi",
            vec![
                Expr::lit(Symbol::from(name.as_str())),
                Expr::call("vec-of", table),
            ],
        )
    });

    if let Some(i @ Instruction::Label(_)) = node.instructions.first() {
        exprs.push(i.to_egglog_expr());
        exprs.extend(phis);
        exprs.extend(node.instructions.iter().skip(1).map(|i| i.to_egglog_expr()));
    } else {
        exprs.extend(phis);
        exprs.extend(node.instructions.iter().map(|i| i.to_egglog_expr()));
    }

    Expr::call("Block", vec![Expr::call("vec-of", exprs)])
}

pub fn do_egglog<'a>(ssa: &Ssa<'a, Instruction>) -> Vec<Instruction> {
    const PRELUDE: &str = include_str!("./prelude.egg");

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(PRELUDE).unwrap();

    let values = ssa
        .cfg
        .nodes
        .iter()
        .map(|(id, _)| {
            let (_, v) = egraph.eval_expr(&node_egglog(ssa, *id)).unwrap();
            (*id, v)
        })
        .collect::<BTreeMap<usize, _>>();

    egraph.parse_and_run_program("(run 1000)").unwrap();

    let label_map = ssa
        .cfg
        .label_map
        .iter()
        .filter_map(|(l, id)| {
            if let NodeId::Block(id) = id {
                Some((l.clone(), *id))
            } else {
                None
            }
        })
        .collect();
    reconstruct(&label_map, &egraph, &values)
}

trait FromEgglog {
    fn from_egglog(termdag: &TermDag, term: &Term) -> Self;
}

impl FromEgglog for i64 {
    fn from_egglog(_termdag: &TermDag, term: &Term) -> Self {
        match term {
            Term::Lit(Literal::Int(i)) => *i,
            _ => panic!(),
        }
    }
}

impl FromEgglog for f64 {
    fn from_egglog(_termdag: &TermDag, term: &Term) -> Self {
        match term {
            Term::Lit(Literal::F64(f)) => f.into_inner(),
            _ => panic!(),
        }
    }
}

impl FromEgglog for Const {
    fn from_egglog(termdag: &TermDag, term: &Term) -> Self {
        match term {
            Term::App(head, args) => match head.as_str() {
                // TODO: Handle other types
                "Integer" => Const::Int(i64::from_egglog(termdag, &termdag.get(args[0])) as i32),
                "Double" => Const::Double(f64::from_egglog(termdag, &termdag.get(args[0]))),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }
}

impl FromEgglog for EcoString {
    fn from_egglog(_termdag: &TermDag, term: &Term) -> Self {
        match term {
            Term::Lit(Literal::String(s)) => s.as_str().into(),
            _ => panic!(),
        }
    }
}

impl FromEgglog for Val {
    fn from_egglog(termdag: &TermDag, term: &Term) -> Self {
        match term {
            Term::App(head, args) => match head.as_str() {
                "Constant" => Val::Constant(Const::from_egglog(termdag, &termdag.get(args[0]))),
                "Var" => Val::Var(EcoString::from_egglog(termdag, &termdag.get(args[0]))),
                "Static" => Val::Var(EcoString::from_egglog(termdag, &termdag.get(args[0]))),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }
}

impl FromEgglog for UnaryOp {
    fn from_egglog(_termdag: &TermDag, term: &Term) -> Self {
        match term {
            Term::App(head, _) => match head.as_str() {
                "Negate" => UnaryOp::Negate,
                "Complement" => UnaryOp::Complement,
                "Not" => UnaryOp::Not,
                _ => panic!(),
            },
            _ => panic!(),
        }
    }
}

impl FromEgglog for BinaryOp {
    fn from_egglog(_termdag: &TermDag, term: &Term) -> Self {
        match term {
            Term::App(head, _) => match head.as_str() {
                "Add" => BinaryOp::Add,
                "Subtract" => BinaryOp::Subtract,
                "Multiply" => BinaryOp::Multiply,
                "Divide" => BinaryOp::Divide,
                "Remainder" => BinaryOp::Remainder,
                "Equal" => BinaryOp::Equal,
                "NotEqual" => BinaryOp::NotEqual,
                "LessThan" => BinaryOp::LessThan,
                "LessOrEqual" => BinaryOp::LessOrEqual,
                "GreaterThan" => BinaryOp::GreaterThan,
                "GreaterOrEqual" => BinaryOp::GreaterOrEqual,
                _ => panic!(),
            },
            _ => panic!(),
        }
    }
}

impl FromEgglog for Instruction {
    fn from_egglog(termdag: &TermDag, term: &Term) -> Self {
        if let Term::App(head, args) = term {
            match head.as_str() {
                "Nop" => Instruction::Nop,
                "Return" => {
                    if let Term::App(head, args) = &termdag.get(args[0]) {
                        match head.as_str() {
                            "Some" => Instruction::Return(Some(Val::from_egglog(
                                termdag,
                                &termdag.get(args[0]),
                            ))),
                            "None" => Instruction::Return(None),
                            _ => panic!(),
                        }
                    } else {
                        panic!();
                    }
                }
                "Cast" => Instruction::Cast {
                    src: Val::from_egglog(termdag, &termdag.get(args[0])),
                    dst: Val::from_egglog(termdag, &termdag.get(args[1])),
                },
                "Unary" => {
                    let op = UnaryOp::from_egglog(termdag, &termdag.get(args[0]));
                    let src = Val::from_egglog(termdag, &termdag.get(args[1]));
                    let dst = Val::from_egglog(termdag, &termdag.get(args[2]));

                    Instruction::Unary { op, src, dst }
                }
                "Binary" => {
                    let op = BinaryOp::from_egglog(termdag, &termdag.get(args[0]));
                    let lhs = Val::from_egglog(termdag, &termdag.get(args[1]));
                    let rhs = Val::from_egglog(termdag, &termdag.get(args[2]));
                    let dst = Val::from_egglog(termdag, &termdag.get(args[3]));

                    Instruction::Binary { op, lhs, rhs, dst }
                }
                "Copy" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let dst = Val::from_egglog(termdag, &termdag.get(args[1]));

                    Instruction::Copy { src, dst }
                }
                "GetAddress" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let dst = Val::from_egglog(termdag, &termdag.get(args[1]));

                    Instruction::GetAddress { src, dst }
                }
                "Load" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let dst = Val::from_egglog(termdag, &termdag.get(args[1]));

                    Instruction::Load { src, dst }
                }
                "Store" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let dst = Val::from_egglog(termdag, &termdag.get(args[1]));

                    Instruction::Store { src, dst }
                }
                "Jump" => {
                    let l = EcoString::from_egglog(termdag, &termdag.get(args[0]));

                    Instruction::Jump(l)
                }
                "JumpIfZero" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let dst = EcoString::from_egglog(termdag, &termdag.get(args[1]));

                    Instruction::JumpIfZero { src, dst }
                }
                "JumpIfNotZero" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let dst = EcoString::from_egglog(termdag, &termdag.get(args[1]));

                    Instruction::JumpIfNotZero { src, dst }
                }
                "Label" => {
                    let l = EcoString::from_egglog(termdag, &termdag.get(args[0]));

                    Instruction::Label(l)
                }
                "FunCall" => {
                    let callee = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let fargs = if let Term::App(head, args) = &termdag.get(args[1]) {
                        match head.as_str() {
                            "vec-of" => args
                                .iter()
                                .map(|arg| Val::from_egglog(termdag, &termdag.get(*arg)))
                                .collect::<Vec<_>>(),
                            _ => panic!(),
                        }
                    } else {
                        panic!();
                    };

                    let dst = if let Term::App(head, args) = &termdag.get(args[2]) {
                        match head.as_str() {
                            "Some" => Some(Val::from_egglog(termdag, &termdag.get(args[0]))),
                            "None" => None,
                            _ => panic!(),
                        }
                    } else {
                        panic!();
                    };

                    Instruction::FunCall {
                        callee,
                        args: fargs,
                        dst,
                    }
                }
                "AddPtr" => {
                    let ptr = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let index = Val::from_egglog(termdag, &termdag.get(args[1]));
                    let scale = i64::from_egglog(termdag, &termdag.get(args[2])) as usize;
                    let dst = Val::from_egglog(termdag, &termdag.get(args[3]));

                    Instruction::AddPtr {
                        ptr,
                        index,
                        scale,
                        dst,
                    }
                }
                "CopyToOffset" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let dst = Val::from_egglog(termdag, &termdag.get(args[1]));
                    let offset = i64::from_egglog(termdag, &termdag.get(args[2])) as usize;

                    Instruction::CopyToOffset { src, dst, offset }
                }
                "CopyFromOffset" => {
                    let src = Val::from_egglog(termdag, &termdag.get(args[0]));
                    let offset = i64::from_egglog(termdag, &termdag.get(args[1])) as usize;
                    let dst = Val::from_egglog(termdag, &termdag.get(args[2]));

                    Instruction::CopyFromOffset { src, offset, dst }
                }
                _ => panic!(),
            }
        } else {
            panic!();
        }
    }
}

struct Phi {
    name: EcoString,
    table: HashMap<usize, Val>,
}

impl FromEgglog for Phi {
    fn from_egglog(termdag: &TermDag, term: &Term) -> Self {
        if let Term::App(head, args) = term {
            match head.as_str() {
                "Phi" => {
                    let name = EcoString::from_egglog(termdag, &termdag.get(args[0]));

                    let table = if let Term::App(head, args) = &termdag.get(args[1]) {
                        match head.as_str() {
                            "vec-of" => args
                                .iter()
                                .map(|arg| {
                                    if let Term::App(head, args) = &termdag.get(*arg) {
                                        match head.as_str() {
                                            "P" => {
                                                let val = Val::from_egglog(
                                                    termdag,
                                                    &termdag.get(args[0]),
                                                );
                                                let pred = i64::from_egglog(
                                                    termdag,
                                                    &termdag.get(args[1]),
                                                )
                                                    as usize;

                                                (pred, val)
                                            }
                                            _ => panic!(),
                                        }
                                    } else {
                                        panic!();
                                    }
                                })
                                .collect::<HashMap<_, _>>(),
                            _ => panic!(),
                        }
                    } else {
                        panic!();
                    };

                    Self { name, table }
                }
                _ => panic!(),
            }
        } else {
            panic!();
        }
    }
}

fn parse_egglog_block(
    egraph: &EGraph,
    value: Value,
) -> (HashMap<EcoString, HashMap<usize, Val>>, Vec<Instruction>) {
    let (termdag, term) = egraph.extract_value(value);

    let term = if let Term::App(head, args) = &term {
        if head.as_str() != "Block" {
            panic!();
        }
        termdag.get(args[0])
    } else {
        panic!();
    };

    let v = match &term {
        Term::App(head, args) => match head.as_str() {
            "vec-of" => args.as_slice(),
            _ => &[],
        },
        _ => panic!(),
    };

    let mut insts = Vec::new();
    let mut index = 0;

    if index < v.len() {
        if let Term::App(head, _) = termdag.get(v[index]) {
            match head.as_str() {
                "Label" => {
                    insts.push(Instruction::from_egglog(&termdag, &termdag.get(v[index])));
                    index += 1;
                }
                _ => {}
            }
        }
    }

    let mut phis = HashMap::new();
    while index < v.len() {
        if let Term::App(head, _) = termdag.get(v[index]) {
            match head.as_str() {
                "Phi" => {
                    let phi = Phi::from_egglog(&termdag, &termdag.get(v[index]));
                    phis.insert(phi.name, phi.table);
                    index += 1;
                }
                _ => {
                    break;
                }
            }
        }
    }

    for i in &v[index..] {
        insts.push(Instruction::from_egglog(&termdag, &termdag.get(*i)));
    }

    (phis, insts)
}

fn reconstruct(
    label_map: &HashMap<EcoString, usize>,
    egraph: &EGraph,
    map: &BTreeMap<usize, Value>,
) -> Vec<Instruction> {
    let mut phis = HashMap::new();
    let mut insts_map = BTreeMap::new();

    for (id, value) in map {
        let (p, insts) = parse_egglog_block(egraph, value.clone());
        phis.insert(*id, p);
        insts_map.insert(*id, insts);
    }

    let mut succs = HashMap::new();

    for (&id, insts) in &insts_map {
        match insts.last() {
            Some(Instruction::Jump(l)) => {
                let to = label_map[l];
                succs.entry(id).or_insert_with(HashSet::new).insert(to);
            }
            Some(Instruction::JumpIfZero { dst, .. }) => {
                let to = label_map[dst];
                succs.entry(id).or_insert_with(HashSet::new).insert(to);

                if let Some(next) = insts_map.keys().skip_while(|i| **i <= id).next() {
                    succs.entry(id).or_insert_with(HashSet::new).insert(*next);
                }
            }
            Some(Instruction::JumpIfNotZero { dst, .. }) => {
                let to = label_map[dst];
                succs.entry(id).or_insert_with(HashSet::new).insert(to);
                if let Some(next) = insts_map.keys().skip_while(|i| **i <= id).next() {
                    succs.entry(id).or_insert_with(HashSet::new).insert(*next);
                }
            }
            _ => {
                if let Some(next) = insts_map.keys().skip_while(|i| **i <= id).next() {
                    succs.entry(id).or_insert_with(HashSet::new).insert(*next);
                }
            }
        }
    }

    for (&id, insts) in &mut insts_map {
        if let Some(succs) = succs.get(&id) {
            for succ in succs {
                let phi = phis.get(succ).unwrap();

                for (name, table) in phi {
                    let incoming = &table[&id];

                    insts.push(Instruction::Copy {
                        src: incoming.clone(),
                        dst: Val::Var(name.clone()),
                    });
                }
            }
        }
    }

    let mut insts = Vec::new();

    for (_, mut i) in insts_map {
        insts.append(&mut i);
    }

    insts
}

#[test]
fn test_egglog() {
    use egglog::ast::{Action, Command};
    const PRELUDE: &str = include_str!("./prelude.egg");

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(PRELUDE).unwrap();

    let expr = Expr::lit(42);

    let (_, v) = egraph.eval_expr(&expr).unwrap();

    let test_command = Command::Action(Action::Let((), "x".into(), expr));

    egraph.run_program(vec![test_command]).unwrap();

    dbg!(egraph.extract_value_to_string(v));
}
