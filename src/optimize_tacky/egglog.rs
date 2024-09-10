use std::f32::consts::{E, PI};

use ecow::vec;
use egglog::ast::{Action, Command, Expr, Symbol};
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
        }

        Expr::call("vec-of", exprs)
    }
}

pub fn do_egglog<'a>(ssa: &Ssa<'a, Instruction>) {
    const PRELUDE: &str = include_str!("./prelude.egg");

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(PRELUDE).unwrap();

    let program = ssa.to_egglog_expr();

    let (_, v) = egraph.eval_expr(&program).unwrap();

    egraph.parse_and_run_program("(run 1000)").unwrap();

    println!("{}", egraph.extract_value_to_string(v));
}

#[test]
fn test_egglog() {
    const PRELUDE: &str = include_str!("./prelude.egg");

    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(PRELUDE).unwrap();

    let expr = Expr::lit(42);

    let (_, v) = egraph.eval_expr(&expr).unwrap();

    let test_command = Command::Action(Action::Let((), "x".into(), expr));

    egraph.run_program(vec![test_command]).unwrap();

    dbg!(egraph.extract_value_to_string(v));
}
