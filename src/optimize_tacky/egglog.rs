use std::f32::consts::E;

use ecow::vec;
use egglog::ast::{Action, Command, Expr, Symbol};
use ordered_float::OrderedFloat;

use crate::tacky::{BinaryOp, UnaryOp, Val};

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

impl ToEgglogExpr for i64 {
    fn to_egglog_expr(&self) -> Expr {
        Expr::lit(*self)
    }
}

impl ToEgglogExpr for f64 {
    fn to_egglog_expr(&self) -> Expr {
        Expr::lit(OrderedFloat::from(*self))
    }
}

impl ToEgglogExpr for Val {
    fn to_egglog_expr(&self) -> Expr {
        match self {
            Val::Constant(_) => todo!(),
            Val::Var(var) => Expr::call(
                "Var",
                std::iter::once(Expr::lit(Symbol::from(var.as_str()))),
            ),
        }
    }
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
