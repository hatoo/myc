use ecow::EcoString;

pub struct Program {
    pub function_definition: Function,
}

pub struct Function {
    pub name: EcoString,
    pub body: Statement,
}

pub enum Statement {
    Return(Expression),
}

pub enum Expression {
    Constant(i32),
}
