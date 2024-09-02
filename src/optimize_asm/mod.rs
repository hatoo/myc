use std::collections::HashMap;

use ecow::EcoString;

use crate::codegen::Register;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NodeId {
    Register(Register),
    Pseudo(EcoString),
}

#[derive(Debug, Clone)]
struct Node {
    id: NodeId,
    neighbors: Vec<NodeId>,
    spill_cost: f32,
    color: Option<usize>,
    pruned: bool,
}

struct Graph {
    map: HashMap<NodeId, Node>,
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

impl Graph {
    fn base() -> Self {
        const REGISTERS: [Register; 11] = [
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

        let mut map = HashMap::new();

        for &reg in &REGISTERS {
            map.insert(NodeId::Register(reg), Node::new(NodeId::Register(reg)));
        }

        for &reg in &REGISTERS {
            for &neighbor in &REGISTERS {
                if reg != neighbor {
                    map.get_mut(&NodeId::Register(reg))
                        .unwrap()
                        .neighbors
                        .push(NodeId::Register(neighbor));
                }
            }
        }

        Self { map }
    }
}
