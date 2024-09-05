use std::collections::HashMap;

use super::NodeId;

#[derive(Debug, Clone, Default)]
pub struct DisjointSet {
    map: HashMap<NodeId, NodeId>,
}

impl DisjointSet {
    pub fn find(&mut self, x: &NodeId) -> NodeId {
        match x {
            NodeId::Register(_) => x.clone(),
            NodeId::Pseudo(_) => {
                if let Some(p) = self.map.get(x) {
                    if let NodeId::Register(_) = p {
                        return p.clone();
                    }
                    if p == x {
                        return x.clone();
                    }
                    let p = p.clone();
                    let x = x.clone();
                    let root = self.find(&p);
                    self.map.insert(x, root.clone());
                    root
                } else {
                    x.clone()
                }
            }
        }
    }

    pub fn merge(&mut self, x: &NodeId, y: &NodeId) {
        match (x, y) {
            (NodeId::Register(_), NodeId::Register(_)) => panic!(),
            (NodeId::Register(_), NodeId::Pseudo(_)) => {
                self.map.insert(y.clone(), x.clone());
            }
            (NodeId::Pseudo(_), NodeId::Register(_)) => {
                self.map.insert(x.clone(), y.clone());
            }
            _ => {
                let x = self.find(x);
                let y = self.find(y);
                self.map.insert(x, y);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
