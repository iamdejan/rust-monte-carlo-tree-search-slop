#![deny(unused_variables)]
#![deny(unused_imports)]

use std::collections::HashMap;

use crate::grid_world;

// ============================================================================
// MCTS Tree Node
// ============================================================================

#[derive(Clone)]
pub struct MctsNode {
    pub state: grid_world::State,
    pub parent: Option<usize>,
    // Store children as (action, state) -> child_index to handle duplicate states from different actions
    pub children: Vec<(grid_world::Action, usize)>,
    pub visit_count: u32,
    pub total_reward: f64,
    pub untried_actions: Vec<grid_world::Action>,
    pub is_terminal: bool,
}

impl MctsNode {
    pub fn new(
        state: grid_world::State,
        parent: Option<usize>,
        actions: Vec<grid_world::Action>,
        is_terminal: bool,
    ) -> Self {
        return MctsNode {
            state,
            parent,
            children: Vec::new(),
            visit_count: 0,
            total_reward: 0.0,
            untried_actions: actions,
            is_terminal,
        };
    }

    pub fn average_reward(&self) -> f64 {
        if self.visit_count == 0 {
            return 0.0;
        } else {
            return self.total_reward / self.visit_count as f64;
        }
    }

    pub fn is_fully_expanded(&self) -> bool {
        return self.untried_actions.is_empty();
    }

    pub fn get_child_by_action(&self, action: grid_world::Action) -> Option<usize> {
        for (a, idx) in &self.children {
            if *a == action {
                return Some(*idx);
            }
        }
        return None;
    }
}

// ============================================================================
// MCTS Tree
// ============================================================================

pub struct MctsTree {
    nodes: Vec<MctsNode>,
    // Map state to all node indices (there can be multiple nodes for same state from different paths)
    state_to_indices: HashMap<grid_world::State, Vec<usize>>,
}

impl MctsTree {
    pub fn new() -> Self {
        return MctsTree {
            nodes: Vec::new(),
            state_to_indices: HashMap::new(),
        };
    }

    pub fn add_node(&mut self, node: MctsNode) -> usize {
        let index = self.nodes.len();
        let state = node.state.clone();
        self.nodes.push(node);
        self.state_to_indices.entry(state).or_default().push(index);
        return index;
    }

    pub fn get_node(&self, index: usize) -> &MctsNode {
        return &self.nodes[index];
    }

    pub fn get_node_mut(&mut self, index: usize) -> &mut MctsNode {
        return &mut self.nodes[index];
    }

    pub fn get_nodes_by_state(&self, state: &grid_world::State) -> Vec<usize> {
        return self
            .state_to_indices
            .get(state)
            .cloned()
            .unwrap_or_default();
    }

    pub fn num_nodes(&self) -> usize {
        return self.nodes.len();
    }
}
