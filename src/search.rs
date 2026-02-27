#![deny(unused_variables)]
#![deny(unused_imports)]

use rand::SeedableRng;
use rand::prelude::*;

use crate::grid_world;
use crate::mcts;

// ============================================================================
// Monte-Carlo Tree Search
// ============================================================================

const EXPLORATION_CONSTANT: f64 = 1.414; // sqrt(2)
const MAX_TREE_SIZE: usize = 10000;

pub struct Mcts {
    pub tree: mcts::MctsTree,
    environment: grid_world::GridWorld,
    max_depth: usize,
    rng: StdRng,
}

impl Mcts {
    pub fn new(environment: grid_world::GridWorld, seed: u64, max_depth: usize) -> Self {
        return Mcts {
            tree: mcts::MctsTree::new(),
            environment,
            max_depth,
            rng: StdRng::seed_from_u64(seed),
        };
    }

    // Selection: traverse tree using UCB1 until we find a node that can be expanded
    fn selection(&self, node_index: usize) -> usize {
        let mut current_index = node_index;
        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > 1000 {
                // Safety limit to prevent infinite loop
                return current_index;
            }

            let node = self.tree.get_node(current_index);

            // If terminal or not fully expanded, stop selection
            if node.is_terminal || !node.is_fully_expanded() {
                return current_index;
            }

            // Node is fully expanded, select best child using UCB1
            let mut best_action = grid_world::Action::Up;
            let mut best_value = f64::NEG_INFINITY;
            let parent_visits = node.visit_count;

            for (action, child_index) in &node.children {
                let child = self.tree.get_node(*child_index);
                let ucb_value = self.ucb1(child, parent_visits);

                if ucb_value > best_value {
                    best_value = ucb_value;
                    best_action = *action;
                }
            }

            // Move to best child
            if let Some(next_index) = node.get_child_by_action(best_action) {
                current_index = next_index;
            } else {
                // Should not happen if node is fully expanded
                return current_index;
            }
        }
    }

    fn ucb1(&self, node: &mcts::MctsNode, parent_visits: u32) -> f64 {
        if node.visit_count == 0 {
            return f64::INFINITY;
        } else {
            let exploitation = node.average_reward();
            let exploration = EXPLORATION_CONSTANT
                * ((parent_visits as f64).ln() / node.visit_count as f64).sqrt();
            return exploitation + exploration;
        }
    }

    fn expansion(&mut self, node_index: usize) -> usize {
        let node = self.tree.get_node(node_index).clone();

        if node.is_terminal {
            return node_index;
        }

        // Check tree size limit
        if self.tree.num_nodes() >= MAX_TREE_SIZE {
            return node_index;
        }

        // Select a random untried action
        let action_idx = self.rng.gen_range(0..node.untried_actions.len());
        let action = node.untried_actions[action_idx];

        // Remove the selected action from untried
        {
            let node_mut = self.tree.get_node_mut(node_index);
            node_mut.untried_actions.remove(action_idx);
        }

        // Create new state
        let new_state = self.environment.transition(&node.state, &action);

        // Get actions for new state
        let new_actions = if self.environment.is_terminal(&new_state) {
            Vec::new()
        } else {
            self.environment.get_actions(&new_state)
        };

        let is_terminal = self.environment.is_terminal(&new_state);

        // Create new node
        let new_node = mcts::MctsNode::new(
            new_state.clone(),
            Some(node_index),
            new_actions,
            is_terminal,
        );

        let new_index = self.tree.add_node(new_node);

        // Add to parent's children
        let node_mut = self.tree.get_node_mut(node_index);
        node_mut.children.push((action, new_index));

        return new_index;
    }

    // FIXED: Changed to `&mut self`, removed hardcoded RNG, added discount factor
    fn simulation(&mut self, node_index: usize) -> f64 {
        // Clone the state so we don't hold a reference to self.tree
        let mut current_state = self.tree.get_node(node_index).state.clone();

        if self.environment.is_terminal(&current_state) {
            return self.environment.reward(&current_state);
        }

        let mut depth = 0;
        let gamma = 0.95_f64; // Discount factor for longer paths

        while depth < self.max_depth && !self.environment.is_terminal(&current_state) {
            let actions = self.environment.get_actions(&current_state);

            if actions.is_empty() {
                break;
            }

            // Use the instance's RNG, ensuring diverse simulations
            let idx = self.rng.gen_range(0..actions.len());
            let action = actions[idx];
            current_state = self.environment.transition(&current_state, &action);
            depth += 1;
        }

        // Apply discount factor so the agent prefers shorter paths to the goal
        return self.environment.reward(&current_state) * gamma.powi(depth as i32);
    }

    fn backpropagation(&mut self, node_index: usize, reward: f64) {
        let mut current_index = Some(node_index);

        while let Some(idx) = current_index {
            let node = self.tree.get_node_mut(idx);
            node.visit_count += 1;
            node.total_reward += reward;
            current_index = node.parent;
        }
    }

    fn run_iteration(&mut self, root_index: usize) -> bool {
        if self.tree.num_nodes() >= MAX_TREE_SIZE {
            return false;
        }

        let selected_index = self.selection(root_index);

        {
            let node = self.tree.get_node(selected_index);
            if node.is_terminal {
                let reward = self.environment.reward(&node.state);
                self.backpropagation(selected_index, reward);
                return true;
            }
        }

        let new_index = self.expansion(selected_index);
        let reward = self.simulation(new_index);
        self.backpropagation(new_index, reward);

        return true;
    }

    pub fn run(&mut self, root_state: &grid_world::State, num_iterations: usize) {
        let root_index = if let Some(idx) = self.tree.get_nodes_by_state(root_state).first() {
            *idx
        } else {
            let actions = self.environment.get_actions(root_state);
            let is_terminal = self.environment.is_terminal(root_state);
            let root_node = mcts::MctsNode::new(root_state.clone(), None, actions, is_terminal);
            self.tree.add_node(root_node)
        };

        for i in 0..num_iterations {
            if !self.run_iteration(root_index) {
                eprintln!("Warning: Tree size limit reached at iteration {}", i);
                break;
            }
        }
    }

    pub fn get_best_action(
        &mut self,
        root_state: &grid_world::State,
    ) -> Option<grid_world::Action> {
        let root_indices = self.tree.get_nodes_by_state(root_state);
        if root_indices.is_empty() {
            return None;
        }

        let root_index = root_indices[0];
        let root = self.tree.get_node(root_index);

        if root.children.is_empty() {
            return None;
        }

        let mut best_action = grid_world::Action::Up;
        let mut best_visits = 0u32;

        for (action, child_index) in &root.children {
            let child = self.tree.get_node(*child_index);
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_action = *action;
            }
        }

        return Some(best_action);
    }

    pub fn get_statistics(&self, state: &grid_world::State) -> Option<(u32, f64)> {
        let indices = self.tree.get_nodes_by_state(state);
        if indices.is_empty() {
            return None;
        }
        let node = self.tree.get_node(indices[0]);
        return Some((node.visit_count, node.average_reward()));
    }
}

pub fn generate_policy_string(seed: u64, num_iterations: usize, max_depth: usize) -> String {
    let environment = grid_world::GridWorld::new();
    let mut output = String::new();

    for r in 0..environment.rows {
        for c in 0..environment.cols {
            let state = grid_world::State { row: r, col: c };

            // Handle terminal and blocked states
            if environment.is_terminal(&state) {
                if state == environment.positive_reward {
                    output.push_str(" +1 ");
                } else {
                    output.push_str(" -1 ");
                }
            } else if state == environment.blocked {
                output.push_str(" XX ");
            } else {
                // For valid states, run MCTS to find the best action
                let cell_seed = seed + (r * environment.cols + c) as u64;
                let mut mcts = Mcts::new(grid_world::GridWorld::new(), cell_seed, max_depth);

                mcts.run(&state, num_iterations);

                if let Some(action) = mcts.get_best_action(&state) {
                    let symbol = match action {
                        grid_world::Action::Up => "  ^ ",
                        grid_world::Action::Down => "  v ",
                        grid_world::Action::Left => "  < ",
                        grid_world::Action::Right => "  > ",
                    };
                    output.push_str(symbol);
                } else {
                    output.push_str("  ? "); // Dead end / no actions
                }
            }
        }
        output.push('\n');
    }

    return output;
}

pub fn visualize_policy(seed: u64, num_iterations: usize, max_depth: usize) {
    println!("\n=== Grid Policy Visualization ===");
    println!("Legend: [^ v < >] = Best Action, [XX] = Blocked, [+1/-1] = Terminal");
    println!("---------------------------------");
    print!(
        "{}",
        generate_policy_string(seed, num_iterations, max_depth)
    );
    println!("---------------------------------");
}
