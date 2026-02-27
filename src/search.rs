#![deny(unused_variables)]
#![deny(unused_imports)]

//! Monte-Carlo Tree Search algorithm implementation module.
//!
//! This module implements the core MCTS algorithm with the four main phases:
//! Selection, Expansion, Simulation, and Backpropagation.
//! It also provides visualization utilities for the learned policy.

// SeedableRng allows creating RNG with a fixed seed for reproducibility
use rand::SeedableRng;
// prelude imports common RNG utilities like gen_range
use rand::prelude::*;

use crate::grid_world;
use crate::mcts;

// ============================================================================
// Monte-Carlo Tree Search
// ============================================================================

/// The exploration constant for UCB1 (Upper Confidence Bound).
///
/// This value is √2 ≈ 1.414, which is commonly used for rewards in [0, 1].
/// Higher values encourage more exploration of less-visited nodes.
const EXPLORATION_CONSTANT: f64 = 1.414; // sqrt(2)

/// Maximum number of nodes allowed in the search tree.
///
/// This limit prevents memory exhaustion during long-running searches.
const MAX_TREE_SIZE: usize = 10000;

/// The main Monte-Carlo Tree Search interface.
///
/// This struct manages the search tree, environment, and random number generator
/// to perform MCTS iterations. It provides methods to run the search and extract
/// the best actions and statistics.
///
/// # Examples
///
/// ```
/// use rust_monte_carlo_tree_search::{grid_world, search};
///
/// let env = grid_world::GridWorld::new();
/// let mut mcts = search::Mcts::new(env, 12345, 50);
///
/// let start = grid_world::State { row: 1, col: 0 };
/// mcts.run(&start, 1000);
///
/// if let Some(action) = mcts.get_best_action(&start) {
///     println!("Best action: {:?}", action);
/// }
/// ```
pub struct Mcts {
    /// The search tree containing all nodes.
    pub tree: mcts::MctsTree,
    /// The environment being searched.
    environment: grid_world::GridWorld,
    /// Maximum depth for simulation rollouts.
    max_depth: usize,
    /// Seeded random number generator for reproducibility.
    rng: StdRng,
}

impl Mcts {
    /// Creates a new MCTS instance.
    ///
    /// # Arguments
    ///
    /// * `environment` - The GridWorld environment to search
    /// * `seed` - Seed for the random number generator (enables reproducible runs)
    /// * `max_depth` - Maximum number of steps in a simulation rollout
    ///
    /// # Returns
    ///
    /// A new `Mcts` instance with an empty tree
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, search};
    ///
    /// let env = grid_world::GridWorld::new();
    /// let mcts = search::Mcts::new(env, 12345, 50);
    ///
    /// assert_eq!(mcts.tree.num_nodes(), 0);
    /// ```
    pub fn new(environment: grid_world::GridWorld, seed: u64, max_depth: usize) -> Self {
        // Initialize MCTS with empty tree, cloned environment, and seeded RNG
        return Mcts {
            tree: mcts::MctsTree::new(),
            environment,
            max_depth,
            // seed_from_u64 creates a reproducible RNG sequence
            rng: StdRng::seed_from_u64(seed),
        };
    }

    /// Runs the MCTS algorithm for a specified number of iterations.
    ///
    /// Each iteration consists of four phases:
    /// 1. **Selection**: Find a promising node using UCB1
    /// 2. **Expansion**: Add a new child node
    /// 3. **Simulation**: Run a random rollout from the new node
    /// 4. **Backpropagation**: Update statistics up the tree
    ///
    /// # Arguments
    ///
    /// * `root_state` - The starting state for the search
    /// * `num_iterations` - Number of MCTS iterations to perform
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, search};
    ///
    /// let env = grid_world::GridWorld::new();
    /// let mut mcts = search::Mcts::new(env, 12345, 50);
    ///
    /// let start = grid_world::State { row: 1, col: 0 };
    /// mcts.run(&start, 100);
    ///
    /// assert!(mcts.tree.num_nodes() > 0);
    /// ```
    pub fn run(&mut self, root_state: &grid_world::State, num_iterations: usize) {
        // Check if a root node already exists for this state
        let root_index = if let Some(idx) = self.tree.get_nodes_by_state(root_state).first() {
            // Reuse existing root node
            *idx
        } else {
            // Create new root node with all possible actions
            let actions = self.environment.get_actions(root_state);
            let is_terminal = self.environment.is_terminal(root_state);
            let root_node = mcts::MctsNode::new(root_state.clone(), None, actions, is_terminal);
            self.tree.add_node(root_node)
        };

        // Run the specified number of MCTS iterations
        for i in 0..num_iterations {
            // Stop early if tree size limit reached
            if !self.run_iteration(root_index) {
                eprintln!("Warning: Tree size limit reached at iteration {}", i);
                break;
            }
        }
    }

    /// Returns the best action from the root state based on visit counts.
    ///
    /// The action leading to the most visited child node is selected.
    /// This represents the action with the highest exploitation value.
    ///
    /// # Arguments
    ///
    /// * `root_state` - The state to find the best action from
    ///
    /// # Returns
    ///
    /// - `Some(action)` - The best action found
    /// - `None` - If no root node exists or the root has no children
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, search};
    ///
    /// let env = grid_world::GridWorld::new();
    /// let mut mcts = search::Mcts::new(env, 12345, 50);
    ///
    /// let start = grid_world::State { row: 1, col: 0 };
    /// mcts.run(&start, 100);
    ///
    /// if let Some(action) = mcts.get_best_action(&start) {
    ///     println!("Best action: {:?}", action);
    /// }
    /// ```
    pub fn get_best_action(
        &mut self,
        root_state: &grid_world::State,
    ) -> Option<grid_world::Action> {
        // Find all nodes matching the root state
        let root_indices = self.tree.get_nodes_by_state(root_state);
        if root_indices.is_empty() {
            return None;
        }

        // Use the first matching node as root
        let root_index = root_indices[0];
        let root = self.tree.get_node(root_index);

        // Can't determine best action if root has no children
        if root.children.is_empty() {
            return None;
        }

        // Track the action with highest visit count (most explored = most promising)
        let mut best_action = grid_world::Action::Up;
        let mut best_visits = 0u32;

        // Iterate through all children to find the most visited
        for (action, child_index) in &root.children {
            let child = self.tree.get_node(*child_index);
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_action = *action;
            }
        }

        return Some(best_action);
    }

    /// Returns statistics for a given state.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to get statistics for
    ///
    /// # Returns
    ///
    /// - `Some((visit_count, average_reward))` - The node's statistics
    /// - `None` - If no node exists for the given state
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, search};
    ///
    /// let env = grid_world::GridWorld::new();
    /// let mut mcts = search::Mcts::new(env, 12345, 50);
    ///
    /// let start = grid_world::State { row: 1, col: 0 };
    /// mcts.run(&start, 100);
    ///
    /// if let Some((visits, reward)) = mcts.get_statistics(&start) {
    ///     println!("Visits: {}, Avg Reward: {}", visits, reward);
    /// }
    /// ```
    pub fn get_statistics(&self, state: &grid_world::State) -> Option<(u32, f64)> {
        // Look up the state in the tree
        let indices = self.tree.get_nodes_by_state(state);
        if indices.is_empty() {
            return None;
        }
        // Return visit count and average reward for the first matching node
        let node = self.tree.get_node(indices[0]);
        return Some((node.visit_count, node.average_reward()));
    }

    /// Selection phase: traverse the tree using UCB1 until finding a node to expand.
    ///
    /// Starting from the given node, this method traverses down the tree by
    /// selecting the child with the highest UCB1 value at each step. The traversal
    /// stops when reaching a terminal node or a node that is not fully expanded.
    ///
    /// # Arguments
    ///
    /// * `node_index` - The index of the starting node
    ///
    /// # Returns
    ///
    /// The index of the selected node (for expansion or terminal evaluation)
    fn selection(&self, node_index: usize) -> usize {
        let mut current_index = node_index;
        let mut iterations = 0;

        // Keep traversing until we find a node to expand or hit a terminal
        loop {
            iterations += 1;
            if iterations > 1000 {
                // Safety limit to prevent infinite loop
                return current_index;
            }

            let node = self.tree.get_node(current_index);

            // If terminal or not fully expanded, stop selection
            // Terminal nodes can't be expanded, non-fully-expanded nodes should be expanded
            if node.is_terminal || !node.is_fully_expanded() {
                return current_index;
            }

            // Node is fully expanded, select best child using UCB1
            let mut best_action = grid_world::Action::Up;
            let mut best_value = f64::NEG_INFINITY;
            let parent_visits = node.visit_count;

            // Evaluate all children using UCB1 formula
            for (action, child_index) in &node.children {
                let child = self.tree.get_node(*child_index);
                let ucb_value = self.ucb1(child, parent_visits);

                // Keep track of the highest UCB1 value
                if ucb_value > best_value {
                    best_value = ucb_value;
                    best_action = *action;
                }
            }

            // Move to best child for next iteration
            if let Some(next_index) = node.get_child_by_action(best_action) {
                current_index = next_index;
            } else {
                // Should not happen if node is fully expanded
                return current_index;
            }
        }
    }

    /// Calculates the UCB1 (Upper Confidence Bound 1) value for a node.
    ///
    /// The UCB1 formula balances exploration and exploitation:
    /// - **Exploitation**: Preferring nodes with high average reward
    /// - **Exploration**: Preferring less-visited nodes to gather more information
    ///
    /// Formula: `average_reward + EXPLORATION_CONSTANT * sqrt(ln(parent_visits) / visit_count)`
    ///
    /// # Arguments
    ///
    /// * `node` - The node to evaluate
    /// * `parent_visits` - The visit count of the parent node
    ///
    /// # Returns
    ///
    /// - `f64::INFINITY` if the node has never been visited (forces exploration)
    /// - The UCB1 value otherwise
    fn ucb1(&self, node: &mcts::MctsNode, parent_visits: u32) -> f64 {
        // Unvisited nodes have infinite UCB1 value to ensure they're explored first
        if node.visit_count == 0 {
            return f64::INFINITY;
        } else {
            // Exploitation term: how good is this node on average?
            let exploitation = node.average_reward();
            // Exploration term: how uncertain are we about this node?
            // Nodes with fewer visits get higher exploration bonus
            let exploration = EXPLORATION_CONSTANT
                * ((parent_visits as f64).ln() / node.visit_count as f64).sqrt();
            return exploitation + exploration;
        }
    }

    /// Expansion phase: add a new child node for an untried action.
    ///
    /// This method selects a random untried action from the given node,
    /// creates a new child node for the resulting state, and adds it to the tree.
    ///
    /// # Arguments
    ///
    /// * `node_index` - The index of the node to expand
    ///
    /// # Returns
    ///
    /// The index of the newly created node, or the original node index if:
    /// - The node is terminal
    /// - The tree size limit has been reached
    fn expansion(&mut self, node_index: usize) -> usize {
        // Clone node to avoid borrow checker issues with mutable refs
        let node = self.tree.get_node(node_index).clone();

        // Terminal nodes can't be expanded
        if node.is_terminal {
            return node_index;
        }

        // Check tree size limit to prevent memory issues
        if self.tree.num_nodes() >= MAX_TREE_SIZE {
            return node_index;
        }

        // Select a random untried action to explore
        let action_idx = self.rng.gen_range(0..node.untried_actions.len());
        let action = node.untried_actions[action_idx];

        // Remove the selected action from untried list (mark it as tried)
        {
            let node_mut = self.tree.get_node_mut(node_index);
            node_mut.untried_actions.remove(action_idx);
        }

        // Compute the resulting state after taking this action
        let new_state = self.environment.transition(&node.state, &action);

        // Get available actions for the new state
        // Terminal states have no actions (episode ends there)
        let new_actions = if self.environment.is_terminal(&new_state) {
            Vec::new()
        } else {
            self.environment.get_actions(&new_state)
        };

        let is_terminal = self.environment.is_terminal(&new_state);

        // Create the new child node
        let new_node = mcts::MctsNode::new(
            new_state.clone(),
            Some(node_index), // Parent is the current node
            new_actions,
            is_terminal,
        );

        // Add the new node to the tree
        let new_index = self.tree.add_node(new_node);

        // Add the child to the parent's children list
        let node_mut = self.tree.get_node_mut(node_index);
        node_mut.children.push((action, new_index));

        return new_index;
    }

    /// Simulation phase: perform a random rollout from the given node.
    ///
    /// Starting from the node's state, this method performs random actions
    /// until reaching a terminal state or the maximum depth.
    ///
    /// # Arguments
    ///
    /// * `node_index` - The index of the node to start the simulation from
    ///
    /// # Returns
    ///
    /// The discounted reward from the final state of the simulation.
    /// The reward is discounted by `gamma^depth` to encourage shorter paths.
    fn simulation(&mut self, node_index: usize) -> f64 {
        // Clone the state so we don't hold a reference to self.tree
        let mut current_state = self.tree.get_node(node_index).state.clone();

        // If starting from terminal state, just return its reward
        if self.environment.is_terminal(&current_state) {
            return self.environment.reward(&current_state);
        }

        let mut depth = 0;
        let gamma = 0.95_f64; // Discount factor for longer paths

        // Perform random rollout until terminal or max depth
        while depth < self.max_depth && !self.environment.is_terminal(&current_state) {
            let actions = self.environment.get_actions(&current_state);

            // No actions available - end simulation
            if actions.is_empty() {
                break;
            }

            // Choose random action for exploration
            let idx = self.rng.gen_range(0..actions.len());
            let action = actions[idx];
            current_state = self.environment.transition(&current_state, &action);
            depth += 1;
        }

        // Apply discount factor so the agent prefers shorter paths to the goal
        // Earlier rewards are better than later rewards
        return self.environment.reward(&current_state) * gamma.powi(depth as i32);
    }

    /// Backpropagation phase: propagate the reward up the tree.
    ///
    /// Starting from the given node, this method updates the visit count
    /// and total reward for the node and all its ancestors.
    ///
    /// # Arguments
    ///
    /// * `node_index` - The index of the node to start backpropagation from
    /// * `reward` - The reward to propagate
    fn backpropagation(&mut self, node_index: usize, reward: f64) {
        let mut current_index = Some(node_index);

        // Walk up the tree from leaf to root
        while let Some(idx) = current_index {
            let node = self.tree.get_node_mut(idx);
            // Increment visit count for this node
            node.visit_count += 1;
            // Accumulate the reward from the simulation
            node.total_reward += reward;
            // Move to parent (None when we reach root, ending the loop)
            current_index = node.parent;
        }
    }

    /// Executes a single MCTS iteration.
    ///
    /// This method runs one complete iteration of the four MCTS phases:
    /// Selection → Expansion → Simulation → Backpropagation
    ///
    /// # Arguments
    ///
    /// * `root_index` - The index of the root node
    ///
    /// # Returns
    ///
    /// - `true` if the iteration completed successfully
    /// - `false` if the tree size limit was reached
    fn run_iteration(&mut self, root_index: usize) -> bool {
        // Check tree size limit before starting
        if self.tree.num_nodes() >= MAX_TREE_SIZE {
            return false;
        }

        // Phase 1: Selection - traverse tree to find a node to expand
        let selected_index = self.selection(root_index);

        // Check if selected node is terminal (no expansion needed)
        {
            let node = self.tree.get_node(selected_index);
            if node.is_terminal {
                // Terminal nodes just get their reward backpropagated
                let reward = self.environment.reward(&node.state);
                self.backpropagation(selected_index, reward);
                return true;
            }
        }

        // Phase 2: Expansion - add a new child node
        let new_index = self.expansion(selected_index);

        // Phase 3: Simulation - random rollout from new node
        let reward = self.simulation(new_index);

        // Phase 4: Backpropagation - update statistics up the tree
        self.backpropagation(new_index, reward);

        return true;
    }
}

/// Generates a visual representation of the policy learned by MCTS.
///
/// This function runs MCTS from every cell in the grid and generates
/// a string showing the best action (as an arrow) for each cell.
///
/// # Symbols
///
/// - ` ^ ` : Best action is Up
/// - ` v ` : Best action is Down
/// - ` < ` : Best action is Left
/// - ` > ` : Best action is Right
/// - ` XX` : Blocked cell
/// - ` +1` : Positive terminal state
/// - ` -1` : Negative terminal state
/// - ` ? ` : No action available (dead end)
///
/// # Arguments
///
/// * `seed` - Base random seed for MCTS
/// * `num_iterations` - Number of MCTS iterations per cell
/// * `max_depth` - Maximum depth for simulation rollouts
///
/// # Returns
///
/// A string containing the visual representation of the policy
///
/// # Examples
///
/// ```
/// use rust_monte_carlo_tree_search::search;
///
/// let policy = search::generate_policy_string(12345, 100, 50);
/// println!("{}", policy);
/// ```
pub fn generate_policy_string(seed: u64, num_iterations: usize, max_depth: usize) -> String {
    let environment = grid_world::GridWorld::new();
    let mut output = String::new();

    // Iterate through all grid cells
    for r in 0..environment.rows {
        for c in 0..environment.cols {
            let state = grid_world::State { row: r, col: c };

            // Handle terminal and blocked states with fixed symbols
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
                // Use unique seed per cell for varied exploration
                let cell_seed = seed + (r * environment.cols + c) as u64;
                let mut mcts = Mcts::new(grid_world::GridWorld::new(), cell_seed, max_depth);

                mcts.run(&state, num_iterations);

                // Map the best action to a visual symbol
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

/// Prints a visual representation of the policy learned by MCTS.
///
/// This function generates and prints the policy string along with
/// a legend explaining the symbols.
///
/// # Arguments
///
/// * `seed` - Base random seed for MCTS
/// * `num_iterations` - Number of MCTS iterations per cell
/// * `max_depth` - Maximum depth for simulation rollouts
///
/// # Examples
///
/// ```
/// use rust_monte_carlo_tree_search::search;
///
/// // Visualize the policy
/// search::visualize_policy(12345, 100, 50);
/// ```
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
