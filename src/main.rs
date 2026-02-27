use rand::SeedableRng;
use rand::prelude::*;
use std::collections::HashMap;

// ============================================================================
// GridWorld Environment
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    row: usize,
    col: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    fn delta(&self) -> (i32, i32) {
        match self {
            Action::Up => (-1, 0),
            Action::Down => (1, 0),
            Action::Left => (0, -1),
            Action::Right => (0, 1),
        }
    }
}

struct GridWorld {
    rows: usize,
    cols: usize,
    blocked: State,
    positive_reward: State,
    negative_reward: State,
}

impl GridWorld {
    fn new() -> Self {
        GridWorld {
            rows: 3,
            cols: 4,
            blocked: State { row: 1, col: 1 },
            positive_reward: State { row: 0, col: 3 },
            negative_reward: State { row: 1, col: 3 },
        }
    }

    fn get_actions(&self, _state: &State) -> Vec<Action> {
        vec![Action::Up, Action::Down, Action::Left, Action::Right]
    }

    fn transition(&self, state: &State, action: &Action) -> State {
        let (dr, dc) = action.delta();
        let new_row = (state.row as i32 + dr).clamp(0, self.rows as i32 - 1) as usize;
        let new_col = (state.col as i32 + dc).clamp(0, self.cols as i32 - 1) as usize;

        let new_state = State {
            row: new_row,
            col: new_col,
        };

        // If the new state is blocked, go back to the original state
        if new_state == self.blocked {
            State {
                row: state.row,
                col: state.col,
            }
        } else {
            new_state
        }
    }

    fn reward(&self, state: &State) -> f64 {
        if *state == self.positive_reward {
            1.0
        } else if *state == self.negative_reward {
            -1.0
        } else {
            0.0
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        *state == self.positive_reward || *state == self.negative_reward
    }
}

// ============================================================================
// Expectimax Tree Node
// ============================================================================

#[derive(Clone)]
struct ExpectimaxNode {
    state: State,
    parent: Option<usize>,
    // Store children as (action, state) -> child_index to handle duplicate states from different actions
    children: Vec<(Action, usize)>,
    visit_count: u32,
    total_reward: f64,
    untried_actions: Vec<Action>,
    is_terminal: bool,
}

impl ExpectimaxNode {
    fn new(state: State, parent: Option<usize>, actions: Vec<Action>, is_terminal: bool) -> Self {
        ExpectimaxNode {
            state,
            parent,
            children: Vec::new(),
            visit_count: 0,
            total_reward: 0.0,
            untried_actions: actions,
            is_terminal,
        }
    }

    fn average_reward(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_reward / self.visit_count as f64
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.untried_actions.is_empty()
    }

    fn get_child_by_action(&self, action: Action) -> Option<usize> {
        for (a, idx) in &self.children {
            if *a == action {
                return Some(*idx);
            }
        }
        None
    }
}

// ============================================================================
// Expectimax Tree
// ============================================================================

struct ExpectimaxTree {
    nodes: Vec<ExpectimaxNode>,
    // Map state to all node indices (there can be multiple nodes for same state from different paths)
    state_to_indices: HashMap<State, Vec<usize>>,
}

impl ExpectimaxTree {
    fn new() -> Self {
        ExpectimaxTree {
            nodes: Vec::new(),
            state_to_indices: HashMap::new(),
        }
    }

    fn add_node(&mut self, node: ExpectimaxNode) -> usize {
        let index = self.nodes.len();
        let state = node.state.clone();
        self.nodes.push(node);
        self.state_to_indices.entry(state).or_default().push(index);
        index
    }

    fn get_node(&self, index: usize) -> &ExpectimaxNode {
        &self.nodes[index]
    }

    fn get_node_mut(&mut self, index: usize) -> &mut ExpectimaxNode {
        &mut self.nodes[index]
    }

    fn get_nodes_by_state(&self, state: &State) -> Vec<usize> {
        self.state_to_indices
            .get(state)
            .cloned()
            .unwrap_or_default()
    }

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

// ============================================================================
// Monte-Carlo Tree Search
// ============================================================================

const EXPLORATION_CONSTANT: f64 = 1.414; // sqrt(2)
const MAX_TREE_SIZE: usize = 10000;

struct Mcts {
    tree: ExpectimaxTree,
    environment: GridWorld,
    max_depth: usize,
    rng: StdRng,
}

impl Mcts {
    fn new(environment: GridWorld, seed: u64, max_depth: usize) -> Self {
        Mcts {
            tree: ExpectimaxTree::new(),
            environment,
            max_depth,
            rng: StdRng::seed_from_u64(seed),
        }
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
            let mut best_action = Action::Up;
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

    fn ucb1(&self, node: &ExpectimaxNode, parent_visits: u32) -> f64 {
        if node.visit_count == 0 {
            f64::INFINITY
        } else {
            let exploitation = node.average_reward();
            let exploration = EXPLORATION_CONSTANT
                * ((parent_visits as f64).ln() / node.visit_count as f64).sqrt();
            exploitation + exploration
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

        // Check if state already exists in tree - find one that doesn't have this action from current parent
        // For simplicity, always create a new node (MCTS typically expands once per visit anyway)

        // Get actions for new state
        let new_actions = if self.environment.is_terminal(&new_state) {
            Vec::new()
        } else {
            self.environment.get_actions(&new_state)
        };

        let is_terminal = self.environment.is_terminal(&new_state);

        // Create new node
        let new_node = ExpectimaxNode::new(
            new_state.clone(),
            Some(node_index),
            new_actions,
            is_terminal,
        );

        let new_index = self.tree.add_node(new_node);

        // Add to parent's children
        let node_mut = self.tree.get_node_mut(node_index);
        node_mut.children.push((action, new_index));

        new_index
    }

    fn simulation(&self, node_index: usize) -> f64 {
        let node = self.tree.get_node(node_index);
        let mut current_state = node.state.clone();

        // If starting from a terminal state, return its reward
        if self.environment.is_terminal(&current_state) {
            return self.environment.reward(&current_state);
        }

        // Create a new RNG for simulation with the fixed seed
        let mut sim_rng = StdRng::seed_from_u64(1772163951);
        let mut depth = 0;

        // Simulate until terminal or max depth
        while depth < self.max_depth && !self.environment.is_terminal(&current_state) {
            let actions = self.environment.get_actions(&current_state);

            // If no valid actions, return current reward
            if actions.is_empty() {
                return self.environment.reward(&current_state);
            }

            let idx = sim_rng.gen_range(0..actions.len());
            let action = actions[idx];
            current_state = self.environment.transition(&current_state, &action);
            depth += 1;
        }

        self.environment.reward(&current_state)
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
        // Check tree size
        if self.tree.num_nodes() >= MAX_TREE_SIZE {
            return false;
        }

        // Selection: traverse tree until we find a node to expand
        let selected_index = self.selection(root_index);

        // Check if terminal
        {
            let node = self.tree.get_node(selected_index);
            if node.is_terminal {
                let reward = self.environment.reward(&node.state);
                self.backpropagation(selected_index, reward);
                return true;
            }
        }

        // Expansion: add a new child node
        let new_index = self.expansion(selected_index);

        // Simulation: play out the game randomly
        let reward = self.simulation(new_index);

        // Backpropagation: update all nodes on the path
        self.backpropagation(new_index, reward);

        true
    }

    fn run(&mut self, root_state: &State, num_iterations: usize) {
        // Initialize root node if not exists
        let root_index = if let Some(idx) = self.tree.get_nodes_by_state(root_state).first() {
            *idx
        } else {
            let actions = self.environment.get_actions(root_state);
            let is_terminal = self.environment.is_terminal(root_state);
            let root_node = ExpectimaxNode::new(root_state.clone(), None, actions, is_terminal);
            self.tree.add_node(root_node)
        };

        // Run iterations
        for i in 0..num_iterations {
            if !self.run_iteration(root_index) {
                eprintln!("Warning: Tree size limit reached at iteration {}", i);
                break;
            }
            // Print progress every 100 iterations
            if i > 0 && i % 100 == 0 {
                eprintln!(
                    "Iteration {} complete, tree size: {}",
                    i,
                    self.tree.num_nodes()
                );
            }
        }
    }

    fn get_best_action(&mut self, root_state: &State) -> Option<Action> {
        // Find root node
        let root_indices = self.tree.get_nodes_by_state(root_state);
        if root_indices.is_empty() {
            return None;
        }
        let root_index = root_indices[0];
        let root = self.tree.get_node(root_index);

        if root.children.is_empty() {
            return None;
        }

        // Select child with highest visit count
        let mut best_action = Action::Up;
        let mut best_visits = 0u32;

        for (action, child_index) in &root.children {
            let child = self.tree.get_node(*child_index);
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_action = *action;
            }
        }

        Some(best_action)
    }

    fn get_statistics(&self, state: &State) -> Option<(u32, f64)> {
        let indices = self.tree.get_nodes_by_state(state);
        if indices.is_empty() {
            return None;
        }
        let node = self.tree.get_node(indices[0]);
        Some((node.visit_count, node.average_reward()))
    }
}

// ============================================================================
// Main Function
// ============================================================================

fn main() {
    // Create the environment
    let seed = 1772163951;
    let environment = GridWorld::new();

    // Create MCTS
    let max_depth = 50;
    let num_iterations = 1000;
    let mut mcts = Mcts::new(environment, seed, max_depth);

    // Starting state
    let start_state = State { row: 1, col: 0 };

    println!("=== Monte-Carlo Tree Search ===");
    println!("Grid World: 4 columns x 3 rows");
    println!(
        "Starting position: row={}, col={}",
        start_state.row, start_state.col
    );
    println!("Positive reward: row=0, col=3 (+1.0)");
    println!("Negative reward: row=1, col=3 (-1.0)");
    println!("Blocked cell: row=1, col=1");
    println!("\nRunning MCTS with {} iterations...", num_iterations);

    // Run MCTS
    mcts.run(&start_state, num_iterations);

    // Get statistics for root
    if let Some((visits, avg_reward)) = mcts.get_statistics(&start_state) {
        println!(
            "Root node - Visits: {}, Average Reward: {:.4}",
            visits, avg_reward
        );
    }

    // Get best action
    if let Some(best_action) = mcts.get_best_action(&start_state) {
        println!("\nBest action from start state: {:?}", best_action);

        // Show child statistics
        let root_indices = mcts.tree.get_nodes_by_state(&start_state);
        if !root_indices.is_empty() {
            let root = mcts.tree.get_node(root_indices[0]);

            println!("\nAll child statistics:");
            for (action, child_idx) in &root.children {
                let child = mcts.tree.get_node(*child_idx);
                println!(
                    "  Action {:?}: state=({}, {}), visits={}, avg_reward={:.4}",
                    action,
                    child.state.row,
                    child.state.col,
                    child.visit_count,
                    child.average_reward()
                );
            }
        }
    }

    // Simulate a full path from start
    println!("\n=== Simulated Path ===");
    let mut current_state = start_state;
    let mut steps = 0;
    let max_steps = 20;

    while steps < max_steps {
        println!(
            "Step {}: state=({}, {})",
            steps, current_state.row, current_state.col
        );

        if mcts.environment.is_terminal(&current_state) {
            let reward = mcts.environment.reward(&current_state);
            println!("Reached terminal state with reward: {}", reward);
            break;
        }

        if let Some(action) = mcts.get_best_action(&current_state) {
            current_state = mcts.environment.transition(&current_state, &action);
            println!("  Action taken: {:?}", action);
        } else {
            println!("  No action available (dead end)");
            break;
        }

        steps += 1;
    }

    if steps >= max_steps {
        println!("Reached max steps without reaching terminal");
    }

    println!("\n=== MCTS Implementation Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gridworld_transition() {
        let env = GridWorld::new();

        // Test basic movement
        let state = State { row: 1, col: 0 };

        // Move right - blocked cell at {1, 1}, so stays in place
        let next = env.transition(&state, &Action::Right);
        assert_eq!(next, State { row: 1, col: 0 });

        // Move down
        let next = env.transition(&state, &Action::Down);
        assert_eq!(next, State { row: 2, col: 0 });

        // Move up
        let next = env.transition(&state, &Action::Up);
        assert_eq!(next, State { row: 0, col: 0 });
    }

    #[test]
    fn test_gridworld_blocked() {
        let env = GridWorld::new();

        // Starting at (1, 0), move right to (1, 1) which is blocked
        let state = State { row: 1, col: 0 };
        let next = env.transition(&state, &Action::Right);

        // Should stay at (1, 0) because (1, 1) is blocked
        assert_eq!(next, State { row: 1, col: 0 });
    }

    #[test]
    fn test_gridworld_rewards() {
        let env = GridWorld::new();

        // Positive reward
        let positive = State { row: 0, col: 3 };
        assert_eq!(env.reward(&positive), 1.0);

        // Negative reward
        let negative = State { row: 1, col: 3 };
        assert_eq!(env.reward(&negative), -1.0);

        // No reward
        let neutral = State { row: 0, col: 0 };
        assert_eq!(env.reward(&neutral), 0.0);
    }

    #[test]
    fn test_gridworld_terminal() {
        let env = GridWorld::new();

        assert!(env.is_terminal(&State { row: 0, col: 3 }));
        assert!(env.is_terminal(&State { row: 1, col: 3 }));
        assert!(!env.is_terminal(&State { row: 0, col: 0 }));
    }

    #[test]
    fn test_expectimax_node() {
        let actions = vec![Action::Up, Action::Down];
        let node = ExpectimaxNode::new(State { row: 0, col: 0 }, None, actions.clone(), false);

        assert_eq!(node.visit_count, 0);
        assert_eq!(node.total_reward, 0.0);
        assert_eq!(node.untried_actions.len(), 2);
    }

    #[test]
    fn test_mcts_basic() {
        let env = GridWorld::new();
        let mut mcts = Mcts::new(env, 1772163951, 50);

        let start_state = State { row: 1, col: 0 };

        // Run MCTS
        mcts.run(&start_state, 10);

        // Check that root was created
        let indices = mcts.tree.get_nodes_by_state(&start_state);
        assert!(!indices.is_empty());
    }
}
