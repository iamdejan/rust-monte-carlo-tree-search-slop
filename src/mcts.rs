#![deny(unused_variables)]
#![deny(unused_imports)]

//! Core MCTS data structures module.
//!
//! This module provides the fundamental data structures for the Monte-Carlo Tree
//! Search algorithm: the tree node (`MctsNode`) and the tree container (`MctsTree`).

use std::collections::HashMap;

use crate::grid_world;

// ============================================================================
// MCTS Tree Node
// ============================================================================

/// Represents a single node in the MCTS tree.
///
/// Each node corresponds to a state in the environment and maintains statistics
/// used by the search algorithm to guide exploration and exploitation.
///
/// # Statistics
///
/// - `visit_count`: Number of times this node has been visited during simulations
/// - `total_reward`: Cumulative reward accumulated through this node
/// - `average_reward`: `total_reward / visit_count` (computed on demand)
///
/// # Examples
///
/// ```
/// use rust_monte_carlo_tree_search::{grid_world, mcts};
///
/// let state = grid_world::State { row: 0, col: 0 };
/// let actions = vec![grid_world::Action::Up, grid_world::Action::Right];
///
/// let node = mcts::MctsNode::new(state, None, actions, false);
///
/// assert_eq!(node.visit_count, 0);
/// assert_eq!(node.total_reward, 0.0);
/// ```
#[derive(Clone)]
pub struct MctsNode {
    /// The environment state this node represents.
    pub state: grid_world::State,
    /// Index of the parent node in the tree (`None` for the root node).
    pub parent: Option<usize>,
    /// List of child nodes as pairs of (action, child_index).
    /// Each entry represents a transition to a child state via a specific action.
    pub children: Vec<(grid_world::Action, usize)>,
    /// Number of times this node has been visited during MCTS iterations.
    pub visit_count: u32,
    /// Cumulative reward accumulated through all simulations that passed through this node.
    pub total_reward: f64,
    /// Actions that have not yet been tried from this state.
    /// When empty, the node is considered "fully expanded".
    pub untried_actions: Vec<grid_world::Action>,
    /// Whether this node represents a terminal state (episode ends here).
    pub is_terminal: bool,
}

impl MctsNode {
    /// Creates a new MCTS node.
    ///
    /// # Arguments
    ///
    /// * `state` - The environment state this node represents
    /// * `parent` - Optional index of the parent node in the tree
    /// * `actions` - List of available actions from this state (becomes `untried_actions`)
    /// * `is_terminal` - Whether this is a terminal state
    ///
    /// # Initial State
    ///
    /// The new node starts with:
    /// - `visit_count = 0`
    /// - `total_reward = 0.0`
    /// - `children = []` (empty)
    /// - `untried_actions = actions` (all actions are initially untried)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, mcts};
    ///
    /// let state = grid_world::State { row: 1, col: 0 };
    /// let parent = Some(0); // Parent at index 0
    /// let actions = vec![grid_world::Action::Up, grid_world::Action::Down];
    ///
    /// let node = mcts::MctsNode::new(state, parent, actions, false);
    ///
    /// assert_eq!(node.visit_count, 0);
    /// assert!(!node.is_fully_expanded());
    /// ```
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

    /// Returns the average reward per visit for this node.
    ///
    /// This value represents the exploitation component in the UCB1 formula.
    ///
    /// # Returns
    ///
    /// - `0.0` if the node has never been visited (`visit_count == 0`)
    /// - `total_reward / visit_count` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, mcts};
    ///
    /// let state = grid_world::State { row: 0, col: 0 };
    /// let mut node = mcts::MctsNode::new(state, None, vec![], false);
    ///
    /// // Initially, average reward is 0
    /// assert_eq!(node.average_reward(), 0.0);
    ///
    /// // After accumulating reward
    /// node.visit_count = 10;
    /// node.total_reward = 5.0;
    /// assert_eq!(node.average_reward(), 0.5);
    /// ```
    pub fn average_reward(&self) -> f64 {
        if self.visit_count == 0 {
            return 0.0;
        } else {
            return self.total_reward / self.visit_count as f64;
        }
    }

    /// Checks if all possible actions from this state have been tried.
    ///
    /// A node is "fully expanded" when `untried_actions` is empty.
    /// During the selection phase, fully expanded nodes are traversed using UCB1,
    /// while non-fully-expanded nodes are selected for expansion.
    ///
    /// # Returns
    ///
    /// `true` if all actions have been tried, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, mcts};
    ///
    /// let state = grid_world::State { row: 0, col: 0 };
    /// let actions = vec![grid_world::Action::Up];
    ///
    /// let mut node = mcts::MctsNode::new(state, None, actions, false);
    ///
    /// // Initially not fully expanded
    /// assert!(!node.is_fully_expanded());
    ///
    /// // After exhausting all actions
    /// node.untried_actions.clear();
    /// assert!(node.is_fully_expanded());
    /// ```
    pub fn is_fully_expanded(&self) -> bool {
        return self.untried_actions.is_empty();
    }

    /// Finds the child node index corresponding to a specific action.
    ///
    /// This method is used during the selection phase to traverse the tree
    /// after choosing the best action via UCB1.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to look up
    ///
    /// # Returns
    ///
    /// - `Some(index)` if a child exists for the given action
    /// - `None` if no child exists for that action
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, mcts};
    ///
    /// let state = grid_world::State { row: 0, col: 0 };
    /// let mut node = mcts::MctsNode::new(state, None, vec![], false);
    ///
    /// // Add a child
    /// node.children.push((grid_world::Action::Right, 5));
    ///
    /// // Look up the child
    /// assert_eq!(node.get_child_by_action(grid_world::Action::Right), Some(5));
    /// assert_eq!(node.get_child_by_action(grid_world::Action::Up), None);
    /// ```
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

/// Manages a collection of MCTS nodes.
///
/// `MctsTree` stores all nodes in a vector and maintains a mapping from states
/// to node indices for efficient lookup. This allows the same state to appear
/// multiple times in the tree (reached via different paths).
///
/// # Examples
///
/// ```
/// use rust_monte_carlo_tree_search::{grid_world, mcts};
///
/// let mut tree = mcts::MctsTree::new();
/// let state = grid_world::State { row: 0, col: 0 };
/// let node = mcts::MctsNode::new(state, None, vec![], false);
///
/// let index = tree.add_node(node);
/// assert_eq!(tree.num_nodes(), 1);
/// ```
pub struct MctsTree {
    /// Storage for all nodes in the tree, indexed by their position.
    nodes: Vec<MctsNode>,
    /// Maps states to all node indices representing that state.
    /// Multiple indices are possible since the same state can be reached
    /// through different action sequences.
    state_to_indices: HashMap<grid_world::State, Vec<usize>>,
}

impl MctsTree {
    /// Creates an empty MCTS tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::mcts::MctsTree;
    ///
    /// let tree = MctsTree::new();
    /// assert_eq!(tree.num_nodes(), 0);
    /// ```
    pub fn new() -> Self {
        return MctsTree {
            nodes: Vec::new(),
            state_to_indices: HashMap::new(),
        };
    }

    /// Adds a new node to the tree.
    ///
    /// The node is appended to the internal vector, and its state is recorded
    /// in the state-to-indices map for lookup.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to add
    ///
    /// # Returns
    ///
    /// The index (position in the vector) of the newly added node
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, mcts};
    ///
    /// let mut tree = mcts::MctsTree::new();
    /// let state = grid_world::State { row: 0, col: 0 };
    /// let node = mcts::MctsNode::new(state, None, vec![], false);
    ///
    /// let index = tree.add_node(node);
    /// assert_eq!(index, 0);
    /// ```
    pub fn add_node(&mut self, node: MctsNode) -> usize {
        let index = self.nodes.len();
        let state = node.state.clone();
        self.nodes.push(node);
        self.state_to_indices.entry(state).or_default().push(index);
        return index;
    }

    /// Returns an immutable reference to the node at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the node
    ///
    /// # Returns
    ///
    /// A reference to the node
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds
    pub fn get_node(&self, index: usize) -> &MctsNode {
        return &self.nodes[index];
    }

    /// Returns a mutable reference to the node at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the node
    ///
    /// # Returns
    ///
    /// A mutable reference to the node
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds
    pub fn get_node_mut(&mut self, index: usize) -> &mut MctsNode {
        return &mut self.nodes[index];
    }

    /// Returns all node indices that represent the given state.
    ///
    /// Since the same state can be reached through different paths, multiple
    /// nodes may exist for the same state.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to look up
    ///
    /// # Returns
    ///
    /// A vector of node indices, or an empty vector if no nodes exist for the state
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::{grid_world, mcts};
    ///
    /// let mut tree = mcts::MctsTree::new();
    /// let state = grid_world::State { row: 0, col: 0 };
    /// let node = mcts::MctsNode::new(state.clone(), None, vec![], false);
    ///
    /// tree.add_node(node);
    ///
    /// let indices = tree.get_nodes_by_state(&state);
    /// assert_eq!(indices.len(), 1);
    /// ```
    pub fn get_nodes_by_state(&self, state: &grid_world::State) -> Vec<usize> {
        return self
            .state_to_indices
            .get(state)
            .cloned()
            .unwrap_or_default();
    }

    /// Returns the total number of nodes in the tree.
    ///
    /// This is used to enforce the maximum tree size limit.
    ///
    /// # Returns
    ///
    /// The number of nodes currently in the tree
    pub fn num_nodes(&self) -> usize {
        return self.nodes.len();
    }
}
