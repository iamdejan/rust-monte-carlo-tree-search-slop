#![deny(unused_variables)]
#![deny(unused_imports)]

//! GridWorld environment module.
//!
//! This module defines a simple grid-based environment where an agent can move
//! in four directions. The goal is to reach a positive reward while avoiding
//! obstacles and negative rewards.

// ============================================================================
// GridWorld Environment
// ============================================================================

/// Represents a position in the GridWorld grid.
///
/// The state is defined by row and column indices (0-indexed).
/// Row 0 is the top row, and column 0 is the leftmost column.
///
/// # Examples
///
/// ```
/// use rust_monte_carlo_tree_search::grid_world::State;
///
/// let start = State { row: 1, col: 0 };
/// assert_eq!(start.row, 1);
/// assert_eq!(start.col, 0);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    /// The row index (0-indexed, from top to bottom).
    pub row: usize,
    /// The column index (0-indexed, from left to right).
    pub col: usize,
}

/// Represents the possible movement actions in the GridWorld.
///
/// The agent can move in four cardinal directions: Up, Down, Left, and Right.
/// Movement is constrained by grid boundaries and blocked cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Move one cell up (decreases row index).
    Up,
    /// Move one cell down (increases row index).
    Down,
    /// Move one cell left (decreases column index).
    Left,
    /// Move one cell right (increases column index).
    Right,
}

impl Action {
    /// Returns the row and column delta for this action.
    ///
    /// The delta represents the change in position when the action is applied.
    ///
    /// # Returns
    ///
    /// A tuple `(row_delta, col_delta)` where:
    /// - `row_delta` is the change in row (-1 for up, +1 for down, 0 otherwise)
    /// - `col_delta` is the change in column (-1 for left, +1 for right, 0 otherwise)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::grid_world::Action;
    ///
    /// assert_eq!(Action::Up.delta(), (-1, 0));
    /// assert_eq!(Action::Right.delta(), (0, 1));
    /// ```
    fn delta(&self) -> (i32, i32) {
        match self {
            Action::Up => return (-1, 0),
            Action::Down => return (1, 0),
            Action::Left => return (0, -1),
            Action::Right => return (0, 1),
        }
    }
}

/// The GridWorld environment definition.
///
/// This struct defines a 3x4 grid with:
/// - A blocked cell at position (1, 1)
/// - A positive reward (+1.0) at position (0, 3)
/// - A negative reward (-1.0) at position (1, 3)
///
/// The agent can move in four directions, but movement is constrained by:
/// - Grid boundaries (cannot move outside the 3x4 grid)
/// - The blocked cell (attempting to enter bounces back to original position)
///
/// # Examples
///
/// ```
/// use rust_monte_carlo_tree_search::grid_world::{GridWorld, State, Action};
///
/// let env = GridWorld::new();
/// let start = State { row: 1, col: 0 };
///
/// // Move up
/// let next = env.transition(&start, &Action::Up);
/// assert_eq!(next, State { row: 0, col: 0 });
/// ```
pub struct GridWorld {
    /// Number of rows in the grid (3).
    pub rows: usize,
    /// Number of columns in the grid (4).
    pub cols: usize,
    /// The blocked cell position (1, 1) that cannot be entered.
    pub blocked: State,
    /// The goal state (0, 3) with reward +1.0.
    pub positive_reward: State,
    /// The bad state (1, 3) with reward -1.0.
    pub negative_reward: State,
}

impl GridWorld {
    /// Creates a new GridWorld with default configuration.
    ///
    /// The default grid is 3 rows × 4 columns with:
    /// - Blocked cell at (1, 1)
    /// - Positive reward at (0, 3)
    /// - Negative reward at (1, 3)
    ///
    /// # Returns
    ///
    /// A new `GridWorld` instance with the default configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::grid_world::GridWorld;
    ///
    /// let env = GridWorld::new();
    /// assert_eq!(env.rows, 3);
    /// assert_eq!(env.cols, 4);
    /// ```
    pub fn new() -> Self {
        return GridWorld {
            rows: 3,
            cols: 4,
            blocked: State { row: 1, col: 1 },
            positive_reward: State { row: 0, col: 3 },
            negative_reward: State { row: 1, col: 3 },
        };
    }

    /// Returns all possible actions from any given state.
    ///
    /// In this implementation, all four actions are always available regardless
    /// of the current state. Boundary checking is handled by the `transition`
    /// method.
    ///
    /// # Arguments
    ///
    /// * `_state` - The current state (unused, as all actions are always available)
    ///
    /// # Returns
    ///
    /// A vector containing all four actions: `[Up, Down, Left, Right]`
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::grid_world::{GridWorld, State, Action};
    ///
    /// let env = GridWorld::new();
    /// let state = State { row: 0, col: 0 };
    /// let actions = env.get_actions(&state);
    ///
    /// assert_eq!(actions.len(), 4);
    /// assert!(actions.contains(&Action::Up));
    /// assert!(actions.contains(&Action::Right));
    /// ```
    pub fn get_actions(&self, _state: &State) -> Vec<Action> {
        return vec![Action::Up, Action::Down, Action::Left, Action::Right];
    }

    /// Computes the next state after taking an action from the current state.
    ///
    /// The transition follows these rules:
    /// 1. Calculate the new position by applying the action's delta
    /// 2. Clamp the new position to grid boundaries (cannot go outside)
    /// 3. If the new position is the blocked cell, return to the original state
    /// 4. Otherwise, return the new position
    ///
    /// # Arguments
    ///
    /// * `state` - The current state
    /// * `action` - The action to take
    ///
    /// # Returns
    ///
    /// The resulting state after applying the action
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::grid_world::{GridWorld, State, Action};
    ///
    /// let env = GridWorld::new();
    /// let state = State { row: 1, col: 0 };
    ///
    /// // Move up - valid move
    /// let next = env.transition(&state, &Action::Up);
    /// assert_eq!(next, State { row: 0, col: 0 });
    ///
    /// // Move right - blocked by obstacle at (1, 1), stays in place
    /// let next = env.transition(&state, &Action::Right);
    /// assert_eq!(next, State { row: 1, col: 0 });
    /// ```
    pub fn transition(&self, state: &State, action: &Action) -> State {
        let (dr, dc) = action.delta();
        let new_row = (state.row as i32 + dr).clamp(0, self.rows as i32 - 1) as usize;
        let new_col = (state.col as i32 + dc).clamp(0, self.cols as i32 - 1) as usize;

        let new_state = State {
            row: new_row,
            col: new_col,
        };

        // If the new state is blocked, go back to the original state
        if new_state == self.blocked {
            return State {
                row: state.row,
                col: state.col,
            };
        } else {
            return new_state;
        }
    }

    /// Returns the immediate reward for being in a state.
    ///
    /// Rewards are assigned as follows:
    /// - `+1.0` for the positive reward state (0, 3)
    /// - `-1.0` for the negative reward state (1, 3)
    /// - `0.0` for all other states
    ///
    /// # Arguments
    ///
    /// * `state` - The state to evaluate
    ///
    /// # Returns
    ///
    /// The reward value for the given state
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::grid_world::{GridWorld, State};
    ///
    /// let env = GridWorld::new();
    ///
    /// assert_eq!(env.reward(&State { row: 0, col: 3 }), 1.0);
    /// assert_eq!(env.reward(&State { row: 1, col: 3 }), -1.0);
    /// assert_eq!(env.reward(&State { row: 0, col: 0 }), 0.0);
    /// ```
    pub fn reward(&self, state: &State) -> f64 {
        if *state == self.positive_reward {
            return 1.0;
        } else if *state == self.negative_reward {
            return -1.0;
        } else {
            return 0.0;
        }
    }

    /// Checks if a state is terminal.
    ///
    /// Terminal states are those with non-zero rewards (positive or negative).
    /// Once the agent reaches a terminal state, the episode ends.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to check
    ///
    /// # Returns
    ///
    /// `true` if the state is terminal, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_monte_carlo_tree_search::grid_world::{GridWorld, State};
    ///
    /// let env = GridWorld::new();
    ///
    /// // Terminal states (reward states)
    /// assert!(env.is_terminal(&State { row: 0, col: 3 }));
    /// assert!(env.is_terminal(&State { row: 1, col: 3 }));
    ///
    /// // Non-terminal states
    /// assert!(!env.is_terminal(&State { row: 0, col: 0 }));
    /// assert!(!env.is_terminal(&State { row: 2, col: 2 }));
    /// ```
    pub fn is_terminal(&self, state: &State) -> bool {
        return *state == self.positive_reward || *state == self.negative_reward;
    }
}
