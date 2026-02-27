#![deny(unused_variables)]
#![deny(unused_imports)]

// Import the three modules that make up this project
mod grid_world; // The environment: GridWorld with states, actions, rewards
mod mcts; // Core data structures: MctsNode and MctsTree
mod search; // The MCTS algorithm implementation

fn main() {
    // Fixed seed for reproducible results across runs
    let seed = 1772163951;

    // Create the GridWorld environment
    let environment = grid_world::GridWorld::new();

    // MCTS configuration parameters
    let max_depth = 50; // Maximum depth for simulation rollouts
    let num_iterations = 1000; // Number of MCTS iterations per step

    // Starting position in the grid (bottom-left area)
    let start_state = grid_world::State { row: 1, col: 0 };

    // Print header and configuration information
    println!("=== Monte-Carlo Tree Search ===");
    println!("Grid World: 4 columns x 3 rows");
    println!(
        "Starting position: row={}, col={}",
        start_state.row, start_state.col
    );
    println!("Positive reward: row=0, col=3 (+1.0)");
    println!("Negative reward: row=1, col=3 (-1.0)");
    println!("Blocked cell: row=1, col=1");

    // Initial run just to print root statistics
    // Create a fresh MCTS instance for the initial analysis
    let mut initial_mcts = search::Mcts::new(grid_world::GridWorld::new(), seed, max_depth);
    initial_mcts.run(&start_state, num_iterations);

    // Display statistics for the root node (starting position)
    if let Some((visits, avg_reward)) = initial_mcts.get_statistics(&start_state) {
        println!(
            "\nRoot node - Visits: {}, Average Reward: {:.4}",
            visits, avg_reward
        );
    }

    // Simulate a full path from start to a terminal state
    println!("\n=== Simulated Path ===");
    let mut current_state = start_state.clone();
    let mut steps = 0;
    let max_steps = 20; // Safety limit to prevent infinite loops

    // Main simulation loop: keep taking actions until terminal or max steps
    while steps < max_steps {
        println!(
            "Step {}: state=({}, {})",
            steps, current_state.row, current_state.col
        );

        // Check if we've reached a terminal state (+1 or -1 reward)
        if environment.is_terminal(&current_state) {
            let reward = environment.reward(&current_state);
            println!("Reached terminal state with reward: {}", reward);
            break;
        }

        // Re-run MCTS from the current state to guarantee a fresh tree
        // Using a different seed per step ensures exploration diversity
        let mut step_mcts =
            search::Mcts::new(grid_world::GridWorld::new(), seed + steps as u64, max_depth);
        step_mcts.run(&current_state, num_iterations);

        // Get the best action according to MCTS (most visited child)
        if let Some(action) = step_mcts.get_best_action(&current_state) {
            // Apply the action to get the next state
            current_state = environment.transition(&current_state, &action);
            println!("  Action taken: {:?}", action);
        } else {
            println!("  No action available (dead end)");
            break;
        }

        steps += 1;
    }

    // Check if we hit the step limit without reaching terminal
    if steps >= max_steps {
        println!("Reached max steps without reaching terminal");
    }

    println!("\n=== MCTS Implementation Complete ===");

    // Display a visual map of the learned policy for all grid cells
    search::visualize_policy(seed, num_iterations, max_depth);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gridworld_transition() {
        let env = grid_world::GridWorld::new();

        // Test basic movement from middle-left position
        let state = grid_world::State { row: 1, col: 0 };

        // Move right - blocked cell at {1, 1}, so stays in place
        let next = env.transition(&state, &grid_world::Action::Right);
        assert_eq!(next, grid_world::State { row: 1, col: 0 });

        // Move down - valid move to bottom row
        let next = env.transition(&state, &grid_world::Action::Down);
        assert_eq!(next, grid_world::State { row: 2, col: 0 });

        // Move up - valid move to top row
        let next = env.transition(&state, &grid_world::Action::Up);
        assert_eq!(next, grid_world::State { row: 0, col: 0 });
    }

    #[test]
    fn test_gridworld_blocked() {
        let env = grid_world::GridWorld::new();

        // Starting at (1, 0), move right to (1, 1) which is blocked
        let state = grid_world::State { row: 1, col: 0 };
        let next = env.transition(&state, &grid_world::Action::Right);

        // Should stay at (1, 0) because (1, 1) is blocked
        assert_eq!(next, grid_world::State { row: 1, col: 0 });
    }

    #[test]
    fn test_gridworld_rewards() {
        let env = grid_world::GridWorld::new();

        // Positive reward at top-right corner
        let positive = grid_world::State { row: 0, col: 3 };
        assert_eq!(env.reward(&positive), 1.0);

        // Negative reward at middle-right
        let negative = grid_world::State { row: 1, col: 3 };
        assert_eq!(env.reward(&negative), -1.0);

        // No reward at starting position
        let neutral = grid_world::State { row: 0, col: 0 };
        assert_eq!(env.reward(&neutral), 0.0);
    }

    #[test]
    fn test_gridworld_terminal() {
        let env = grid_world::GridWorld::new();

        // Only the reward states are terminal
        assert!(env.is_terminal(&grid_world::State { row: 0, col: 3 }));
        assert!(env.is_terminal(&grid_world::State { row: 1, col: 3 }));
        assert!(!env.is_terminal(&grid_world::State { row: 0, col: 0 }));
    }

    #[test]
    fn test_mcts_node() {
        // Test that MctsNode is properly initialized
        let actions = vec![grid_world::Action::Up, grid_world::Action::Down];
        let node = mcts::MctsNode::new(
            grid_world::State { row: 0, col: 0 },
            None, // No parent (root node)
            actions.clone(),
            false, // Not terminal
        );

        // Verify initial statistics are zero
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.total_reward, 0.0);
        // Verify untried actions are populated
        assert_eq!(node.untried_actions.len(), 2);
    }

    #[test]
    fn test_mcts_basic() {
        let env = grid_world::GridWorld::new();
        let mut mcts = search::Mcts::new(env, 1772163951, 50);

        let start_state = grid_world::State { row: 1, col: 0 };

        // Run MCTS
        mcts.run(&start_state, 10);

        // Check that root was created
        let indices = mcts.tree.get_nodes_by_state(&start_state);
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_policy_visualization_structure() {
        // Run with very few iterations just to test the string generation quickly
        let grid_string = search::generate_policy_string(12345, 10, 10);

        // Use .lines() instead of .trim_end().split('\n')
        let lines: Vec<&str> = grid_string.lines().collect();

        // 1. Check dimensions - grid should have 3 rows
        assert_eq!(lines.len(), 3, "Grid should have exactly 3 rows");

        // Each row should have 16 characters (4 cells * 4 chars each)
        for (i, line) in lines.iter().enumerate() {
            assert_eq!(
                line.len(),
                16,
                "Row {} should have exactly 16 characters (4 cells * 4 chars)",
                i
            );
        }

        // 2. Extract specific cells (each cell is 4 characters wide)
        let get_cell = |row: usize, col: usize| -> &str {
            let start = col * 4;
            let end = start + 4;
            &lines[row][start..end]
        };

        // 3. Check fixed elements based on GridWorld configuration
        assert_eq!(
            get_cell(0, 3),
            " +1 ",
            "Positive reward missing or in wrong place"
        );
        assert_eq!(
            get_cell(1, 3),
            " -1 ",
            "Negative reward missing or in wrong place"
        );
        assert_eq!(
            get_cell(1, 1),
            " XX ",
            "Blocked cell missing or in wrong place"
        );

        // 4. Check that a standard cell contains a valid action symbol
        let start_cell = get_cell(1, 0);
        let valid_symbols = ["  ^ ", "  v ", "  < ", "  > ", "  ? "];
        assert!(
            valid_symbols.contains(&start_cell),
            "Standard cell contains invalid symbol: '{}'",
            start_cell
        );
    }
}
