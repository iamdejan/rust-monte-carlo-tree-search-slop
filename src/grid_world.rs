// ============================================================================
// GridWorld Environment
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
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

pub struct GridWorld {
    pub rows: usize,
    pub cols: usize,
    pub blocked: State,
    pub positive_reward: State,
    pub negative_reward: State,
}

impl GridWorld {
    pub fn new() -> Self {
        GridWorld {
            rows: 3,
            cols: 4,
            blocked: State { row: 1, col: 1 },
            positive_reward: State { row: 0, col: 3 },
            negative_reward: State { row: 1, col: 3 },
        }
    }

    pub fn get_actions(&self, _state: &State) -> Vec<Action> {
        vec![Action::Up, Action::Down, Action::Left, Action::Right]
    }

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
            State {
                row: state.row,
                col: state.col,
            }
        } else {
            new_state
        }
    }

    pub fn reward(&self, state: &State) -> f64 {
        if *state == self.positive_reward {
            1.0
        } else if *state == self.negative_reward {
            -1.0
        } else {
            0.0
        }
    }

    pub fn is_terminal(&self, state: &State) -> bool {
        *state == self.positive_reward || *state == self.negative_reward
    }
}
