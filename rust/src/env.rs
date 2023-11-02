use rand::Rng;

const GRID_SIZE: usize = 4;
const COIN_IDX: usize = 0;
const PIT_IDX: usize = 1;
const WALL_IDX: usize = 2;
const BOX_IDX: usize = 3;

type State = Vec<Vec<Vec<bool>>>;
type Position = (usize, usize);

/// Gym-like interface for the environment.
pub struct GridEnv {
    /// Stack of 2D arrays.
    pub grid: Vec<Vec<Vec<bool>>>,
    pub goal_pos: Position,
    pub agent_pos: Position,
}

impl GridEnv {
    pub fn new() -> Self {
        Self {
            grid: Vec::new(),
            goal_pos: (0, 0),
            agent_pos: (0, 0),
        }
    }

    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        let ref_grid: Vec<_> = (0..(GRID_SIZE * GRID_SIZE))
            .map(|_| rng.gen_range(0..4))
            .collect();
        let mut grid = vec![vec![vec![false; GRID_SIZE]; GRID_SIZE]; 3];
        for (i, &val) in ref_grid.iter().enumerate() {
            let y = i / GRID_SIZE;
            let x = i % GRID_SIZE;
            if val > 0 {
                grid[val - 1][y][x] = true;
            }
        }
        *self = Self {
            grid,
            goal_pos: (rng.gen_range(0..GRID_SIZE), rng.gen_range(0..GRID_SIZE)),
            agent_pos: (rng.gen_range(0..GRID_SIZE), rng.gen_range(0..GRID_SIZE)),
        };
    }

    pub fn step(&mut self, action: u32) -> (State, u32, f32, bool) {
        let mut dx = 0;
        let mut dy = 0;
        match action {
            0 => dx = -1,
            1 => dx = 1,
            2 => dy = -1,
            3 => dy = 1,
            _ => panic!(),
        }

        let mut x = (self.agent_pos.0 as i32 + dx).clamp(0, GRID_SIZE as i32) as usize;
        let mut y = (self.agent_pos.1 as i32 + dy).clamp(0, GRID_SIZE as i32) as usize;
        let mut reward = 0.;
        let mut done = false;

        // Moving into a wall.
        if self.grid[WALL_IDX][y][x] {
            (x, y) = self.agent_pos;
        }
        // Moving a box.
        else if self.grid[BOX_IDX][y][x] {
            let bx = x as i32 + dx;
            let by = y as i32 + dy;
            if is_border(bx, by)
                || self.grid[WALL_IDX][by as usize][bx as usize]
                || self.grid[BOX_IDX][by as usize][bx as usize]
            {
                (x, y) = self.agent_pos;
            } else {
                self.grid[BOX_IDX][by as usize][bx as usize] = true;
                self.grid[BOX_IDX][y][x] = false;
            }
        }
        // Moving into a coin.
        else if self.grid[COIN_IDX][y][x] {
            reward += 1.;
            self.grid[COIN_IDX][y][x] = false;
        }
        // Moving into the goal.
        else if (x, y) == self.goal_pos {
            reward += 10.;
            done = true;
        }
        // Moving into a pit.
        else if self.grid[PIT_IDX][y][x] {
            reward -= 10.;
            done = true;
        }

        self.agent_pos = (x, y);

        (self.get_obs(), action, reward, done)
    }

    fn get_obs(&self) -> State {
        let mut state = self.grid.clone();
        let mut goal_layer = vec![vec![false; GRID_SIZE]; GRID_SIZE];
        goal_layer[self.goal_pos.1][self.goal_pos.0] = true;
        let mut agent_layer = vec![vec![false; GRID_SIZE]; GRID_SIZE];
        agent_layer[self.goal_pos.1][self.goal_pos.0] = true;
        state.push(goal_layer);
        state.push(agent_layer);
        state
    }
}
fn is_border(x: i32, y: i32) -> bool {
    x < 0 || x >= GRID_SIZE as i32 || y < 0 || y >= GRID_SIZE as i32
}
