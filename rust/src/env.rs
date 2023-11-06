use std::collections::VecDeque;

use rand::Rng;

pub const GRID_SIZE: usize = 6;
const COIN_IDX: usize = 0;
const PIT_IDX: usize = 1;
const WALL_IDX: usize = 2;
const BOX_IDX: usize = 3;
pub const NUM_CHANNELS: usize = BOX_IDX + 1 + 2;
pub const MAX_TIME: u32 = 16;
pub const MAX_SAME: usize = 4; // How long an agent is allowed to stay in the same spot.

pub type State = Vec<Vec<Vec<bool>>>;
type Position = (usize, usize);

/// Gym-like interface for the environment.
pub struct GridEnv {
    /// Stack of 2D arrays.
    pub grid: Vec<Vec<Vec<bool>>>,
    pub goal_pos: Position,
    pub agent_pos: Position,
    pub timer: u32,
    pub pos_buf: VecDeque<Position>,
}

/// Returns the position of an empty cell.
fn get_empty(ref_grid: &[usize], taken: &[Position]) -> Position {
    let mut rng = rand::thread_rng();
    let mut new_pos = (rng.gen_range(0..GRID_SIZE), rng.gen_range(0..GRID_SIZE));
    while ref_grid[new_pos.1 * GRID_SIZE + new_pos.0] != 0 || taken.contains(&new_pos) {
        new_pos = (rng.gen_range(0..GRID_SIZE), rng.gen_range(0..GRID_SIZE));
    }
    new_pos
}

impl GridEnv {
    pub fn new() -> Self {
        Self {
            grid: Vec::new(),
            goal_pos: (0, 0),
            agent_pos: (0, 0),
            timer: 0,
            pos_buf: VecDeque::new(),
        }
    }

    pub fn reset(&mut self) -> State {
        let mut rng = rand::thread_rng();
        let ref_grid = loop {
            let ref_grid = vec![
                3, 3, 3, 3, 3, 3,
                3, 0, 0, 3, 0, 3,
                3, 1, 1, 1, 1, 3,
                3, 1, 2, 3, 0, 3,
                3, 0, 0, 3, 1, 3,
                3, 3, 3, 3, 3, 3,
            ];
            // let mut ref_grid: Vec<_> = (0..(GRID_SIZE * GRID_SIZE))
            //     .map(|_| rng.gen_range(0..(BOX_IDX + 2)))
            //     .collect();
            // for y in 0..GRID_SIZE {
            //     for x in 0..GRID_SIZE {
            //         if (1..GRID_SIZE - 1).contains(&x) && (1..GRID_SIZE - 1).contains(&y) {
            //             continue;
            //         }
            //         ref_grid[y * GRID_SIZE + x] = WALL_IDX + 1;
            //     }
            // }
            // if ref_grid.iter().filter(|&&c| c == 0).count() > 2
            //     && ref_grid.iter().filter(|&&c| c == WALL_IDX + 1).count()
            //         <= (GRID_SIZE * 4 - 4 + 2)
            //     && ref_grid.iter().filter(|&&c| c == PIT_IDX + 1).count() <= 1
            //     && ref_grid.iter().filter(|&&c| c == BOX_IDX + 1).count() <= 1
            // {
                break ref_grid;
            // }
        };
        let mut grid = vec![vec![vec![false; GRID_SIZE]; GRID_SIZE]; BOX_IDX + 1];
        for (i, &val) in ref_grid.iter().enumerate() {
            let y = i / GRID_SIZE;
            let x = i % GRID_SIZE;
            if val > 0 {
                grid[val - 1][y][x] = true;
            }
        }
        let goal_pos = (4, 1);//get_empty(&ref_grid, &[]);
        let agent_pos = (1, 4);//get_empty(&ref_grid, &[goal_pos]);
        *self = Self {
            grid,
            goal_pos,
            agent_pos,
            timer: 0,
            pos_buf: VecDeque::new(),
        };
        self.get_obs()
    }

    pub fn step(&mut self, action: u32) -> (State, f32, bool, bool) {
        let mut dx = 0;
        let mut dy = 0;
        match action {
            0 => dx = -1,
            1 => dx = 1,
            2 => dy = -1,
            3 => dy = 1,
            _ => panic!(),
        }

        let mut x = (self.agent_pos.0 as i32 + dx).clamp(0, GRID_SIZE as i32 - 1) as usize;
        let mut y = (self.agent_pos.1 as i32 + dy).clamp(0, GRID_SIZE as i32 - 1) as usize;
        let mut reward = -0.001;
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
            reward += 0.1;
            self.grid[COIN_IDX][y][x] = false;
        }
        // Moving into the goal.
        else if (x, y) == self.goal_pos {
            reward += 1.;
            println!("Found goal!");
            done = true;
        }
        // Moving into a pit.
        else if self.grid[PIT_IDX][y][x] {
            reward -= 1.;
            done = true;
        }

        // Check how long we've been in the same spot
        self.agent_pos = (x, y);
        let mut trunc = self
            .pos_buf
            .iter()
            .filter(|&&p| p == self.agent_pos)
            .count()
            == MAX_SAME;
        if self.pos_buf.len() == MAX_SAME {
            self.pos_buf.pop_front();
        }
        self.pos_buf.push_back(self.agent_pos);

        self.timer += 1;
        trunc = trunc || self.timer >= MAX_TIME;

        (self.get_obs(), reward, done, trunc)
    }

    fn get_obs(&self) -> State {
        let mut state = self.grid.clone();
        let mut goal_layer = vec![vec![false; GRID_SIZE]; GRID_SIZE];
        goal_layer[self.goal_pos.1][self.goal_pos.0] = true;
        let mut agent_layer = vec![vec![false; GRID_SIZE]; GRID_SIZE];
        agent_layer[self.agent_pos.1][self.agent_pos.0] = true;
        state.push(goal_layer);
        state.push(agent_layer);
        state
    }

    pub fn render(&self) {
        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                if self.grid[COIN_IDX][y][x] {
                    print!("@");
                } else if self.grid[PIT_IDX][y][x] {
                    print!("!");
                } else if self.grid[WALL_IDX][y][x] {
                    print!("*");
                } else if self.grid[BOX_IDX][y][x] {
                    print!("O");
                } else if self.goal_pos == (x, y) {
                    print!("G");
                } else if self.agent_pos == (x, y) {
                    print!("A");
                } else {
                    print!(" ");
                }
            }
            println!();
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
fn is_border(x: i32, y: i32) -> bool {
    x < 0 || x >= GRID_SIZE as i32 || y < 0 || y >= GRID_SIZE as i32
}
