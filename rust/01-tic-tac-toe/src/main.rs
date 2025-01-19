use std::fmt::{self};
use std::io::BufRead;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Cell {
    row: usize,
    column: usize,
}
impl Cell {
    fn new(row: usize, column: usize) -> Cell {
        Cell { row, column }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Marker {
    EMPTY,
    CROSS,
    NULL,
}
impl fmt::Display for Marker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            Marker::EMPTY => " ",
            Marker::CROSS => "X",
            Marker::NULL => "O",
        };
        write!(formatter, "{}", symbol)
    }
}
impl Default for Marker {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Board {
    markers: [[Marker; 3]; 3],
    count_of_free_cells: u8,
}
impl Board {
    fn get(&self, cell: &Cell) -> Marker {
        self.markers[cell.row][cell.column]
    }
    fn process_move(&mut self, cell: &Cell, marker: &Marker) -> bool {
        self.markers[cell.row][cell.column] = *marker;
        self.count_of_free_cells -= 1;
        self.is_win(cell)
    }
    fn is_win(&self, cell: &Cell) -> bool {
        self.is_win_direction(cell, 1, 0)
            || self.is_win_direction(cell, 0, 1)
            || self.is_win_direction(cell, 1, 1)
            || self.is_win_direction(cell, 1, -1)
    }
    fn is_win_direction(&self, cell: &Cell, delta_row: i32, delta_column: i32) -> bool {
        let same_markers_count =
            self.get_same_markers_count_for_direction(cell, -delta_row, -delta_column)
                + self.get_same_markers_count_for_direction(cell, delta_row, delta_column);
        same_markers_count == 2
    }
    fn get_same_markers_count_for_direction(
        &self,
        cell: &Cell,
        delta_row: i32,
        delta_column: i32,
    ) -> i32 {
        let marker = self.get(cell);
        let mut row: i32 = cell.row as i32 + delta_row;
        let mut column: i32 = cell.column as i32 + delta_column;
        let mut count = 0;
        while Self::is_in_board(row, column)
            && self.markers[row as usize][column as usize] == marker
        {
            count += 1;
            row += delta_row;
            column += delta_column;
        }
        count
    }
    fn is_in_board(row: i32, column: i32) -> bool {
        (0..3).contains(&row) && (0..3).contains(&column)
    }
    fn get_empty_cells(&self) -> Vec<Cell> {
        let mut vec = Vec::new();
        for row in 0..3 {
            for column in 0..3 {
                if self.markers[row][column] == Marker::EMPTY {
                    vec.push(Cell::new(row, column));
                }
            }
        }
        vec
    }
}
impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "+---+---+---+")?;
        for row in 0..3 {
            write!(f, "|")?;
            for marker in self.markers[row] {
                write!(f, " {} |", marker)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "+---+---+---+")
    }
}
impl Default for Board {
    fn default() -> Self {
        Self {
            markers: Default::default(),
            count_of_free_cells: 9,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Game {
    board: Board,
    marker: Marker,
    over: bool,
}
impl Game {
    fn play(&mut self) {
        println!("---- Tic-Tac-Toe ----");
        let player = Player::new();
        while !self.over && self.board.count_of_free_cells > 0 {
            let cell = if self.marker == player.marker {
                println!("{}", self.board);
                let position = player.make_a_move();
                let cell = Cell::new(position.0, position.1);
                if self.board.get(&cell) != Marker::EMPTY {
                    println!("Is not empty cell.");
                    continue;
                }
                cell
            } else {
                self.minimax()
            };
            self.make_move(&cell);
        }
        println!("Finally board state:");
        print!("{}", self.board);
        print!("The game is over. Result: ");
        if self.board.count_of_free_cells == 0 {
            println!("draw");
        } else {
            println!("player '{}' lost...", self.marker);
        }
    }
    fn make_move(&mut self, cell: &Cell) {
        self.over = self.board.process_move(cell, &self.marker);
        self.marker = if self.marker == Marker::CROSS {
            Marker::NULL
        } else {
            Marker::CROSS
        };
    }
    fn minimax(&self) -> Cell {
        let mut minimax = Minimax::new(self.marker);
        minimax.walk_tree(*self, 0);
        minimax.choise.unwrap()
    }
}
impl Default for Game {
    fn default() -> Self {
        Self {
            board: Default::default(),
            marker: Marker::CROSS,
            over: false,
        }
    }
}

struct Minimax {
    player: Marker,
    choise: Option<Cell>,
}
impl Minimax {
    fn new(player: Marker) -> Self {
        Self {
            player,
            choise: None,
        }
    }
    fn score(&self, game: &Game, depth: i32) -> i32 {
        if self.player != game.marker {
            10 - depth
        } else {
            depth - 10
        }
    }
    fn walk_tree(&mut self, game: Game, depth: i32) -> i32 {
        if game.over {
            return self.score(&game, depth);
        }
        let depth = depth + 1;
        let mut best_score = None;
        let mut best_cell = None;
        for cell in game.board.get_empty_cells() {
            let mut possible_game = game;
            possible_game.make_move(&cell);
            let possible_score = self.walk_tree(possible_game, depth);
            if best_score.is_none()
                || game.marker == self.player && best_score.unwrap() < possible_score
                || game.marker != self.player && best_score.unwrap() > possible_score
            {
                best_score = Some(possible_score);
                best_cell = Some(cell);
            }
        }
        self.choise = best_cell;
        if let Some(score) = best_score {
            score
        } else {
            0
        }
    }
}

struct Player {
    marker: Marker,
}
impl Player {
    fn new() -> Player {
        let mut stdin = std::io::stdin().lock();
        println!("Choose a side (X goes first or O goes second):");
        loop {
            let mut line = String::new();
            stdin.read_line(&mut line).unwrap();
            match Player::parse_side(&line) {
                Some(marker) => return Player { marker },
                None => println!("Invalid format. Try again:"),
            }
        }
    }
    fn make_a_move(&self) -> (usize, usize) {
        let mut stdin = std::io::stdin().lock();
        println!("Choose a cell:");
        loop {
            let mut line = String::new();
            stdin.read_line(&mut line).unwrap();
            match Player::parse_pos(&line) {
                Some(pos) => return pos,
                None => println!("Invalid format. Try again:"),
            }
        }
    }

    fn parse_side(s: &str) -> Option<Marker> {
        match s.trim() {
            "X" => Some(Marker::CROSS),
            "O" => Some(Marker::NULL),
            _ => None,
        }
    }

    fn parse_pos(s: &str) -> Option<(usize, usize)> {
        let mut parts = s.split(',');
        let row = parts.next()?.trim().parse().ok()?;
        let col = parts.next()?.trim().parse().ok()?;
        Some((row, col))
    }
}

fn main() {
    let mut game = Game::default();
    game.play();
}
