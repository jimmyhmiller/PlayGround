#![allow(dead_code)]
use chess::{Board, ChessMove, Piece};
use pgn_reader::{BufferedReader, RawHeader, San, SanPlus, Skip, Visitor};
use std::collections::HashMap;
use std::io::Write;
use std::io::{self, BufRead};
use std::path::Path;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;

const NULL_SAN: San = San::Null;
const NUMBER_MOVES: usize = 20;
const EMPTY_GAME: [San; NUMBER_MOVES] = [NULL_SAN; NUMBER_MOVES];

struct GameCounter {
    games: usize,
    openings: HashMap<[San; NUMBER_MOVES], usize>,
    current: [San; NUMBER_MOVES],
    index: usize,
    is_1100_game: bool,
}

impl GameCounter {
    fn new() -> GameCounter {
        GameCounter {
            games: 0,
            openings: HashMap::new(),
            current: [NULL_SAN; NUMBER_MOVES],
            index: 0,
            is_1100_game: false,
        }
    }
}

impl Visitor for GameCounter {
    type Result = usize;

    fn begin_game(&mut self) {
        self.current = EMPTY_GAME.clone();
        self.index = 0;
        self.is_1100_game = false;
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        if key == b"WhiteElo" {
            let value = value.decode_utf8().unwrap();
            if let Ok(value) = value.parse::<u32>() {
                if value > 1050 && value < 1200 {
                    self.games += 1;
                    self.is_1100_game = true;
                    if self.games % 10000 == 0 {
                        println!("{} 1100 games", self.games);
                    }
                }
            }
        }
    }

    fn san(&mut self, san_plus: SanPlus) {
        if !self.is_1100_game {
            return;
        }
        if self.index >= self.current.len() {
            return;
        }
        self.current[self.index] = san_plus.san;
        self.index += 1;
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {
        if !self.is_1100_game {
            return 0;
        }
        // if exists in openings, increment
        // else add to openings
        let opening = self.current.clone();
        if self.openings.contains_key(&opening) {
            *self.openings.get_mut(&opening).unwrap() += 1;
        } else {
            self.openings.insert(opening, 1);
        }
        0
    }
}

#[allow(dead_code)]
pub fn main_old() -> io::Result<usize> {
    // open file in buffered reader

    let file =
        std::fs::File::open("/Users/jimmyhmiller/Downloads/lichess_db_standard_rated_2022-12.pgn")?;
    let mut reader = BufferedReader::new(file);

    let mut counter = GameCounter::new();
    reader.read_all(&mut counter)?;

    // write to file
    let mut file = std::fs::File::create("openings.txt")?;

    let mut openings = counter.openings.into_iter().collect::<Vec<_>>();
    openings.sort_by(|a, b| b.1.cmp(&a.1));

    for (opening, count) in openings.iter() {
        write!(file, "{}, ", count)?;
        for san in opening {
            write!(file, "{}, ", san.to_string())?;
        }
        writeln!(file)?;
    }

    Ok(counter.games)
}

// I want to make a trie of the openings

#[derive(Clone)]
struct TrieNode {
    children: HashMap<San, TrieNode>,
    count: usize,
    san: San,
}

impl TrieNode {
    fn new(san: San, count: usize) -> Self {
        TrieNode {
            children: HashMap::new(),
            count,
            san,
        }
    }

    fn insert(&mut self, path: Vec<(San, usize)>) {
        let mut current = self;
        for (san, count) in path {
            current = current.children.entry(san.clone()).or_insert_with(|| TrieNode::new(san, count));
        }
    }
}

fn create_trie<'a>(
    parent: &'a mut TrieNode,
    index: usize,
    openings: &Vec<([San; NUMBER_MOVES], usize)>,
) {
    if index >= NUMBER_MOVES {
        return
    }
    for (opening, count) in openings.iter() {
        let mut root = &mut *parent;
        for move_ in opening.iter() {
           if root.children.contains_key(move_) {
               root = root.children.get_mut(move_).unwrap();
               root.count += count;
           } else {
               root.children.insert(
                   move_.clone(),
                   TrieNode {
                       children: HashMap::new(),
                       count: *count,
                       san: move_.clone(),
                   },
               );
               root = root.children.get_mut(move_).unwrap();
           }
        }
    }
}

fn load_openings<'a>(root: &'a mut TrieNode) -> io::Result<&'a TrieNode> {
    println!("Reading openings");
    let file = std::fs::File::open("openings.txt")?;
    let reader = io::BufReader::new(file);
    let mut openings = Vec::new();
    for line in reader.lines() {
        let mut should_include = true;
        let line = line?;
        let line = line.trim_end_matches(", ");
        let mut parts = line.split(", ");
        let count = parts.next().unwrap().parse::<usize>().unwrap();
        let mut opening = [NULL_SAN; NUMBER_MOVES];
        for (i, part) in parts.enumerate() {
            opening[i] =
                San::from_ascii(part.as_bytes()).expect(format!("Invalid SAN. {}", part).as_str());
            if opening[i] == NULL_SAN {
                should_include = false;
            }
        }
        if !should_include {
            continue;
        }
        openings.push((opening, count));
    }

    println!("Creating trie");

    create_trie(root, 0, &openings);
    walk_trie_and_remove_under_threshold(root, 100);

    Ok(root)
}

fn walk_trie_and_remove_under_threshold(trie: &mut TrieNode, threshold: usize) {
    let mut children = HashMap::new();
    for (san, child) in trie.children.iter_mut() {
        walk_trie_and_remove_under_threshold(child, threshold);
        if child.count >= threshold {
            children.insert(san.clone(), child.clone());
        }
    }
    trie.children = children;
}

fn write_trie_to_file(trie: &TrieNode, path: &str) -> io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write_trie_to_file_recursive(trie, &mut file, 0)?;
    Ok(())
}

fn write_trie_to_file_recursive(
    trie: &TrieNode,
    file: &mut std::fs::File,
    depth: usize,
) -> io::Result<()> {
    for _ in 0..depth {
        write!(file, "  ")?;
    }
    writeln!(file, "{}, {}", trie.san.to_string(), trie.count)?;
    let mut children = trie.children.iter().collect::<Vec<_>>();
    children.sort_by(|a, b| b.1.count.cmp(&a.1.count));
    for child in children.iter().map(|(_, child)| child) {
        write_trie_to_file_recursive(child, file, depth + 1)?;
    }
    Ok(())
}

fn read_file_and_construct_trie<P: AsRef<Path>>(path: P) -> io::Result<TrieNode> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut root = TrieNode::new(NULL_SAN, 0);
    let mut lines = reader.lines();
    let mut items = Vec::new();
    while let Some(Ok(line)) = lines.next() {
        let indent = line.chars().take_while(|c| c == &' ').count();
        let depth = indent / 2;
        let line = line.trim();
        let mut parts = line.split(", ");
        let san = parts.next().unwrap();
        let san = San::from_ascii(san.as_bytes()).expect(&format!("Invalid SAN. {}", san));
        let count = parts.next().unwrap().parse::<usize>().unwrap();
        items.push((san, count, depth));
    }
    let mut path : Vec<San> = Vec::new();
    let mut last_depth = 1;
    for (san, count, depth) in items {
        if san == NULL_SAN {
            continue;
        }
        if depth == 1 {
            path.clear();
            path.push(san.clone());
            last_depth = depth;
            root.children.insert(
                san.clone(),
                TrieNode {
                    children: HashMap::new(),
                    count,
                    san,
                },
            );
            continue;
        }
        if depth == last_depth {
            let mut root = &mut root;
            path.pop();
            for san in path.iter() {
                root = root.children.get_mut(san).unwrap();
            }
            path.push(san.clone());
            root.children.insert(
                san.clone(),
                TrieNode {
                    children: HashMap::new(),
                    count,
                    san,
                },
            );
            continue;
        }
        if depth > last_depth {
            last_depth = depth;
            let mut root = &mut root;
            for san in path.iter() {
                root = root.children.get_mut(san).unwrap();
            }
            path.push(san.clone());
            root.children.insert(
                san.clone(),
                TrieNode {
                    children: HashMap::new(),
                    count,
                    san,
                },
            );
            continue;
        }
        if depth < last_depth {
            path.truncate(depth - 1);
            last_depth = depth;
            let mut root = &mut root;
            for san in path.iter() {
                root = root.children.get_mut(san).unwrap();
            }
            path.push(san.clone());
            root.children.insert(
                san.clone(),
                TrieNode {
                    children: HashMap::new(),
                    count,
                    san,
                },
            );
            continue;
        }
    }

    Ok(root)
}


fn normalize_to_percentages_usize(numbers: Vec<usize>) -> Vec<usize> {

    if numbers.is_empty() {
        return numbers;
    }
    let sum: usize = numbers.iter().sum();
    let mut percentages: Vec<usize> = numbers.iter().map(|&x| ((x as f64 / sum as f64) * 100.0).round() as usize).collect();
    for p in percentages.iter_mut() {
        if *p == 0 {
            *p = 1;
        }
    }

    let sum: usize = percentages.iter().sum();
    let mut diff = sum.saturating_sub(100);
    // find the indexes of the largest numbers
    let mut indexes = (0..percentages.len()).collect::<Vec<usize>>();
    indexes.sort_by(|a, b| percentages[*b].cmp(&percentages[*a]));
    while diff > 0 {
        let indexes: Vec<&usize> = indexes.iter().filter(|&&i| percentages[i] > 1).collect::<Vec<_>>();
        for i in indexes.iter().take(diff) {
            diff -= 1;
            percentages[**i] -= 1;
        }
    }

    if sum < 100 {
        let mut diff = 100 - sum;
        while diff > 0 {
            for p in percentages.iter_mut().take(diff) {
                *p += 1;
                diff -= 1;
            }
        }
    }

    if percentages.iter().sum::<usize>() != 100 {
        println!("Percentages do not add up to 100");
    }
    if percentages.iter().any(|&x| x == 0) {
        println!("Percentages contain a 0");
    }

    percentages
}

fn normalize_counts(trie: &mut TrieNode) {
    // For each layer find the total count for that layer
    // then modify the count for each child to be the percentage of the total count
    // then call normalize_counts on each child
    // I need to normalized so they add up to 100
    // with none being 0


    let numbers : Vec<usize> = trie.children.values().map(|child| child.count).collect();
    if numbers.is_empty() {
        return;
    }
    let percentages = normalize_to_percentages_usize(numbers);

    assert!(percentages.iter().sum::<usize>() == 100, "Percentages do not add up to 100");
    assert!(percentages.iter().all(|&x| x > 0), "Percentages contain a 0");
    for (index, child) in trie.children.values_mut().enumerate() {
        child.count = percentages[index];
        normalize_counts(child);
    }
}


pub struct OpeningBook {
    trie: TrieNode,
}

trait PlayedMoveExtensions {
    fn to_san(&self, board: &Board) -> San;
}

impl PlayedMoveExtensions for ChessMove {
    fn to_san(&self, board: &Board) -> San {
        let from = self.get_source();
        let to = self.get_dest();
        let piece = board.piece_on(from).unwrap(); // Assuming a piece is there
        
        let mut san = String::new();
        
        // Pawn moves are handled differently
        if piece != Piece::Pawn {
            san.push_str(&piece.to_string(chess::Color::Black).to_uppercase());
        }
        
        // Include disambiguation if necessary (e.g., two rooks can move to the same square)
        // This is a simplified approach; a full implementation should consider all cases.
        
        // Capture notation
        if board.piece_on(to).is_some() || (piece == Piece::Pawn && from.get_file() != to.get_file()) {
            if piece == Piece::Pawn {
                san.push((b'a' + from.get_file() as u8) as char); // Include file of the pawn
            }
            san.push('x');
        }
        
        // Add the destination
        san.push_str(&format!("{}{}", (b'a' + to.get_file() as u8) as char, to.get_rank().to_index() + 1));
        
        // Check for promotion
        if let Some(promotion) = self.get_promotion() {
            san.push('=');
            san.push_str(&promotion.to_string(chess::Color::Black).to_uppercase());
        }
        
        // Check or checkmate is not directly supported here; you'd need to evaluate the board's state after the move.
        
        San::from_ascii(san.as_bytes()).expect(&format!("Invalid SAN. {}", san))
    }
}


trait SanExtensions {
    fn to_chess_move(&self, board: &Board) -> ChessMove;
}

impl SanExtensions for San {
    fn to_chess_move(&self, board: &Board) -> ChessMove {
        ChessMove::from_san(board, &self.to_string()).expect(&format!("Cannot convert SAN to ChessMove {}", self.to_string()))
    }
}


impl OpeningBook {
    pub fn open(path: &str) -> io::Result<OpeningBook> {
        let trie = read_file_and_construct_trie(path)?;
        Ok(OpeningBook { trie })
    }

    pub fn get_choices(&self, played_moves: &[ChessMove]) -> Option<OpeningBook> {
        let mut current = &self.trie;
        let mut starting_board = Board::default();
        for played_move in played_moves {
            if let Some(child) = current.children.get(&played_move.to_san(&starting_board)) {
                current = child;
            } else {
                return None;
            }
            starting_board = starting_board.make_move_new(*played_move);
        }
        if current.children.is_empty() {
            return None;
        }
        Some(OpeningBook { trie: current.clone() })
    }

    pub fn choose_weighted(&self, board: &Board) -> ChessMove {
        let children = self.trie.children.iter().collect::<Vec<_>>();
        let weights = children.iter().map(|(_, child)| child.count).collect::<Vec<usize>>();
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = rand::thread_rng();
        let san = children[dist.sample(&mut rng)].0;
        println!("Choosing {:?}, {}", san, san.to_string());
        san.to_chess_move(board)

    }
}


pub fn main() -> io::Result<usize> {

    // TODO: Convert raw numbers into percentages
    // Then I can generate a random number to choose a move

    // let trie = load_openings(&mut root)?;
    // println!("Writing trie to file");
    // write_trie_to_file(trie, "trie.txt")?;
    // println!("Reading trie from file");
    let mut trie = read_file_and_construct_trie("trie.txt")?;
    normalize_counts(&mut trie);
    write_trie_to_file(&trie, "trie3.txt")?;
    // write_trie_to_file(&trie, "trie2.txt")?;
    Ok(0)
}
