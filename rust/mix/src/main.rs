
type Byte = u8;


#[derive(Debug)]
#[derive(Clone)]
enum Sign {
    Pos,
    Neg,
}

impl Default for Sign {
    fn default() -> Self {
        Sign::Pos
    }
}


type Index = (Sign, Byte, Byte);

type Word = (Sign, Byte, Byte, Byte, Byte, Byte);

#[derive(Debug)]
#[derive(Default)]
struct SirMixAlot {
    a1: Word,
    a2: Word,
    a3: Word,
    a4: Word,
    a5: Word,
    x1: Word,
    x2: Word,
    x3: Word,
    x4: Word,
    x5: Word,
    i1: Index,
    i2: Index,
    i3: Index,
    i4: Index,
    i5: Index,
    i6: Index,
    jump: (Byte, Byte),
    memory: Vec<Word>
}

impl SirMixAlot {
    fn new() -> Self {
        SirMixAlot { 
            memory: vec![Default::default(); 4000],
            ..Default::default()
        }
    }
}



fn main() {

    println!("{:?}", 2000 & 0b00111111); // Converting numbers to two bytes
    println!("{:?}", ((2000 >> 6) & 0b00111111));

    let _example_pg128 : Word = (Sign::Pos, 16, 31, 2, 3, 8);

    let _s = SirMixAlot::new();

    println!("Hello, world!");
}
