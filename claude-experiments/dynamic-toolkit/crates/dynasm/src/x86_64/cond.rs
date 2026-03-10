/// x86-64 condition codes for Jcc and SETcc instructions.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Condition {
    /// Equal / Zero (ZF=1)
    E = 0x4,
    /// Not Equal / Not Zero (ZF=0)
    NE = 0x5,
    /// Less (SF!=OF)
    L = 0xC,
    /// Less or Equal (ZF=1 or SF!=OF)
    LE = 0xE,
    /// Greater (ZF=0 and SF=OF)
    G = 0xF,
    /// Greater or Equal (SF=OF)
    GE = 0xD,
    /// Below / Carry (CF=1)
    B = 0x2,
    /// Below or Equal (CF=1 or ZF=1)
    BE = 0x6,
    /// Above (CF=0 and ZF=0)
    A = 0x7,
    /// Above or Equal / Not Carry (CF=0)
    AE = 0x3,
}
