//! The multiplicity rig `R = {0, 1, ω}` and the usage order `⊑`.
//!
//! This is the same rig proved out in `../../agda/Rig.agda`. `0` = erased
//! (compile-time only), `1` = linear (used exactly once), `ω` = unrestricted.
//! Crucially `1 + 1 = ω` and `ω ⋢ 1`, so a linear value used twice fails its
//! budget — that is the mechanism behind no-double-free / no-leak.

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mult {
    Zero,
    One,
    Omega,
}

use Mult::*;

impl Mult {
    /// rig addition ("uses accumulate"):  0+x=x, 1+1=ω, ω+_=ω
    pub fn add(self, o: Mult) -> Mult {
        match (self, o) {
            (Zero, x) | (x, Zero) => x,
            (One, One) => Omega,
            _ => Omega,
        }
    }

    /// rig multiplication ("uses under a binder / at a call"):
    /// 0*_=0, 1*x=x, ω*ω=ω
    pub fn mul(self, o: Mult) -> Mult {
        match (self, o) {
            (Zero, _) | (_, Zero) => Zero,
            (One, x) | (x, One) => x,
            (Omega, Omega) => Omega,
        }
    }

    /// usage order: `usage ⊑ budget`. Everything ⊑ ω; otherwise only equals.
    /// (0 and 1 are INCOMPARABLE — that is strict linearity.)
    pub fn leq(self, budget: Mult) -> bool {
        matches!((self, budget), (_, Omega) | (One, One) | (Zero, Zero))
    }
}

impl fmt::Display for Mult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Zero => write!(f, "0"),
            One => write!(f, "1"),
            Omega => write!(f, "ω"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Mult::*;

    #[test]
    fn rig_laws() {
        assert_eq!(One.add(One), Omega); // the load-bearing fact
        assert_eq!(Zero.add(One), One);
        assert_eq!(Omega.mul(Zero), Zero);
        assert_eq!(One.mul(Omega), Omega);
        // strict linearity: 0 and 1 incomparable, both ⊑ ω
        assert!(One.leq(One) && Zero.leq(Zero));
        assert!(!Zero.leq(One) && !One.leq(Zero));
        assert!(Zero.leq(Omega) && One.leq(Omega) && Omega.leq(Omega));
        assert!(!Omega.leq(One)); // used-twice fails a linear budget
    }
}
