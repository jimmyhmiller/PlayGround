const makeExp = (alg) => alg.Add(alg.Lit(2), alg.Lit(3))

const evalAlgebra = {
    Lit: n => n,
    Add: (x, y) => x + y
}
