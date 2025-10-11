(def arr (: (Array Int 10)) (array Int 10 0))
(def len (: Int) (array-length arr))
(printf (c-str "%lld\n") len)
