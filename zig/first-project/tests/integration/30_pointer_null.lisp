(def ptr1 (: (Pointer Int)) pointer-null)
(def ptr2 (: (Pointer Int)) pointer-null)
(def result (: Int) (if (pointer-equal? ptr1 ptr2) 1 0))
(printf (c-str "%d\n") result)
