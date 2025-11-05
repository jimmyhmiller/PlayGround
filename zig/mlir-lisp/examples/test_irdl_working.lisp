;; IRDL Dialect Detection - WORKING EXAMPLE
;; This demonstrates that our 3-pass compilation system successfully:
;; 1. Detects IRDL dialect definitions
;; 2. Loads them into the MLIR context
;; 3. Continues with normal compilation

;; Define a placeholder IRDL dialect
;; (The dialect doesn't do anything useful, but proves detection works)
(operation
  (name irdl.dialect)
  (attributes {:sym_name @mydialect})
  (regions (region)))

;; Regular working program
(defn main [] i64
  (constant %c (: 42 i64))
  (return %c))
