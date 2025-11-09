;; Test: Dialect Detection
;; This file tests that the system can detect and report IRDL operations
;; We're not trying to create a valid IRDL dialect yet, just testing detection

;; A placeholder IRDL dialect definition
;; (This won't actually work as IRDL, but should be detected)
(operation
  (name irdl.dialect)
  (attributes {:sym_name @testdialect})
  (regions (region)))

;; A placeholder transform operation
(operation
  (name transform.sequence)
  (attributes {:failure_propagation_mode (: 1 i32)})
  (regions (region)))

;; Regular program code
(defn main [] i64
  (constant %c (: 42 i64))
  (return %c))
