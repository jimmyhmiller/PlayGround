; Binary Trees Benchmark - Full Implementation
; https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/binarytrees-node-7.html
;
; This is a complete implementation of the Binary Trees benchmark using
; all the high-level Lispier constructs (defn, defstruct, let, if, call, etc.)

(require-dialect arith)
(require-dialect scf)
(require-dialect func)
(require-dialect llvm)

; Link C standard library for malloc/free/putchar
(link-library :c)

; External function declarations
(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))
(extern-fn putchar (-> [i32] [i32]))

; =============================================================================
; Tree Node Structure
; =============================================================================

; Tree node with left and right pointers
(defstruct TreeNode [left !llvm.ptr] [right !llvm.ptr])

; Create a new tree node with given children
(defn tree_node [(: left !llvm.ptr) (: right !llvm.ptr)] -> !llvm.ptr
  (let [node (new TreeNode)]
    (TreeNode/left! node left)
    (TreeNode/right! node right)
    (func.return node)))

; =============================================================================
; Core Tree Operations
; =============================================================================

; Build a complete binary tree of given depth
(defn bottom_up_tree [(: depth i64)] -> !llvm.ptr
  (func.return
    (if {:result !llvm.ptr} (<=i depth (: 0 i64))
      (call !llvm.ptr tree_node (null-ptr) (null-ptr))
      (let [d1 (-i depth (: 1 i64))
            left (call !llvm.ptr bottom_up_tree d1)
            right (call !llvm.ptr bottom_up_tree d1)]
        (call !llvm.ptr tree_node left right)))))

; Count nodes in tree (checksum)
(defn item_check [(: node !llvm.ptr)] -> i64
  (func.return
    (if {:result i64} (null? (TreeNode/left node))
      (: 1 i64)
      (let [left (TreeNode/left node)
            right (TreeNode/right node)
            lc (call i64 item_check left)
            rc (call i64 item_check right)]
        (+i (+i lc rc) (: 1 i64))))))

; =============================================================================
; Printing Utilities
; =============================================================================

; Print a single digit 0-9
(defn print_digit [(: d i64)] -> i64
  (let [ascii (arith.addi (arith.trunci {:result i32} d) (: 48 i32))]
    (call i32 putchar ascii)
    (func.return (: 0 i64))))

; Print a positive integer
(defn print_number [(: n i64)] -> i64
  (func.return
    (if {:result i64} (>=i n (: 10 i64))
      (let [quotient (arith.divsi n (: 10 i64))
            remainder (-i n (*i quotient (: 10 i64)))]
        (call i64 print_number quotient)
        (call i64 print_digit remainder)
        (: 0 i64))
      (let []
        (call i64 print_digit n)
        (: 0 i64)))))

; Print a tab character
(defn print_tab [] -> i64
  (call i32 putchar (: 9 i32))
  (func.return (: 0 i64)))

; Print a newline
(defn print_newline [] -> i64
  (call i32 putchar (: 10 i32))
  (func.return (: 0 i64)))

; Print a space
(defn print_space [] -> i64
  (call i32 putchar (: 32 i32))
  (func.return (: 0 i64)))

; Print: "stretch tree of depth "
(defn print_stretch_tree_of_depth [] -> i64
  ; s t r e t c h   t r e  e     o  f     d  e  p  t  h  (space)
  ; 115 116 114 101 116 99 104 32 116 114 101 101 32 111 102 32 100 101 112 116 104 32
  (call i32 putchar (: 115 i32))  ; s
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 114 i32))  ; r
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 99 i32))   ; c
  (call i32 putchar (: 104 i32))  ; h
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 114 i32))  ; r
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 111 i32))  ; o
  (call i32 putchar (: 102 i32))  ; f
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 100 i32))  ; d
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 112 i32))  ; p
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 104 i32))  ; h
  (call i32 putchar (: 32 i32))   ; (space)
  (func.return (: 0 i64)))

; Print: " trees of depth "
(defn print_trees_of_depth [] -> i64
  ; (tab) t r e e s (space) o f (space) d e p t h (tab)
  (call i32 putchar (: 9 i32))    ; (tab)
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 114 i32))  ; r
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 115 i32))  ; s
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 111 i32))  ; o
  (call i32 putchar (: 102 i32))  ; f
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 100 i32))  ; d
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 112 i32))  ; p
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 104 i32))  ; h
  (call i32 putchar (: 32 i32))   ; (space)
  (func.return (: 0 i64)))

; Print: " check: "
(defn print_check [] -> i64
  ; (tab) c h e c k : (space)
  (call i32 putchar (: 9 i32))    ; (tab)
  (call i32 putchar (: 99 i32))   ; c
  (call i32 putchar (: 104 i32))  ; h
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 99 i32))   ; c
  (call i32 putchar (: 107 i32))  ; k
  (call i32 putchar (: 58 i32))   ; :
  (call i32 putchar (: 32 i32))   ; (space)
  (func.return (: 0 i64)))

; Print: "long lived tree of depth "
(defn print_long_lived_tree_of_depth [] -> i64
  ; l o n g (space) l i v e d (space) t r e e (space) o f (space) d e p t h (space)
  (call i32 putchar (: 108 i32))  ; l
  (call i32 putchar (: 111 i32))  ; o
  (call i32 putchar (: 110 i32))  ; n
  (call i32 putchar (: 103 i32))  ; g
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 108 i32))  ; l
  (call i32 putchar (: 105 i32))  ; i
  (call i32 putchar (: 118 i32))  ; v
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 100 i32))  ; d
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 114 i32))  ; r
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 111 i32))  ; o
  (call i32 putchar (: 102 i32))  ; f
  (call i32 putchar (: 32 i32))   ; (space)
  (call i32 putchar (: 100 i32))  ; d
  (call i32 putchar (: 101 i32))  ; e
  (call i32 putchar (: 112 i32))  ; p
  (call i32 putchar (: 116 i32))  ; t
  (call i32 putchar (: 104 i32))  ; h
  (call i32 putchar (: 32 i32))   ; (space)
  (func.return (: 0 i64)))

; =============================================================================
; Benchmark Work Function
; =============================================================================

; Build 'iterations' trees at 'depth', sum their checksums, print result
(defn work [(: iterations i64) (: depth i64)] -> i64
  ; Accumulate check sum using scf.for loop
  (def check (scf.for {:result i64} (: 0 i64) iterations (: 1 i64) (: 0 i64)
    (region
      (block [(: i i64) (: acc i64)]
        (def tree (call !llvm.ptr bottom_up_tree depth))
        (def c (call i64 item_check tree))
        (scf.yield (+i acc c))))))
  ; Print: iterations \t trees of depth \t depth \t check: \t check
  (call i64 print_number iterations)
  (call i64 print_trees_of_depth)
  (call i64 print_number depth)
  (call i64 print_check)
  (call i64 print_number check)
  (call i64 print_newline)
  (func.return (: 0 i64)))

; =============================================================================
; Main Benchmark
; =============================================================================

(defn main [] -> i64
  ; Configuration: max depth (normally from command line, hardcoded to 12 here)
  (def max_depth (: 12 i64))
  (def min_depth (: 4 i64))

  ; Stretch tree: build at maxDepth + 1, check it
  (def stretch_depth (+i max_depth (: 1 i64)))
  (def stretch_tree (call !llvm.ptr bottom_up_tree stretch_depth))
  (def stretch_check (call i64 item_check stretch_tree))

  ; Print: stretch tree of depth N \t check: N
  (call i64 print_stretch_tree_of_depth)
  (call i64 print_number stretch_depth)
  (call i64 print_check)
  (call i64 print_number stretch_check)
  (call i64 print_newline)

  ; Build long-lived tree (kept alive during benchmark)
  (def long_lived_tree (call !llvm.ptr bottom_up_tree max_depth))

  ; Loop from min_depth to max_depth, stepping by 2
  ; For each depth, calculate iterations = 1 << (maxDepth - depth + 4)
  ; We use scf.for with step 2
  (scf.for {} min_depth (+i max_depth (: 1 i64)) (: 2 i64)
    (region
      (block [(: depth i64)]
        ; iterations = 1 << (max_depth - depth + 4)
        (def shift_amount (+i (-i max_depth depth) (: 4 i64)))
        (def iterations (arith.shli (: 1 i64) shift_amount))
        (call i64 work iterations depth)
        (scf.yield))))

  ; Print long-lived tree result
  (def long_check (call i64 item_check long_lived_tree))
  (call i64 print_long_lived_tree_of_depth)
  (call i64 print_number max_depth)
  (call i64 print_check)
  (call i64 print_number long_check)
  (call i64 print_newline)

  (func.return (: 0 i64)))