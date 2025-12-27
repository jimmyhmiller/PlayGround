; Binary Trees Benchmark - Full Implementation
; https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/binarytrees-node-7.html
;
; This is a complete implementation of the Binary Trees benchmark using
; all the high-level Lispier constructs (defn, defstruct, let, if, call, etc.)

(require-dialect arith)
(require-dialect scf)
(require-dialect func)
(require-dialect llvm)

; Link C standard library for malloc/free/putchar/printf
(link-library :c)

; External function declarations
(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))
(extern-fn putchar (-> [i32] [i32]))
(extern-fn atoi (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr] [i32]))

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
  (print "\t trees of depth ")
  (call i64 print_number depth)
  (print "\t check: ")
  (call i64 print_number check)
  (println "")
  (func.return (: 0 i64)))

; =============================================================================
; Helper: get max of two values
; =============================================================================

(defn max_i64 [(: a i64) (: b i64)] -> i64
  (func.return
    (if {:result i64} (>=i a b)
      a
      b)))

; =============================================================================
; Main Benchmark
; =============================================================================

(defn main [(: argc i64) (: argv !llvm.ptr)] -> i64
  ; Parse max depth from command line, default to 6 if not provided
  ; Like the reference: const maxDepth = Math.max(6, parseInt(process.argv[2]))
  (def input_depth
    (if {:result i64} (>=i argc (: 1 i64))
      ; Get argv[0] (first program argument after the file)
      (let [arg_ptr (llvm.load {:result !llvm.ptr} argv)
            parsed (call i32 atoi arg_ptr)]
        (arith.extsi {:result i64} parsed))
      (: 6 i64)))

  ; max_depth = max(6, input_depth)
  (def max_depth (call i64 max_i64 (: 6 i64) input_depth))
  (def min_depth (: 4 i64))

  ; Stretch tree: build at maxDepth + 1, check it
  (def stretch_depth (+i max_depth (: 1 i64)))
  (def stretch_tree (call !llvm.ptr bottom_up_tree stretch_depth))
  (def stretch_check (call i64 item_check stretch_tree))

  ; Print: stretch tree of depth N \t check: N
  (print "stretch tree of depth ")
  (call i64 print_number stretch_depth)
  (print "\t check: ")
  (call i64 print_number stretch_check)
  (println "")

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
  (print "long lived tree of depth ")
  (call i64 print_number max_depth)
  (print "\t check: ")
  (call i64 print_number long_check)
  (println "")

  (func.return (: 0 i64)))
