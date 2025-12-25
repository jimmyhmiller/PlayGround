; Binary Trees Benchmark
; https://benchmarksgame-team.pages.debian.net/benchmarksgame/
;
; Demonstrates: recursion, heap allocation, defstruct, control flow macros

(require-dialect arith)
(require-dialect scf)
(require-dialect func)
(require-dialect llvm)

; Link C standard library for malloc/free
(link-library :c)

; Memory allocation declarations
(func.func {:sym_name "malloc" :function_type (-> [i64] [!llvm.ptr]) :sym_visibility "private"})
(func.func {:sym_name "free" :function_type (-> [!llvm.ptr] []) :sym_visibility "private"})

; Tree node with left and right pointers
(defstruct TreeNode [left !llvm.ptr] [right !llvm.ptr])

; Create a new tree node with given children
(defn tree_node [(: left !llvm.ptr) (: right !llvm.ptr)] -> !llvm.ptr
  (def node (new TreeNode))
  (TreeNode/left! node left)
  (TreeNode/right! node right)
  (func.return node))

; Build a complete binary tree of given depth
(defn bottom_up_tree [(: depth i64)] -> !llvm.ptr
  (def is_leaf (arith.cmpi {:predicate "sle"} depth (: 0 i64)))
  (def result (scf.if {:result !llvm.ptr} is_leaf
    (region (block []
      (def null (llvm.mlir.zero {:result !llvm.ptr}))
      (scf.yield (func.call {:callee @tree_node :result !llvm.ptr} null null))))
    (region (block []
      (def d1 (arith.subi depth (: 1 i64)))
      (def left (func.call {:callee @bottom_up_tree :result !llvm.ptr} d1))
      (def right (func.call {:callee @bottom_up_tree :result !llvm.ptr} d1))
      (scf.yield (func.call {:callee @tree_node :result !llvm.ptr} left right))))))
  (func.return result))

; Count nodes in tree (checksum)
(defn item_check [(: node !llvm.ptr)] -> i64
  (def left (TreeNode/left node))
  (def null (llvm.mlir.zero {:result !llvm.ptr}))
  (def is_leaf (llvm.icmp {:predicate 0} left null))
  (def result (scf.if {:result i64} is_leaf
    (region (block [] (scf.yield (: 1 i64))))
    (region (block []
      (def right (TreeNode/right node))
      (def lc (func.call {:callee @item_check :result i64} left))
      (def rc (func.call {:callee @item_check :result i64} right))
      (scf.yield (arith.addi (arith.addi lc rc) (: 1 i64)))))))
  (func.return result))

; Main: build tree at depth 12 and count nodes
; depth 12 = 2^13 - 1 = 8191 nodes
(defn main [] -> i64
  (def depth (: 12 i64))
  (def tree (func.call {:callee @bottom_up_tree :result !llvm.ptr} depth))
  (def count (func.call {:callee @item_check :result i64} tree))
  (func.return count))
