; Binary Trees Benchmark - High-Level Version
; https://benchmarksgame-team.pages.debian.net/benchmarksgame/
;
; Demonstrates: defstruct, let, high-level macros for cleaner code
; Compare with binary_trees.lisp for the low-level version

(require-dialect arith)
(require-dialect scf)
(require-dialect func)
(require-dialect llvm)

; Link C standard library for malloc/free
(link-library :c)

; External function declarations
(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))

; Tree node with left and right pointers
(defstruct TreeNode [left !llvm.ptr] [right !llvm.ptr])

; Create a new tree node with given children
(defn tree_node [(: left !llvm.ptr) (: right !llvm.ptr)] -> !llvm.ptr
  (let [node (new TreeNode)]
    (TreeNode/left! node left)
    (TreeNode/right! node right)
    (func.return node)))

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

; Main: build tree at depth 12 and count nodes
; depth 12 = 2^13 - 1 = 8191 nodes
(defn main [] -> i64
  (func.return
    (call i64 item_check
      (call !llvm.ptr bottom_up_tree (: 12 i64)))))
