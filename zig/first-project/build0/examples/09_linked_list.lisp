;; Example 9: Linked List
;; Demonstrates: Pointers, recursive data structures, heap allocation, struct manipulation

(include-header "stdio.h")
(include-header "stdlib.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn free [ptr (Pointer U8)] -> Nil)

;; Node structure (need to use pointer for recursive type)
(def Node (: Type)
  (Struct [value I32] [next (Pointer Node)]))

;; Create a new node
(def make-node (: (-> [I32 (Pointer Node)] (Pointer Node)))
  (fn [value next]
    (let [node (: (Pointer Node)) (allocate Node (Node value pointer-null))]
      (pointer-field-write! node value value)
      (pointer-field-write! node next next)
      node)))

;; Prepend a value to a list
(def prepend (: (-> [I32 (Pointer Node)] (Pointer Node)))
  (fn [value list]
    (make-node value list)))

;; Print the list
(def print-list (: (-> [(Pointer Node)] I32))
  (fn [list]
    (if (pointer-equal? list pointer-null)
        (printf (c-str "nil\n"))
        (let [_1 (: I32) (printf (c-str "%d -> ") (pointer-field-read list value))]
          (print-list (pointer-field-read list next))))))

;; Get list length
(def list-length (: (-> [(Pointer Node)] I32))
  (fn [list]
    (if (pointer-equal? list pointer-null)
        0
        (+ 1 (list-length (pointer-field-read list next))))))

;; Sum all values in the list
(def list-sum (: (-> [(Pointer Node)] I32))
  (fn [list]
    (if (pointer-equal? list pointer-null)
        0
        (+ (pointer-field-read list value)
           (list-sum (pointer-field-read list next))))))

;; Note: free-list omitted due to deallocate codegen bug (see BUGS.md)

(def main-fn (: (-> [] I32))
  (fn []
    ;; Create a list: 1 -> 2 -> 3 -> 4 -> 5 -> nil
    (let [list (: (Pointer Node)) pointer-null]
      (set! list (prepend 5 list))
      (set! list (prepend 4 list))
      (set! list (prepend 3 list))
      (set! list (prepend 2 list))
      (set! list (prepend 1 list))

      (printf (c-str "List: "))
      (print-list list)

      (printf (c-str "Length: %d\n") (list-length list))
      (printf (c-str "Sum: %d\n") (list-sum list))
      0)))

(main-fn)
