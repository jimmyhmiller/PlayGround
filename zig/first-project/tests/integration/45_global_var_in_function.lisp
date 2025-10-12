;; Test: Global variables accessed from functions
;; Expected: Functions should be able to read and write global variables

(def counter (: I32) 0)

(def increment (: (-> [] I32))
  (fn []
    (set! counter (+ counter 1))
    counter))

(def get_counter (: (-> [] I32))
  (fn []
    counter))

(def test (: (-> [] I32))
  (fn []
    (increment)
    (increment)
    (get_counter)))

(test)
