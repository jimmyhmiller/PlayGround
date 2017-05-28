(ns nucleotide-count)

(def empty-count
  {\A 0
   \T 0
   \C 0
   \G 0})

(defn nucleotide-counts [nucleotide]
  (merge 
   empty-count
   (frequencies nucleotide)))

(defn count [elem nucleotide]
  {:pre [(= (empty-count elem) 0)]}
  ((nucleotide-counts nucleotide) elem))
