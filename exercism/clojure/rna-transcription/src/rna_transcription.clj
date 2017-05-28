(ns rna-transcription
  (:require [clojure.string :refer [replace]]))

(def translate-nuclide
  {"G" "C"
   "C" "G"
   "T" "A"
   "A" "U"})

(defn valid-dna? [dna]
  (every? #{\G \C \T \A} dna))

(defn to-rna [dna]
  {:pre [(valid-dna? dna)]}
  (replace dna #"G|C|T|A" translate-nuclide))
