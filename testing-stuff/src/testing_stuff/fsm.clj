(ns fsm
  (:require [fsmviz.core :as fsmviz]
            [specviz.graphviz :as graphviz]
            [rhizome.viz :as viz]))



(defn fsm->dot [fsm]
  (str "digraph {\nrankdir=LR;\n" (graphviz/dot-string fsm) "\n}"))

(defn view-fsm [fsm]
  (->> fsm
       fsmviz/fsm->graphviz
       fsm->dot
       viz/dot->image
       viz/view-image))

