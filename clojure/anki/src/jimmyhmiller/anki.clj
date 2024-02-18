(ns jimmyhmiller.anki
  (:require [clj-anki.core :as anki]
            [clj-chess.pgn :as pgn]))



(pgn/games-in-file  "/Users/jimmyhmiller/Downloads/lichess_study_london-system-gotham-chess-style_by_xGROM_2021.01.26.pgn")


(def gotham-london-study (pgn/parse-pgn (slurp "/Users/jimmyhmiller/Downloads/lichess_study_london-system-gotham-chess-style_by_xGROM_2021.01.26.pgn")))

(def notes (anki/read-notes "/Users/jimmyhmiller/Downloads/Chess_Openings.apkg"))

(def london (anki/read-notes "/Users/jimmyhmiller/Downloads/London system.apkg"))

(first london)
