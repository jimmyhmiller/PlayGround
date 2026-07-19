;; `clojure.pprint` — written ENTIRELY in the language, like clojure.string and
;; clojure.set beside it.
;;
;; SCOPE, stated plainly: this is `print-table` only. It is here because REAL
;; LIBRARIES need it — meander/epsilon requires clojure.pprint solely for
;; `print-table`, and could not load without it.
;;
;; `pprint`/`cl-format`/`pprint-dispatch` are deliberately ABSENT rather than
;; approximated. Clojure's pretty printer is a genuine implementation (miser
;; mode, dispatch tables, a full common-lisp FORMAT); a `(defn pprint [x]
;; (println (pr-str x)))` would print on one line while claiming to pretty
;; print, which is exactly the kind of quiet lie that is worse than a missing
;; function. Calling `pprint` here raises "Unable to resolve symbol" — loud,
;; catchable, and honest about what is not built yet.
(ns clojure.pprint)

;; A faithful port of clojure.pprint/print-table: same widths, same alignment,
;; same leading blank line, same silence on an empty `rows`. Verified against
;; real Clojure, including a missing key rendering as an empty cell.
(defn print-table
  ([ks rows]
   (when (seq rows)
     (let [widths (map (fn [k]
                         (apply max (count (str k))
                                (map (fn [row] (count (str (get row k)))) rows)))
                       ks)
           spacers (map (fn [w] (apply str (repeat w "-"))) widths)
           fmt-row (fn [leader divider trailer row]
                     (str leader
                          (apply str
                                 (interpose divider
                                            (map (fn [k w]
                                                   (format (str "%" w "s") (str (get row k))))
                                                 ks widths)))
                          trailer))]
       (println)
       (println (fmt-row "| " " | " " |" (zipmap ks ks)))
       (println (fmt-row "|-" "-+-" "-|" (zipmap ks spacers)))
       (doseq [row rows]
         (println (fmt-row "| " " | " " |" row))))))
  ([rows] (print-table (keys (first rows)) rows)))
