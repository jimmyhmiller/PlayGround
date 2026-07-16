;; Joins two suite.clj runs into a comparison table. Runs on JVM Clojure (it is
;; the tool, not a subject — nothing here is timed).
;;
;; usage: clojure -M compare.clj microclj.tsv jvm.tsv
;;
;; This is the gate, not a pretty-printer. It REFUSES to print a table it
;; cannot vouch for: a checksum disagreement means the two runtimes did not
;; run the same computation, so the ratio between their times is meaningless.
;; The suite it replaces had no such check and duly reported a 170x "win" that
;; was really a workload that never ran.

(require '[clojure.string :as str])

(defn- read-results [path]
  (into {}
        (for [line (str/split-lines (slurp path))
              :when (str/starts-with? line "RESULT\t")
              :let [[_ nm checksum med mn p95 spread n batch] (str/split line #"\t")]]
          [nm {:checksum checksum
               :median (Double/parseDouble med)
               :min (Double/parseDouble mn)
               :p95 (Double/parseDouble p95)
               :spread (Double/parseDouble spread)
               :n (Long/parseLong n)
               :batch (Long/parseLong batch)}])))

(def ^:private spread-limit 15.0)

(defn -main [& [micro-path jvm-path]]
  (let [micro (read-results micro-path)
        jvm (read-results jvm-path)
        names (filter #(contains? jvm %) (keys micro))
        mismatches (for [nm names
                         :when (not= (:checksum (micro nm)) (:checksum (jvm nm)))]
                     [nm (:checksum (micro nm)) (:checksum (jvm nm))])
        noisy (for [nm names
                    :let [s (max (:spread (micro nm)) (:spread (jvm nm)))]
                    :when (> s spread-limit)]
                [nm s])]

    (when (seq mismatches)
      (println "CHECKSUM MISMATCH — the two runtimes did not compute the same thing.")
      (println "No table: these timings are not comparable.\n")
      (doseq [[nm m j] mismatches]
        (println (format "  %-14s microclj=%s  jvm=%s" nm m j)))
      (System/exit 1))

    (when (empty? names)
      (println "No workloads in common between the two runs.")
      (System/exit 1))

    (println (format "%-14s %12s %12s %8s  %s" "workload" "microclj_ms" "jvm_ms" "ratio" "spread"))
    (println (apply str (repeat 60 "-")))
    (let [rows (sort-by #(- (/ (:median (micro %)) (:median (jvm %)))) names)]
      (doseq [nm rows]
        (let [m (:median (micro nm))
              j (:median (jvm nm))
              s (max (:spread (micro nm)) (:spread (jvm nm)))]
          (println (format "%-14s %12.3f %12.3f %7.2fx  %s"
                           nm m j (/ m j)
                           (if (> s spread-limit) (format "NOISY %.0f%%" s) "")))))
      (let [ratios (map #(/ (:median (micro %)) (:median (jvm %))) rows)]
        (println (apply str (repeat 60 "-")))
        (println (format "%-14s %12s %12s %7.2fx" "geomean" "" ""
                         (Math/pow (reduce * 1.0 ratios) (/ 1.0 (count ratios)))))
        (println (format "%-14s %12s %12s %7.2fx" "worst" "" "" (apply max ratios)))))

    (when (seq noisy)
      (println (format "\n%d workload(s) exceeded the %.0f%% spread limit — treat those rows as indicative only."
                       (count noisy) spread-limit)))
    (println "\nAll checksums agree across runtimes.")))

(apply -main *command-line-args*)
