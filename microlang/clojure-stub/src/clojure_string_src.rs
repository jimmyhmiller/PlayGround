//! `clojure.string`, written ENTIRELY in the language on top of ONE new primitive
//! (`%str->chars`) plus `str`/`apply` to rebuild. No string operation is a Rust
//! builtin — every function here is ordinary mini-Clojure over the char list. This
//! is the proof that the primitive base is complete enough for the rest of the
//! standard library to be library code.

pub const CLOJURE_STRING: &str = r##"
(ns clojure.string)

;; ── char helpers (case mapping via code points) ──
(defn- -upper-char [c] (let [n (%char-code c)] (if (and (< 96 n) (< n 123)) (%char-of (- n 32)) c)))
(defn- -lower-char [c] (let [n (%char-code c)] (if (and (< 64 n) (< n 91)) (%char-of (+ n 32)) c)))
(defn- -ws? [c] (let [n (%char-code c)] (or (= n 32) (= n 9) (= n 10) (= n 13))))

;; ── case ──
(defn upper-case [s] (apply str (map -upper-char (%str->chars s))))
(defn lower-case [s] (apply str (map -lower-char (%str->chars s))))
(defn capitalize [s]
  (let [cs (%str->chars s)]
    (if (nil? (seq cs)) s
        (apply str (cons (-upper-char (first cs)) (map -lower-char (rest cs)))))))

;; ── reverse / blank ──
(defn reverse [s] (apply str (clojure.core/reverse (%str->chars s))))
(defn blank? [s] (if (nil? s) true (every? -ws? (%str->chars s))))

;; ── trim ──
(defn triml [s] (apply str (drop-while -ws? (%str->chars s))))
(defn trimr [s] (apply str (clojure.core/reverse (drop-while -ws? (clojure.core/reverse (%str->chars s))))))
(defn trim [s] (triml (trimr s)))

;; ── join ──
(defn join [sep & more]
  (if (nil? more)
    (apply str sep)
    (apply str (interpose sep (first more)))))

;; ── substring search (char-list scan) ──
(defn- -starts-at? [hay needle]
  (cond (nil? (seq needle)) true
        (nil? (seq hay)) false
        (= (first hay) (first needle)) (-starts-at? (rest hay) (rest needle))
        true false))
(defn- -index-loop [hay needle i]
  (cond (-starts-at? hay needle) i
        (nil? (seq hay)) nil
        true (-index-loop (rest hay) needle (+ i 1))))
(defn index-of [s value]
  (-index-loop (%str->chars s) (%str->chars (str value)) 0))
(defn last-index-of [s value]
  (let [sub (%str->chars (str value))
        n (count (%str->chars s))]
    (loop [i (index-of s value) best nil]
      (if (nil? i) best
          (let [rest-s (subs s (+ i 1))
                nxt (index-of rest-s value)]
            (recur (if (nil? nxt) nil (+ i 1 nxt)) i))))))
(defn starts-with? [s sub] (-starts-at? (%str->chars s) (%str->chars sub)))
(defn ends-with? [s sub]
  (-starts-at? (clojure.core/reverse (%str->chars s)) (clojure.core/reverse (%str->chars sub))))
(defn includes? [s sub] (not (nil? (index-of s sub))))

;; ── split / replace ──
;; drop trailing empty strings from a split result (limit-0 semantics).
(defn -drop-trailing-empty [v]
  (loop [v v] (if (and (%lt 0 (count v)) (%num-eq (count (peek v)) 0)) (recur (pop v)) v)))
;; split s at every match of the regex re.
(defn -split-re [s re]
  (-drop-trailing-empty
    (loop [pos 0 acc []]
      (if (%lt (count s) pos) acc
        (let [sub (subs s pos) m (-rx-first re sub)]
          (if (nil? m) (conj acc sub)
            (let [st (:start m) en (:end m)]
              (if (%num-eq st en) (conj acc sub)                 ; zero-width: stop
                (recur (%add pos en) (conj acc (subs sub 0 st)))))))))))
(defn split [s sep]
  (if (regexp? sep)
    (-split-re s sep)
    (let [m (str sep) mc (count m)]
      (loop [cur s acc []]
        (let [i (index-of cur m)]
          (if (nil? i)
            (conj acc cur)
            (recur (subs cur (+ i mc)) (conj acc (subs cur 0 i)))))))))
(defn split-lines [s] (split s "
"))
;; replace every regex match of re in s with the string rep (or (repfn match) when
;; rep is a fn).
(defn -replace-re [s re rep]
  (loop [pos 0 out ""]
    (if (%num-eq pos (count s)) out
      (let [sub (subs s pos) m (-rx-first re sub)]
        (if (nil? m) (str out sub)
          (let [st (:start m) en (:end m)
                r (if (fn? rep) (rep (:match m)) rep)]
            (if (%num-eq st en)
              (recur (%add pos 1) (str out (subs sub 0 1)))       ; zero-width: emit 1 char
              (recur (%add pos en) (str out (subs sub 0 st) r)))))))))
(defn replace [s match replacement]
  (if (regexp? match)
    (-replace-re s match replacement)
    (join replacement (split s match))))
(defn replace-first [s match replacement]
  (if (regexp? match)
    (let [m (-rx-first match s)]
      (if (nil? m)
        s
        (str (subs s 0 (:start m))
             (if (fn? replacement) (replacement (:match m)) replacement)
             (subs s (:end m)))))
    (let [m (str match) i (index-of s m)]
      (if (nil? i)
        s
        (str (subs s 0 i) replacement (subs s (+ i (count m))))))))

;; `re-match?` — does the pattern (string or Regex) match anywhere in s? A thin
;; predicate over the real regex engine in clojure.core.
(defn re-match? [pattern s] (not (nil? (re-find (re-pattern pattern) s))))
"##;
