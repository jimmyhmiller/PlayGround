//! `clojure.data.json` — a real, universally-known Clojure library — implemented
//! ENTIRELY in the language. `read-str` parses a JSON string into Clojure data
//! (maps/vectors/strings/numbers/bool/nil); `write-str` serializes it back. The
//! only string capability used is the ONE primitive `%str->chars` (plus `str` to
//! rebuild and `%char-code`/`%char-of` for code points). No parsing/encoding logic
//! is a Rust builtin — this is the proof that a real library is library code.

pub const CLOJURE_DATA_JSON: &str = r##"
(ns clojure.data.json)

;; char-code predicates (bulletproof vs char-literal ambiguity)
(defn- -ch= [c code] (if (nil? c) false (= (%char-code c) code)))
(defn- -ws? [c] (let [n (%char-code c)] (or (= n 32) (= n 9) (= n 10) (= n 13))))
(defn- -digit? [c] (let [n (%char-code c)] (and (< 47 n) (< n 58))))
(defn- -num-char? [c]
  (or (-digit? c) (let [n (%char-code c)] (or (= n 45) (= n 43) (= n 46) (= n 101) (= n 69)))))
(defn- -skip-ws [cs] (drop-while -ws? cs))

;; ── numbers ──
(defn- -pow10 [n] (if (= n 0) 1 (* 10 (-pow10 (- n 1)))))
(defn- -digits->int [ds] (reduce (fn [a c] (+ (* a 10) (- (%char-code c) 48))) 0 ds))
(defn- -chars->number [cs]
  (let [neg (-ch= (first cs) 45)
        cs (if neg (rest cs) cs)
        ipart (take-while -digit? cs)
        r1 (drop-while -digit? cs)
        ival (-digits->int ipart)]
    (if (-ch= (first r1) 46)
      (let [fpart (take-while -digit? (rest r1))
            k (count fpart)
            v (/ (+ (* ival (-pow10 k)) (-digits->int fpart)) (-pow10 k))]
        (if neg (- v) v))
      (if neg (- ival) ival))))
(defn- -parse-number [cs]
  [(-chars->number (take-while -num-char? cs)) (drop-while -num-char? cs)])

;; ── strings ──
(defn- -unescape [c]
  (let [n (%char-code c)]
    (cond (= n 110) (%char-of 10)   ; \n
          (= n 116) (%char-of 9)    ; \t
          (= n 114) (%char-of 13)   ; \r
          true c)))                 ; \" \\ \/ -> literal
(defn- -parse-string [cs acc]
  (let [c (first cs)]
    (cond
      (nil? c) (throw (ex-info "Unterminated JSON string" {}))
      (-ch= c 34) [(apply str acc) (rest cs)]
      (-ch= c 92) (-parse-string (rest (rest cs)) (conj acc (-unescape (first (rest cs)))))
      true (-parse-string (rest cs) (conj acc c)))))

;; ── value / array / object ── (each returns [value rest-chars])
(defn- -parse [cs]
  (let [cs (-skip-ws cs) c (first cs)]
    (cond
      (nil? c) (throw (ex-info "Unexpected end of JSON" {}))
      (-ch= c 123) (-parse-object (rest cs) {})   ; {
      (-ch= c 91) (-parse-array (rest cs) [])      ; [
      (-ch= c 34) (-parse-string (rest cs) [])     ; "
      (-ch= c 116) [true (drop 4 cs)]              ; t(rue)
      (-ch= c 102) [false (drop 5 cs)]             ; f(alse)
      (-ch= c 110) [nil (drop 4 cs)]               ; n(ull)
      true (-parse-number cs))))
(defn- -parse-array [cs acc]
  (let [cs (-skip-ws cs) c (first cs)]
    (cond
      (nil? c) (throw (ex-info "Unterminated array" {}))
      (-ch= c 93) [acc (rest cs)]                  ; ]
      (-ch= c 44) (-parse-array (rest cs) acc)     ; ,
      true (let [pv (-parse cs)] (-parse-array (second pv) (conj acc (first pv)))))))
(defn- -parse-object [cs acc]
  (let [cs (-skip-ws cs) c (first cs)]
    (cond
      (nil? c) (throw (ex-info "Unterminated object" {}))
      (-ch= c 125) [acc (rest cs)]                 ; }
      (-ch= c 44) (-parse-object (rest cs) acc)    ; ,
      (-ch= c 34) (let [kv (-parse-string (rest cs) [])
                        r (-skip-ws (second kv))]  ; expect ':'
                    (let [vv (-parse (rest r))]
                      (-parse-object (second vv) (assoc acc (first kv) (first vv)))))
      true (throw (ex-info "Bad JSON object" {:char c})))))

(defn read-str [s] (first (-parse (%str->chars s))))

;; ── writer ──
(defn- -num? [x] (or (= (type-of x) 'Long) (= (type-of x) 'Double)))
(defn- -esc [c]
  (let [n (%char-code c)]
    (cond (= n 34) (str (%char-of 92) (%char-of 34))
          (= n 92) (str (%char-of 92) (%char-of 92))
          (= n 10) (str (%char-of 92) (%char-of 110))
          (= n 9)  (str (%char-of 92) (%char-of 116))
          (= n 13) (str (%char-of 92) (%char-of 114))
          true (str c))))
(defn- -qstr [s] (str (%char-of 34) (apply str (map -esc (%str->chars s))) (%char-of 34)))
(defn- -key-str [k] (if (keyword? k) (name k) (str k)))
(defn write-str [x]
  (cond
    (nil? x) "null"
    (= x true) "true"
    (= x false) "false"
    (string? x) (-qstr x)
    (keyword? x) (-qstr (name x))
    (-num? x) (str x)
    (map? x) (str "{"
                  (clojure.string/join "," (map (fn [e] (str (-qstr (-key-str (first e))) ":" (write-str (second e)))) x))
                  "}")
    (vector? x) (str "[" (clojure.string/join "," (map write-str x)) "]")
    (list? x) (str "[" (clojure.string/join "," (map write-str x)) "]")
    true (throw (ex-info "Cannot JSON-encode value" {:value x}))))
"##;
