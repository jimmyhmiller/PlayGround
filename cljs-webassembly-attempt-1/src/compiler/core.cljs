(ns compiler.core
  (:require ["wabt" :as wabt-init])
  (:require-macros [compiler.helpers :refer [s-expr]]))

(def wabt (wabt-init))

(defn run [source]
  (.main
   (.-exports
    (js/WebAssembly.Instance.
     (js/WebAssembly.Module
      (.-buffer
       (.toBinary
        (.parseWat wabt "test.wat" (str source)) #js {})))))))

(def number-flag 2r00)
(def immediate-flag 2r1111)
(def bool-false 2r00101111)
(def bool-true 2r01101111)
(def bool-shift 6)
(def nil-obj 2r00111111)

(def number-rep {:shift 2
                 :flag number-flag})

(def char-rep {:shift 8
               :flag immediate-flag})

(defn rep-value [{:keys [flag shift]} value]
  (bit-or flag (bit-shift-left value shift)))

(defn rep-char [char]
  (rep-value char-rep (.charCodeAt char 0)))

(defn rep-number [number]
  (rep-value number-rep number))

(defn rep-bool [bool]
  (if bool bool-true bool-false))

(defn rep-nil [value]
  (if (nil? value)
    nil-obj
    (throw (ex-info "not nill" {:value value}))))

(defn unrep-value [{:keys [shift]} value]
  (bit-shift-right value shift))

(defn unrep-number [value]
  (unrep-value number-rep value))

(defn unrep-bool [bool]
  (cond (= bool-true bool) true 
        (= bool-false bool) false
        :else (throw (ex-info "not a bool" {:value bool}))))

(defn unrep-nil [value]
  (if (= nil-obj value)
    nil))

(defn unrep-char [value]
  (char (unrep-value char-rep value)))


(def compile-if)
(def compile-expr)


(def bool-operators 
  {'= 'i32.eq
   'not= 'i32.ne
   '< 'i32.lt_s
   '<= 'i32.le_s
   '> 'i32.gt_s
   '>= 'i32.ge_s})

(def number-operators 
  {'+ 'i32.add
   '- 'i32.sub
   '* 'i32.mul
   '/ 'i32.div_s})

(def operators (merge bool-operators number-operators))

(defn to-bool [expr]
  (s-expr (i32.or (i32.shl ~expr (i32.const ~bool-shift))
                  (i32.const ~bool-false))))

(defn compile-bool-op [[op lhs rhs]]
  (to-bool (s-expr (~(compile-expr op) ~(compile-expr lhs) ~(compile-expr rhs)))))

(defn compile-number-op [[op lhs rhs]]
  (cond
    (= op '*)
    (s-expr (i32.div_s (~(compile-expr op) ~(compile-expr lhs) ~(compile-expr rhs)) (i32.const 4)))
    (= op '/)
    (s-expr (i32.mul (~(compile-expr op) ~(compile-expr lhs) ~(compile-expr rhs)) (i32.const 4)))
    :else (s-expr (~(compile-expr op) ~(compile-expr lhs) ~(compile-expr rhs)))))


(defn compile-binary-application [[op lhs rhs :as app]]
  (cond (contains? bool-operators op) (compile-bool-op app)
        (contains? number-operators op) (compile-number-op app)))

(defn compile-if [[_ pred true-case false-case]]
  (s-expr (if (result i32) (i32.eq ~(compile-expr pred) ~(compile-expr true))
              (then ~(compile-expr true-case))
              (else ~(compile-expr false-case)))))

(defn compile-expr [expr]
  (cond
    (symbol? expr) (operators expr)
    (number? expr) (s-expr (i32.const ~(rep-number expr)))
    (boolean? expr) (s-expr (i32.const ~(rep-bool expr)))
    (char? expr) (s-expr (i32.const ~(rep-char expr)))
    (nil? expr) (s-expr (i32.const ~(rep-nil expr)))
    (and (seq? expr) (= (first expr) 'if)) (compile-if expr)
    (and (seq? expr) (= (first expr) 'inline)) (second expr)
    (and (seq? expr) (= (count expr) 3)) (compile-binary-application expr)))

;; Pulling in meander here to do some pattern matching would be good.
;; It should be easy to add in primitive webassembly functions.
;; I should then add function calls (not hard?).
;; Multiple arity via table?

(defn compile [expr]
  (s-expr
   (module 
    (export "main" (func $main))
    (func $main (result i32)
           ~(compile-expr expr)))))


(unrep-bool
 (run (compile (s-expr (if (>= (/ 16 8) 2) true false)))))



