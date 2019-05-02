(ns compiler.core
  (:require ["wabt" :as wabt-init])
  (:require-macros [compiler.helpers :refer [s-expr s-exprs]]))


(def wabt (wabt-init))

(defn make-module [source]
  (.-exports
    (js/WebAssembly.Instance.
     (js/WebAssembly.Module
      (.-buffer
       (.toBinary
        (.parseWat wabt "test.wat" (str source)) #js {}))))))

(defn run [source]
  (.main (make-module source)))

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
(def compile-defn)


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


(defn compile-if [[_ pred true-case false-case]]
  (s-expr (if (result i32) (i32.eq ~(compile-expr pred) ~(compile-expr true))
              (then ~(compile-expr true-case))
              (else ~(compile-expr false-case)))))

(defn $ [sym]
  (symbol (str "$" sym)))


(defn compile-defn [[_ name args & body]]
  (concat (s-expr (func ~($ name)))
          (map (fn [x] (s-expr (param ~($ x) i32))) args)
          (s-expr ((result i32)))
          (map compile-expr body)))


(defn compile-export [[_ name]]
  (s-expr (export ~(str name) (func ~($ name)))))


(defn compile-application [[name & args :as app]]
  (cond (contains? bool-operators name) (compile-bool-op app)
        (contains? number-operators name) (compile-number-op app)
        :else (concat (s-expr (call ~($ name)))
                      (map compile-expr args))))


(defn compile-expr [expr]
  (cond
    (symbol? expr) (or (operators expr) (s-expr (get_local ~($ expr))))
    (number? expr) (s-expr (i32.const ~(rep-number expr)))
    (boolean? expr) (s-expr (i32.const ~(rep-bool expr)))
    (char? expr) (s-expr (i32.const ~(rep-char expr)))
    (nil? expr) (s-expr (i32.const ~(rep-nil expr)))
    (and (seq? expr) (= (first expr) 'if)) (compile-if expr)
    (and (seq? expr) (= (first expr) 'defn)) (compile-defn expr)
    (and (seq? expr) (= (first expr) 'export)) (compile-export expr)
    (and (seq? expr) (= (first expr) 'inline)) (second expr)
    (and (seq? expr)) (compile-application expr)))

(defn compile-exprs [exprs]
  (map compile-expr exprs))

;; Multiple arity via table?
;; Need heap allocation to do real things
;; After heap allocation I guess I should make GC?
;; I should also make closures work
;; Abstract interpreter for dead code elimination would be cool

(defn compile [exprs]
  (concat '(module)
          (compile-exprs exprs)))

(unrep-number
 (run (compile
       (s-exprs

        (export main)
        (export thing)

        (defn thing [x] x)
        
        (defn fact [x]
          (if (= x 0)
            1
            (* x (fact (- x 1)))))

        (defn add [x y]
          (+ x y))
        
        (defn main []
          3)))
   ))




