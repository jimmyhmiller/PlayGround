;; Test the new builtins directly

(println "Testing __is_string:")
(println "  string? on \"hello\":" (string? "hello"))
(println "  string? on 42:" (string? 42))

(println "Testing __is_symbol:")
(println "  symbol? on 'foo:" (symbol? 'foo))
(println "  symbol? on 42:" (symbol? 42))

(println "Testing symbol constructor:")
(def my-sym (symbol "my-name"))
(println "  Created symbol:" my-sym)
(println "  symbol? on it:" (symbol? my-sym))

(println "Testing gensym:")
(def gs1 (gensym))
(def gs2 (gensym "prefix__"))
(println "  gensym:" gs1)
(println "  gensym with prefix:" gs2)

(println "All builtin tests done!")
