(def foo (fn [] 42))
(__set_macro! (var foo))
(println "Done!")
