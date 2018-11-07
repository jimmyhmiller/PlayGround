(ns protocol-playground.core)

(defprotocol Foo
  (foo [x]))

(def bar 
  (with-meta [:bar]
    {`protocol-playground.core/foo (fn [x] "bar foo")}))

(def baz
  (with-meta [:baz]
    {`protocol-playground.core/foo (fn [x] "baz foo")}))

(foo bar)

(foo baz)
