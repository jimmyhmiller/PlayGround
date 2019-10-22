(ns notre-dame.utils
  (:require [meander.epsilon :as m]))

(defn str* [xs]
  (apply str xs))

(defn to-html [hiccup]
  (m/rewrite hiccup
    ;; Borrow :<> from Reagent.
    [:<> . (m/cata !content) ...]
    (m/app str* (!content ...))

    ;; Void tags.
    (m/with [%attributes {(m/keyword !attr) !val & (m/or %attributes _)}]
      [(m/keyword (m/re #"area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr" ?tag))
       & (m/or [%attributes . _ ...]
               [_ ...])])
    (m/app str* ("<" ?tag . " " !attr "=\"" !val "\"" ... "/>"))

    ;; Normal tags.
    (m/with [%content (m/cata !content)
             %attributes {(m/keyword !attr) !val & (m/or %attributes _)}]
      [(m/keyword ?tag)
       & (m/or [%attributes . %content ...]
               [%content ...])])
    (m/app str* ("<" ?tag . " " !attr "=\"" !val "\"" ... ">" . !content ... "</" ?tag ">"))

    ;; Sequences.
    ((m/cata !content) ...)
    (m/app str* (!content ...))

    ;; Everythign else.
    ?x
    (m/app str ?x)))
