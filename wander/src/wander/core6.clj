(ns wander.core6
  (:require [meander.epsilon :as m]
            [meander.strategy.epsilon :as r]
            [meander.syntax.epsilon :as syntax]
            [meander.match.syntax.epsilon :as match.syntax]
            [clojure.test.check.generators :as gen]
            [clojure.spec.alpha :as s]
            [hiccup.core :as hiccup]))




(def html
  [:html {:lang "en"}
   [:head
    [:meta {:charset "UTF-8"}]
    [:meta {:name "viewport"
            :content "width=device-width, initial-scale=1"}]
    [:title]
    [:link {:href "https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css"
            :rel "stylesheet"}]]
   [:body
    (for [color ["blue" "dark-blue" "light-blue"]]
      [:div
       [:p {:class color} color]
       [:br]])]])

(def lots-of-hiccup
  [:html {:lang "en"}
   [:head
    [:meta {:charset "UTF-8"}]
    [:meta {:name "viewport"
            :content "width=device-width, initial-scale=1"}]
    [:title]
    [:link {:href "https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css"
            :rel "stylesheet"}]]
   [:body
    (for [color (mapcat identity (repeat 1000 ["blue" "dark-blue" "light-blue"]))]
      [:div
       [:p {:class color} color]
       [:br]])]])

(def void-tags #{:area :base :br :col :embed :hr :img :input :link :meta :param :source :track :wbr})

(defn hiccup-strategy [s]
  (fn rec [t]
    ((r/pipe
      (r/rewrite
       (m/pred vector? ?x) ~(mapv rec ?x)
       (m/pred seq? ?x) ~(map rec ?x)
       ?x ?x)
      s)
     t)))

(def hiccup->html
  (hiccup-strategy
   (r/rewrite
    ;; Borrow :<> from Reagent.
    [:<> . !content ...]
    (m/app str !content ...)

    ;; Void tags.
    (with [%tag-name (m/pred void-tags ?tag-name)
           %attrs (m/seqable [!attr-names !attr-values] ...)]
          ;; Note: Content is ignored.
          (m/or [%tag-name %attrs . _ ...]
                [%tag-name . _ ...]))
    (with [%tag-name (m/app name ?tag-name)
           %attr (m/app str (m/app name !attr-names) "=" "\""  !attr-values "\"")]
          (m/app str "<"  %tag-name . " " %attr ... " />"))

    ;; Other tags.
    (m/or [?tag-name (m/seqable [!attr-names !attr-values] ...) . !content ...]
          [?tag-name . !content ...])
    (with [%tag-name (m/app name ?tag-name)
           %attr (m/app str (m/app name !attr-names) "=" "\""  !attr-values "\"")]
          (m/app str "<"  %tag-name . " " %attr ... ">" . !content ... "</" %tag-name  ">"))

    ;; Everything else.

    
    (!xs ...)
    (m/app str !xs ...)

    ?x
    ?x)))
(=
 (frequencies (hiccup/html lots-of-hiccup))
 (frequencies (hiccup->html lots-of-hiccup)))

(time
 (do
   (hiccup/html lots-of-hiccup)
   nil))

(time
 (do
   (hiccup->html lots-of-hiccup)
   nil))


(defn my-tuple [& args]
  (println (map gen/generator? args))
  (apply gen/tuple args))


(gen/sample
 (gen/let
     [?x (clojure.spec.alpha/gen number?)]
   (apply
    gen/tuple
    (list (gen/return 1) (gen/return ?x) (gen/return 2) (gen/return ?x)))))





(defn pre-transform [env]
  (r/rewrite
   {:tag :meander.match.syntax.epsilon/pred,
    :form ?pred
    :arguments ({:tag :lvr :symbol ?x})} ~(let [generator `(s/gen ~?pred)] 
                                            (swap! env assoc ?x generator)
                                            {:tag :lvr :symbol ?x})


   {:tag :lvr :symbol ?x :as ?lvr} ~(do (when-not (get @env ?x)
                                          (swap! env assoc ?x 'gen/any))
                                        ?lvr)))

(def create-lvr-generators
  (r/rewrite
   {:env (m/seqable [!keys !vals] ...) :expr ?expr}
   (gen/let [!keys !vals ...]
     ?expr)))


(def generator-transform
  (r/rewrite
   {:tag :lvr :symbol ?x} (gen/return ?x)
   {:tag :lit :value ?x} (gen/return ?x)
   {:tag :cat :elements (!elements ...)} (gen/tuple . !elements ...)
   {:tag :prt :left ?left} ~?left
   {:tag :vec :prt ?prt} (gen/fmap vec ?prt)))

(defn create-generator [lhs]
  (let [env (atom {})]
    (let [expr (match.syntax/parse lhs {})
          expr ((r/top-down (r/attempt (pre-transform env))) expr)
          value (create-lvr-generators 
                 {:env @env :expr ((r/bottom-up (r/attempt generator-transform)) expr)})]
      (eval value))))

(gen/sample
 (create-generator '[1 ?x 2 (m/pred pos-int? ?x)]))


(s/gen number?)
(s/gen
 (deref
  (resolve 'number?)))


(gen/sample
 (gen/fmap vec (apply gen/tuple (list (gen/return 1)))))


(gen/sample
 (gen/fmap (in)
           (apply gen/tuple (list (gen/return 1)))))
