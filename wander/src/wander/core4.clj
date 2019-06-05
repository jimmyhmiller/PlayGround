(ns wander.core4)

(require '[clojure.string :as string]
         '[meander.match.delta :as r.match]
         '[meander.strategy.delta :as r]
         '[meander.substitute.delta :as sub]
         '[hiccup.core :as hiccup])


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




(def void-tags #{:area :base :br :col :embed :hr :img :input :link :meta :param :source :track :wbr})


(defn build-attrs [attrs]
  (->>
   (r.match/search attrs
     {?key ?value}
     (str " " (name ?key) "=\"" (name ?value) "\"")))
  (string/join ""))


(macroexpand
 (quote
  (r.match/search attrs
    {?key ?value}
    (str " " (name ?key) "=\"" (name ?value) "\""))))


(defn hiccup->html [data]
  (let [rec (partial trampoline hiccup->html)]
    (r.match/match data

      (or [(pred void-tags ?tag-name) {:as !attrs} . _ ...]
          [(pred void-tags ?tag-name) . _ ...])

      (let [tag (name ?tag-name)]
        (str "<"
             tag 
             (build-attrs (first !attrs))
             " />" ))
      

      (or [?tag-name {:as !attrs} . !content ...]
          [?tag-name . !content ...])

      (let [tag (name ?tag-name)]
        (str "<"
             tag 
             (build-attrs (first !attrs))
             ">"  (string/join "" (map rec !content)) "</" tag ">"))
      
      
      (!xs ...)
      (string/join "" (map rec !xs))


      ;; Everything else.
      ?x
      ?x)))

(time
 (last (hiccup/html lots-of-hiccup)))

(time
 (last
  (hiccup->html lots-of-hiccup)))





(def arith
  (r/until =
    (r/bottom-up
     (r/rewrite

      0 Z

      (pred number? ?n) (succ ($ ~dec ?n))

      (+ Z ?n) ?n
      (+ ?n Z) ?n

      (+ ?n (succ ?m)) (+ (succ ?n) ?m)
      
      (fib Z) Z
      
      (fib (succ Z)) (succ Z)
      
      (fib (succ (succ ?n))) (+ (fib (succ ?n)) (fib ?n))

      ?x ?x
      ))))



(arith '(fib 10))
