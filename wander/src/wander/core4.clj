(ns wander.core4)

(require '[clojure.string :as string]
         '[meander.match.delta :as r.match]
         '[meander.match.delta :as m]
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
     (r/attempt
      (r/rewrite

       0 Z

       (pred number? ?n) (succ ($ ~dec ?n))

       (+ Z ?n) ?n
       (+ ?n Z) ?n

       (+ ?n (succ ?m)) (+ (succ ?n) ?m)
       
       (fib Z) Z
       
       (fib (succ Z)) (succ Z)
       
       (fib (succ (succ ?n))) (+ (fib (succ ?n)) (fib ?n))

       )))))


(let [target__25042 {:k1 1, :k2 2}]
  (if (map? target__25042)
    (let [result__25043 ((fn [m__14273__auto__]
                           (dissoc m__14273__auto__))
                          target__25042)]
      (if (seq? result__25043)
        (let [!xs (vec result__25043)] (list !xs))
        nil))
    nil))







(fn [x] x)
(def fib-rules
  (r/rewrite

   (+ Z ?n) ?n
   (+ ?n Z) ?n

   (+ ?n (succ ?m)) (+ (succ ?n) ?m)
   
   (fib Z) Z
   (fib (succ Z)) (succ Z)
   (fib (succ (succ ?n))) (+ (fib (succ ?n)) (fib ?n))))

(def convert-numbers
  (r/rewrite
   0 Z
   (pred number? ?n) (succ ($ ~dec ?n))))


(def run-fib
  (r/until =
    (r/bottom-up
     (r/attempt
      fib-rules))))

[(run-fib '(fib Z))
 (run-fib '(fib (succ Z)))
 (run-fib '(fib (succ (succ Z))))
 (run-fib '(fib (succ (succ (succ Z)))))
 (run-fib '(fib (succ (succ (succ (succ Z))))))
 (run-fib '(fib (succ (succ (succ (succ (succ Z)))))))
 (run-fib '(fib (succ (succ (succ (succ (succ (succ Z))))))))]


[Z
 (succ Z)
 (succ Z)
 (succ (succ Z))
 (succ (succ (succ Z)))
 (succ (succ (succ (succ (succ Z)))))
 (succ (succ (succ (succ (succ (succ (succ (succ Z))))))))]
