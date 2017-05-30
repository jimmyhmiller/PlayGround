(ns testing-stuff.spec
  (:require [clojure.spec :as s]))


(s/def :html/tag #{:a :abbr :address :area :article :aside :audio :b :base :bdi :bdo :blockquote :body :br :button :canvas :caption :cite :code :col :colgroup :command :datalist :dd :del :details :dfn :div :dl :dt :em :embed :fieldset :figcaption :figure :footer :form :h1 :h2 :h3 :h4 :h5 :h6 :head :header :hgroup :hr :html :i :iframe :img :input :ins})

(s/def :html/element
  (s/cat
   :tag :html/tag
   :attrs (s/map-of keyword? string?)
   :children (s/* (s/alt :element :html/element 
                         :string string?))))


(binding [s/*recursion-limit* 5]
  (s/exercise :html/element 3))



(s/def ::binding
  (s/cat
    :name symbol?
    :value (constantly true)))

(s/def ::bindings
  (s/and vector?
         #(-> % count even?)
         (s/* ::binding)))


(s/def ::description string?)
(s/def ::given 
  (s/? (s/cat
        :given-kw #{'given}
        :bindings ::bindings
        :and (s/* (s/cat
                   :and-kw #{'and}
                   :bindings ::bindings)))))


(s/def ::when 
  (s/cat
   :when-kw #{'when}
   :bindings ::bindings))


(s/def ::then 
  (s/cat
   :then-kw #{'then}
   :variable symbol?
   :should-key #{'should-be}
   :pred (constantly true)))

(s/def ::and 
  (s/* (s/cat
        :and-kw #{'and}
        :variable symbol?
        :should-key #{'should-be}
        :pred (constantly true))))

(s/def ::scenario
  (s/cat
   :description ::description
   :given ::given
   :when ::when
   :then ::then
   :and ::and))




;; (def-gen-method scenario [desc]
;;   (? (:given context) (* :and contexts))
;;   (:when event)
;;   (cat (:then expr) (:should-be val))
;;   (* (:and exprs) (:should-be val)))

(s/fdef scenario
    :args ::scenario
    :ret any?)




(defn run-scenario [scenario]
  (let [given-bindings (s/unform ::bindings (-> scenario :given :bindings))
        and-bindings (->> scenario
                          :given
                          :and
                          (map :bindings)
                          flatten
                          (s/unform ::bindings))
        when-bindings (s/unform ::bindings (-> scenario :when :bindings))
        bindings (into [] (concat given-bindings and-bindings when-bindings))
        then (:then scenario)
        first-case [(:variable then) (:pred then)]
        rest-case (map (fn [x] [(:variable x) (:pred x)]) (:and scenario))
        cases (concat [first-case] rest-case)
        equals (map (fn [[var val]] `(= ~var ~val)) cases)]
    `(let ~bindings (assert (and ~@equals)))))


(defmacro scenario [& args]
  (let [info (s/conform ::scenario args)]
    (run-scenario info)))



(s/valid? ::given '(given [x 3] and [y 4]))
(s/valid? ::given '(given [x 3]))

(scenario "1 + 2 = 3"
          given [x 1]
          and [y 2]
          when [x (+ x y)]
          then x should-be 3
          and y should-be 2)




(s/def :js/function
  (s/cat
   :declartion #{'function}
   :name (s/? symbol?)
   :args (s/coll-of symbol?)
   :body (s/coll-of any?)))


(s/conform :js/function '(function hello (a, b, c) (
   (+ 2 2)
)))


(s/valid? ::scenario
          '("1 + 2 = 3"
            given [x 1]
            and [y 2]
            when [x (+ x y)]
            then x should-be 3
            and y should-be 1))



(s/def :simple-arith/expr
  (s/cat 
   :left (s/or :number number? :expr (s/+ :simple-arith/expr))
   :op #{'+ '- '* '/}
   :right (s/or :number number? :expr (s/+ :simple-arith/expr))))

(s/valid? :simple-arith/expr '(1 + (1 + 1)))



(s/def :arith/expr :arith/add-sub)

(s/def :arith/add-sub
  (s/or
   :mul-div :arith/mul-div
   :add :arith/add
   :sub :arith/sub))

(s/def :arith/add 
  (s/cat
   :left :arith/add-sub
   :op #{'+}
   :right :arith/mul-div))

(s/def :arith/sub
  (s/cat
   :left :arith/add-sub
   :op #{'-}
   :right :arith/mul-div))

(s/def :arith/mul-div
  (s/or
   :term :arith/term
   :mul :arith/mul
   :div :arith/div))

(s/def :arith/mul
  (s/cat
   :left :arith/mul-div
   :op #{'*}
   :right :arith/term))

(s/def :arith/div
  (s/cat
   :left :arith/mul-div
   :op #{'/}
   :right :arith/term))

(s/def :arith/term
  (s/or
   :number number?))


(s/unform :arith/expr
          [:sub
           {:left [:mul-div [:term [:number 1]]],
            :op -,
            :right [:mul {:left [:term [:number 1]], :op *, :right [:number 1]}]}])


(s/explain :arith/expr '(1 + (1 - 1)))

(binding [s/*recursion-limit* 10]
  (s/exercise :arith/add 1))
