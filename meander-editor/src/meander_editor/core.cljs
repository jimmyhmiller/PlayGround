(ns meander-editor.core
  (:require [cljs.js :as cljs]
            [shadow.cljs.bootstrap.browser :as boot]
            [meander-editor.eval-env]
            [meander.match.gamma :as meander :include-macros true]
            [hx.react :as hx :refer [defnc]]
            [hx.hooks :refer [<-state <-effect]]
            ["react-dom" :as react-dom]))


(defonce c-state (cljs/empty-state))

(defn eval-str [source cb]
  (cljs/eval-str
    c-state
    (str "(let [results (atom nil)] (reset! results " source ") @results)")
    "[test]"
    {:eval cljs/js-eval
     :load (partial boot/load c-state)
     :ns   (symbol "meander-editor.eval-env")}
    cb))


(defn initialize-eval [ready-cb]
  (fn []
    (boot/init c-state
               {:path         "/js/bootstrap"
                :load-on-init '#{meander-editor.eval-env}}
               (fn []
                 (ready-cb true)))))



(def example-input 
  [{:name "Jimmy"}
   {:name "Janice"}
   {:name "Lemon"}])


(defn eval-meander [ready? input lhs rhs output-cb]
  (fn []
    (when ready?
      (try (eval-str (str "(meander/match " input " " lhs " " rhs ")") output-cb)
           (catch js/Object e)))))

(defnc Main []
  (let [[ready? update-ready?] (<-state false)
        [input update-input] (<-state (prn-str example-input))
        [lhs update-lhs] (<-state "[{:name !name} ...]")
        [rhs update-rhs] (<-state "!name")
        [output update-output] (<-state nil)]
    (<-effect (initialize-eval update-ready?) [])
    (<-effect (eval-meander ready? input lhs rhs update-output)
              [ready? lhs rhs input])
    [:<>
     [:div
      [:textarea {:value input :on-change #(update-input (.. % -target -value))}]]
     [:div
      [:textarea {:value lhs :on-change #(update-lhs (.. % -target -value))}]]
     [:div
      [:textarea {:value rhs :on-change #(update-rhs (.. % -target -value))}]]
     [:p "Output: " (prn-str (:value output))]]))

(defn render []
  (react-dom/render
   ;; hx/f transforms Hiccup into a React element.
   ;; We only have to use it when we want to use hiccup outside of `defnc` / `defcomponent`
   (hx/f [Main])
   (. js/document getElementById "app")))


