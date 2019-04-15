(ns meander-editor.core
  (:require [cljs.js :as cljs]
            [shadow.cljs.bootstrap.browser :as boot]
            [meander.match.gamma :as meander :include-macros true]
            [hx.react :as hx :refer [defnc]]
            [hx.hooks :refer [<-state <-effect <-ref <-callback]]
            [clojure.pprint :as pprint]
            ["react-dom" :as react-dom]
            [cljs.env :as env]
            ["use-debounce" :refer [useDebounce]]
            [cljs.reader :refer [read-string]]))



(def example-input 
  [{:title "Jimmy"}
   {:title "Janice"}
   {:title "Lemon"}])




(defn render-component [[type value]]
  (if (= type :success)
    (if (and (vector? value) (keyword? (first value)))
      value
      (with-out-str (pprint/pprint value)))
    nil))

(defnc Main []
  (let [[ready? update-ready?] (<-state false)
        [input update-input] (<-state (with-out-str (pprint/pprint example-input)))
        [lhs update-lhs] (<-state "{:title ?title}")
        [rhs update-rhs] (<-state "?title")
        [debounced-input] (useDebounce input 1000)
        [debounced-lhs] (useDebounce lhs 1000)
        [debounced-rhs] (useDebounce rhs 1000)
        [matches update-matches] (<-state [])
        worker (<-ref (js/Worker. "/js/worker.js"))]
    (<-effect (fn []
                (.. @worker (addEventListener "message" (fn [e] (update-matches (read-string (.-data e))))))
                (.. @worker (addEventListener "error" (fn [e] (js/console.log e)))))
              [])
    (<-effect (fn []
                (.. @worker (postMessage (prn-str 
                                         {:lhs debounced-lhs 
                                          :rhs debounced-rhs 
                                          :input debounced-input}))))
              [@worker debounced-lhs debounced-rhs debounced-input])
    (println "render")
    [:<>
     [:link
      {:rel "stylesheet",
       :href "https://unpkg.com/superstylin@1.0.3/src/index.css"}]
     [:div {:style {:display :flex}}
      [:div
       [:div
        [:textarea {:value input :on-change #(update-input (.. % -target -value))}]]
       [:div
        [:textarea {:value lhs :on-change #(update-lhs (.. % -target -value))}]]
       [:div
        [:textarea {:value rhs :on-change #(update-rhs (.. % -target -value))}]]]
      [:div {:style {:margin-left 50}}
       (map render-component matches)]]]))



(defn render []
  (react-dom/render
   ;; hx/f transforms Hiccup into a React element.
   ;; We only have to use it when we want to use hiccup outside of `defnc` / `defcomponent`
   (hx/f [Main])
   (. js/document getElementById "app")))


