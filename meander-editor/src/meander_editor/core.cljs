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
        [debounced-input] (useDebounce input 200)
        [debounced-lhs] (useDebounce lhs 200)
        [debounced-rhs] (useDebounce rhs 200)
        [matches update-matches] (<-state [])
        worker (<-ref nil)]
    (<-effect (fn []
                (println "Create Worker")
                (reset! worker (js/Worker. "/js/worker.js")))
              [])
    (<-effect (fn []
                (.. @worker (addEventListener "message"
                                              (fn [e]
                                                (let [new-coll (read-string (.-data e))
                                                      seconds (filter identity (map second new-coll))]
                                                  (when (not (empty? seconds))
                                                    (update-matches new-coll))))))
                (.. @worker (addEventListener "error" (fn [e] (js/console.log e)))))
              [])
    (<-effect
     (fn []
       (.. @worker (postMessage (prn-str [:lhs debounced-lhs]))))
     [@worker debounced-lhs])
    (<-effect
     (fn []
       (.. @worker (postMessage (prn-str [:rhs debounced-rhs]))))
     [@worker debounced-rhs])
    (<-effect
     (fn []
       (.. @worker (postMessage (prn-str [:input debounced-input]))))
     [@worker debounced-input])
    
    (println "render")
    [:<>
     [:link
      {:rel "stylesheet",
       :href "https://unpkg.com/superstylin@1.0.3/src/index.css"}]
     [:div {:style {:display :flex}}
      [:div
       [:div
        [:textarea 
         {:style {:width 500 :margin-top 10} :value input :on-change #(update-input (.. % -target -value))}]]
       [:div
        [:textarea 
         {:style {:width 500 :margin-top 10} :value lhs :on-change #(update-lhs (.. % -target -value))}]]
       [:div
        [:textarea 
         {:style {:width 500 :margin-top 10} :value rhs :on-change #(update-rhs (.. % -target -value))}]]]
      [:div {:style {:margin-left 50}}
       (map render-component matches)]]]))



(defn render []
  (react-dom/render
   ;; hx/f transforms Hiccup into a React element.
   ;; We only have to use it when we want to use hiccup outside of `defnc` / `defcomponent`
   (hx/f [Main])
   (. js/document getElementById "app")))


