(ns meander-editor.core
  (:require [cljs.js :as cljs]
            [shadow.cljs.bootstrap.browser :as boot]
            [meander.match.gamma :as meander :include-macros true]
            [hx.react :as hx :refer [defnc]]
            [hx.hooks :refer [<-state <-effect <-ref <-callback]]
            ["react" :as react]
            [clojure.pprint :as pprint]
            ["react-dom" :as react-dom]
            ["react-codemirror" :as code-mirror]
            ["parinfer-codemirror" :as parinfer]
            [cljs.env :as env]
            ["codemirror/mode/clojure/clojure"]
            ["use-debounce" :refer [useDebounce]]
            [cljs.reader :refer [read-string]]))



(def example-input 
  [{:title "Jimmy"}
   {:title "Janice"}
   {:title "Lemon"}])


(hx/defcomponent ErrorBoundary
  (constructor
   [this]
   (set! (. this -state)
         #js {:hasError false
              :message ""})
   this)

  ^:static (getDerivedStateFromError
            (fn [error]
              #js {:hasError true
                   :message (.toString error)}))

  (render
   [this]
   (if (.. this -state -hasError)
     [:div "Invalid html"]
     (.. this -props -children))))



(defn render-component [[type value]]
  (if (= type :success)
    (do (println value)
        (if (and (vector? value) (keyword? (first value)))
          value
          (with-out-str (pprint/pprint value))))
    nil))

(defnc Main []
  (let [[input update-input] (<-state (with-out-str (pprint/pprint example-input)))
        [lhs update-lhs] (<-state "{:title ?title}")
        [rhs update-rhs] (<-state "[:p ?title]")
        [debounced-input] (useDebounce input 200)
        [debounced-lhs] (useDebounce lhs 200)
        [debounced-rhs] (useDebounce rhs 200)
        [matches update-matches] (<-state [])
        [render-count update-render-count] (<-state 0)
        worker (<-ref nil)
        cm-input (react/useRef nil)
        cm-lhs (react/useRef nil)
        cm-rhs (react/useRef nil)]
    (<-effect (fn []
                (update-render-count (inc render-count)))
              [matches])
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
    (<-effect (fn []
                (when (.-current cm-rhs)
                  (.init parinfer (.getCodeMirror (.-current cm-rhs)))))
              [cm-rhs])
    (<-effect (fn []
                (when (.-current cm-lhs)
                  (.init parinfer (.getCodeMirror (.-current cm-lhs)))))
          [cm-rhs])
    (<-effect (fn []
                (when (.-current cm-input)
                  (.init parinfer (.getCodeMirror (.-current cm-input)))))
            [cm-rhs])
    (<-effect
     (fn []
       (.. @worker (postMessage (prn-str [:input debounced-input]))))
     [@worker debounced-input])

    [:<>
     [:div {:style {:display :flex}}
      [:div
       [:div {:style {:margin-top 10 :width 800}}
        [code-mirror
         {:options #js {:lineNumbers true}
          :ref cm-input
          :value input 
          :on-change #(update-input %)}]]
       [:div {:style {:margin-top 10 :width 800}}
        [code-mirror
         {:options #js {:lineNumbers true}
          :ref cm-lhs
          :value lhs 
          :on-change #(update-lhs %)}]]
       [:div {:style {:margin-top 10 :width 800}}
        [code-mirror
         {:options #js {:lineNumbers true}
          :ref cm-rhs
          :value rhs 
          :onChange #(update-rhs %)}]]]

      ;; Works, but every update is now a new component.
      ;; I need to be smart about only updating the key on error.
      ;; Maybe render a hidden one?
      [ErrorBoundary {:key render-count}
       [:div {:class "results" :style {:margin-left 50}}
        (map render-component matches)]]]]))



(defn render []
  (react-dom/render
   (hx/f [Main])
   (. js/document getElementById "app")))


