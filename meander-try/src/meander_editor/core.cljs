(ns meander-editor.core
  (:require [cljs.js :as cljs]
            [shadow.cljs.bootstrap.browser :as boot]
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

";; Namespaces already required
;; (require '[meander.match.delta :as m])

(def example [1 2 3 4 5 6])

(defn -main []
  (m/match example
    [!xs !ys ...]
    [!xs !ys]))
")


(defnc Main []
  (let [[input update-input] (<-state example-input)
        [debounced-input] (useDebounce input 200)
        [output update-output] (<-state :uninitialized)
        worker (<-ref nil)
        cm-input (react/useRef nil)]
    (<-effect (fn []
                (println "Create Worker")
                (reset! worker (js/Worker. "/js/worker.js")))
              [])
    (<-effect (fn []
                (.. @worker (addEventListener "message"
                                              (fn [e]
                                                (update-output (read-string (.-data e)))))))
              [])
    
    (<-effect
     (fn []
       (.. @worker (postMessage (prn-str debounced-input))))
     [@worker debounced-input])
   
    
    (<-effect (fn []
                (when (.-current cm-input)
                  (.init parinfer (.getCodeMirror (.-current cm-input)))))
              [cm-input])
    

    [:<>
     [:div {:style {:display :flex}}
      [:div {:style {:margin-top 10 :width "40vw" :min-width 400}}
       [code-mirror
        {:className "left-codemirror"
         :options #js {:lineNumbers true}
         :ref cm-input
         :value input
         :onChange #(update-input %)}]]


      
      
      [:div {:class "results" :style {:margin-left 50}}
       (if (= output :uninitialized)
         [:pre "loading..."]
         [:pre (with-out-str (pprint/pprint output))])]]]))



(defn render []
  (react-dom/render
   (hx/f [Main])
   (. js/document getElementById "app")))


