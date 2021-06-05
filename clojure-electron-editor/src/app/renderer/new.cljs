(ns app.renderer.new
  (:require [helix.core :as helix]
            [helix.hooks :as hooks]
            [helix.dom :as d]
            ["react-dom" :as react-dom]
            ["@codemirror/view" :as codemirror-view]
            ["@codemirror/state" :as codemirror-state]
            ["@codemirror/commands" :as codemirror-commands]))



(def start-state (.create codemirror-state/EditorState #js {:doc "Hello World" :extensions #js []}))




(helix/defnc editor [{:keys [text]}]
  (let [start-state (.create codemirror-state/EditorState #js {:doc text :extensions #js []})
        [editor set-editor] (hooks/use-state nil)
        mount-cb (hooks/use-callback :once (fn [node]
                                             (set-editor
                                              (codemirror-view/EditorView.
                                               #js {:state start-state
                                                    :parent node}))))]
    ;; WHy doesn't cider detext the style indent here?
    (hooks/use-effect [text]
                      (when editor
                        (.dispatch ^codemirror-view/EditorView editor
                                   (.update (.-state editor)
                                            #js {:changes #js {:from 0
                                                               :insert text
                                                               :to (.-length ^codemirror-state/Text (.-doc (.-state editor)))}}))))
    (d/div {:ref mount-cb})))


(helix/defnc app []
  (let [[text change-text] (hooks/use-state "Hello")]
    (d/div
     (d/h1 "Hello World!!!")
     (helix/$ editor {:text text})
     (helix/$ editor {:text "World!"})
     (d/div text)
     (d/button {:on-click (fn [e] (change-text "stuff"))} "Change!"))))


(defn ^:dev/after-load start! []
  (react-dom/render (helix/$ app) (js/document.getElementById "app-container")))


(start!)
