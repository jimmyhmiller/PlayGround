(ns app.renderer.new
  (:require [helix.core :as helix]
            [helix.hooks :as hooks]
            [helix.dom :as d]
            [nextjournal.clojure-mode :as clojure-mode]
            ["react-dom" :as react-dom]
            ["@codemirror/view" :as codemirror-view]
            ["@codemirror/state" :as codemirror-state]
            ["@codemirror/commands" :as codemirror-commands]
            ["@codemirror/highlight" :as highlight]
            ["@codemirror/gutter" :as gutter]
            ["@codemirror/history" :as history]
            ["@codemirror/rangeset" :as rangeset]
            [shadow.cljs.modern :refer (defclass)]))


(defclass MyMarker
  (extends gutter/GutterMarker)
  (constructor [this xx] (super)
               (set! x xx))
  (field x)
  Object
  (toDOM [this view] (println "called" x) (js/document.createTextNode x))
  (compare [this x y] true))


(js/console.log
 gutter/lineNumberMarkers)


(comment)
(def my-line-numbers  (gutter/lineNumbers))
(aset my-line-numbers 1 (gutter/gutter
                         #js {:lineMarker (fn [x line z] (js/console.log line) (if (= (.-from line) 0)  (MyMarker. "thing")  (MyMarker. "stuff") ))}))


(helix/defnc editor [{:keys [text]}]
  (let [start-state (.create codemirror-state/EditorState #js {:doc text
                                                               :extensions #js [highlight/defaultHighlightStyle
                                                                                #_  (gutter/lineNumbers)
                                                                                my-line-numbers
                                                                                (codemirror-view/drawSelection)
                                                                                clojure-mode/default-extensions
                                                                                (history/history)
                                                                                (.of codemirror-view/keymap clojure-mode/complete-keymap)
                                                                                (.of codemirror-view/keymap history/historyKeymap)]})
        [editor set-editor] (hooks/use-state nil)
        mount-cb (hooks/use-callback :once (fn [node]
                                             (set-editor
                                              (codemirror-view/EditorView.
                                               #js {:state start-state
                                                    :parent node}))))]
    ;; Why doesn't cider detect the style indent here?
    (hooks/use-effect [text]
                      (when editor
                        (.dispatch ^codemirror-view/EditorView editor
                                   (.update (.-state editor)
                                            #js {:changes #js {:from 0
                                                               :insert text
                                                               :to (.-length ^codemirror-state/Text (.-doc (.-state editor)))}}))))
    (d/div {:ref mount-cb})))


(helix/defnc app []
  (d/div
   (d/h1 "Hello World!!!")
   (helix/$ editor {:text "(defn f [x] x)
adsf"})))


(defn ^:dev/after-load start! []
  (react-dom/render (helix/$ app) (js/document.getElementById "app-container")))


