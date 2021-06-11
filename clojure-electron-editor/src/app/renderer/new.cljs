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
            ["@codemirror/text" :as codemirror-text]
            ["@codemirror/fold" :as fold]
            ["@codemirror/language" :as language]
            [shadow.cljs.modern :refer (defclass)]))





(defclass MyMarker
  (extends gutter/GutterMarker)
  (constructor [this xx] (super)
               (set! x xx))
  (field x)
  Object
  (toDOM [this view]
         (js/document.createTextNode x))
  (compare [this x y] true))


(js/console.log
 gutter/lineNumberMarkers)


(defonce my-state (atom nil))

(defonce ranges (atom nil))
@my-state

@ranges


;; This is not the right way to do this, but hacking it until I understand the facets and dispatch and all that.
;; Once I get that figured out. It shouldn't be that hard to add the layers, I don't think.

(add-watch my-state :top-level-forms
           (fn [_ _ _ state]
             (let [root (.-node (.cursor (language/syntaxTree state)))]
               (reset! ranges
                       (loop [top-level (.-firstChild root) ranges []]
                         (if-not top-level
                           ranges
                           (recur
                            (.-nextSibling top-level)
                            (conj ranges [(.-from top-level) (.-to top-level)]))))))))

(println (.-number (.lineAt ^codemirror-text/Text
                           (.-doc @my-state)
                           (.-to (.-nextSibling (.-firstChild (.-node (.cursor (language/syntaxTree @my-state)))))))))

(comment)
(def my-line-numbers (gutter/lineNumbers))
(aset my-line-numbers 1 (gutter/gutter
                         #js {:lineMarker (fn [view line z]
                                            (reset! my-state (.-state view))
                                            (if (some (fn [[from to]] (and (>= (.-from ^js line) from)
                                                                           (<= (.-to ^js line) to))) @ranges)
                                              (MyMarker. "|")
                                              nil))}))



(helix/defnc editor [{:keys [text]}]
  (let [start-state (.create codemirror-state/EditorState
                             #js {:doc text
                                  :extensions #js [highlight/defaultHighlightStyle
                                                    (gutter/lineNumbers)
                                                   my-line-numbers
                                                   (codemirror-view/drawSelection)
                                                   #_(fold/foldGutter)
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

(defn fact [n]
  (if (<= n 1)
    1
    (* n (fact (dec n)) )))

(defn fib [n]
  (cond (= n 0)
        0
        (= n 1)
        1
        :else (+ (fib (- n 1))
                 (fib (- n 2)))))


(helix/defnc app []
  (d/div
   (d/h1 "Hello World!!!")
   (helix/$ editor {:text "
(defn fact [n]
  (if (<= n 1)
    1
    (* n (fact (dec n)) )))

(defn fib [n]
  (cond (= n 0)
        0
        (= n 1)
        1
        :else (+ (fib (- n 1))
                 (fib (- n 2)))))

 "})))


(defn ^:dev/after-load start! []
  (react-dom/render (helix/$ app) (js/document.getElementById "app-container")))


