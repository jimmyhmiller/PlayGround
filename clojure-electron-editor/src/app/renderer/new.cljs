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


;; Need to have toggles for the layers
;; Need to update state based on those toggles
;; Need to hide layer controls of hidden top-level forms
;; Need to do all of this in the codemirror state instead of an atom


(def colors ["green" "red" "blue"])


(defclass Hide
  (extends codemirror-view/WidgetType)
  (constructor [this]
               (super))
  Object
  (ignireEvents [this]
                false)
  (toDOM [this view]
         (println "HERE")
         (js/document.createTextNode "")))

(def HideDecoration
  (.replace codemirror-view/Decoration #js {:widget (Hide.)}))

(defonce ranges (atom nil))


(def my-state-field
  (.define codemirror-state/StateField
           #js {:create (fn [] (.-none codemirror-view/Decoration))
                :update (fn [things tr]
                          
                          (js/console.log things )
                          (println (first @ranges))
                          #_(.update things #js {:add #js[(.range HideDecoration
                                                                (first (first @ranges))
                                                                (second (first @ranges)))]})
                          things)
                :provide (fn [f] (.compute (.-decorations codemirror-view/EditorView) #js [f], (fn [s] (.field ^js s f))))}))



(defclass MyMarker
  (extends gutter/GutterMarker)
  (constructor [this my-height]
               (super)
               (set! height my-height))
  (field height)
  (field down)
  (field start 0)
  (field offset 0)
  (field drag 0)
  (field elem)
  Object
  (setStyle [this]
            (set!
             (.-cssText
              (.-style
               elem))
             (str
              "background-color:"
              (get colors (mod (/ (:x drag) 8 ) 3))
              ";width:10px;border-radius:20px;"
              "margin-left:" (:x drag)
              "px;height:" height "px;");
             ))
  (mouseMove [this e]
             (when down
               (set! (.-drag this) {:x(max 0 (min (+ offset
                                                     (- (.-clientX e)
                                                        (:x start)))
                                                  20))
                                    :y (- (.-clientY e)
                                          (:y start))})
               (.setStyle this)))
  (toDOM [this view]
         (let [elem (js/document.createElement "div")
               moved (.bind (.-mouseMove this) this)]
           (set! (.-elem this) elem)
           (set!
            (.-cssText
             (.-style
              elem))
            (.setStyle this))

           (set! (.-onmousedown elem)
                 (fn [e]
                   (.preventDefault e)
                   (set! (.-start this) {:x (.-clientX e)
                                         :y (.-clientY e)})

                   (set! (.-down this) true)
                   (.addEventListener js/window "mousemove" moved)
                   (.addEventListener js/window "mouseup"
                                      (fn [e]
                                        ;; Need to snap based on direction of movement
                                        (set! (.-offset this)  (* 10 ((if (neg? (- (:x drag) offset))
                                                                        js/Math.floor
                                                                        js/Math.ceil)
                                                                      (/ (:x drag) 10))))
                                        (set! (.-drag this) {:x offset})
                                        (.setStyle this)
                                        (.removeEventListener js/window "mousemove" moved true)
                                        (set! (.-down this) false)))))


           elem)
         #_(js/document.createTextNode x))
  (compare [this x y] true))



(defclass MySpacer
  (extends gutter/GutterMarker)
  (constructor [this] (super))
  Object
  (toDOM [this view]
         (let [elem (js/document.createElement "div")]
           (set!
            (.-cssText
             (.-style
              elem))
            "width:30px")
           elem)
         #_(js/document.createTextNode x))
  (compare [this x y] true))






(defonce my-state (atom nil))


@my-state

@ranges


;; This is not the right way to do this, but hacking it until I understand the facets and dispatch and all that.
;; Once I get that figured out. It shouldn't be that hard to add the layers, I don't think.
;; But surprisingly this bad method works fairly well

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


(comment)
(def my-line-numbers (gutter/lineNumbers))
(aset my-line-numbers 1 (gutter/gutter
                         #js {:lineMarker (fn [view line z]
                                            (reset! my-state (.-state view))
                                            (when-let [range (first (filter (fn [[from to]]
                                                                              (= (.-from ^js line) from)) @ranges))]
                                              (let [marker (MyMarker. (* (+ (.-height line) 5)
                                                                         (inc (-
                                                                               (.-number (.lineAt (.-doc ^js (.-state view))
                                                                                                  (second range) ))
                                                                               (.-number (.lineAt (.-doc ^js (.-state view))
                                                                                                  (first range) ))))))]
                                                (set! (.-from marker) (first range))
                                                (set! (.-to  marker) (second range))
                                                marker)))
                              :initialSpacer (fn [view] (MySpacer.))}))


(def my-theme
  (.baseTheme codemirror-view/EditorView #js {".cm-gutter" #js {:overflow "visible !important"}}))

(helix/defnc editor [{:keys [text]}]
  (let [start-state (.create codemirror-state/EditorState
                             #js {:doc text
                                  :extensions #js [highlight/defaultHighlightStyle
                                                   (gutter/lineNumbers)
                                                   my-theme
                                                   my-line-numbers
                                                   (codemirror-view/drawSelection)
                                                   #_(fold/foldGutter)
                                                   clojure-mode/default-extensions
                                                   my-state-field
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
