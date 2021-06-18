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
            ["@codemirror/panel" :as panel]
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
         (js/document.createTextNode "")))

(def HideDecoration
  (.replace codemirror-view/Decoration #js {:widget (Hide.)}))

(defclass MyMarker
  (extends gutter/GutterMarker)
  (constructor [this my-height my-from]
               (super)
               (set! height my-height))
  (field from)
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
           (.setStyle this)

           (set! (.-onmousedown elem)
                 (fn [e]
                   (.preventDefault e)
                   (set! (.-start this) {:x (.-clientX e)
                                         :y (.-clientY e)})

                   (set! (.-down this) true)
                   (.addEventListener js/window "mousemove" moved)
                   (.addEventListener js/window "mouseup"
                                      (fn [e]
                                        (let [pos ((if (neg? (- (:x drag) offset))
                                                     js/Math.floor
                                                     js/Math.ceil)
                                                   (/ (:x drag) 10))]
                                          ;; Need to snap based on direction of movement
                                          (set! (.-offset this) (* 10 pos))
                                          (set! (.-drag this) {:x offset})
                                          (.setStyle this)
                                          (.removeEventListener js/window "mousemove" moved true)
                                          (set! (.-down this) false))))))


           elem))
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

(def toggle-layer (.define codemirror-state/StateEffect))

(declare layer-toggle-state)

(defn create-layer-toggle [view]
  
  (let [dom (js/document.createElement "div")
        update-dom (fn [state dom]
                     (set! (.-innerHTML dom) "")
                     (doseq [[i layer] (map-indexed vector (.field ^js state layer-toggle-state))]
                       (let [link (js/document.createElement "a")]
                         (set! (.-innerHTML link) (str layer))
                         (.addEventListener link "click"
                                            (fn [e]
                                              (.preventDefault e)
                                              (.dispatch ^js view
                                                         #js {:effects (.of toggle-layer [i (not layer)])})))
                         (.appendChild dom link))))]
    (update-dom (.-state view) dom)
    #js {:top true
         :dom dom
         :update (fn [update]
                   (update-dom (.-state update) dom))}))



(def layer-toggle-state
  (.define codemirror-state/StateField
           #js {:create (fn [] [true true true])
                :update (fn [state tr]
                          (reduce (fn [state e]
                                    (if (.is e toggle-layer)
                                      (assoc state (first (.-value e)) (second (.-value e)))
                                      state))
                                  state
                                  (.-effects ^js tr)))
                :provide (fn [f] (.from panel/showPanel f (fn [state] create-layer-toggle)))}))


(defclass Layer
  (extends rangeset/Range)
  (constructor [this _from _to _layer]
               (super _from _to _layer)
               (set! from _from)
               (set! to _to)
               (set! layer _layer))
  (field from)
  (field to)
  (field layer))




;; We need to trigger an event on layer drag.
;; We need to tie layers to the actual state
;; We need to track layers via changes so their state
;; stays in sync
;; Need to fix top level nodes not being correct.


(defn range-exists? [ranges from to]
  (let [found #js {:found false}]
    (.between ^js ranges from to (fn [a b]
                                   (if  (= a from)
                                     (do
                                       (set! (.-found found) true)
                                       false)
                                     true)))
    (.-found found)))


(rangeset/Range.
        0 0
        0)

(def track-layers-state
  (.define codemirror-state/StateField
           (let [init-layers (fn [state init-ranges]
                               (let [root (.-node (.cursor (language/syntaxTree state)))]
                                 (loop [top-level (.-firstChild root) ranges (or init-ranges (.-empty rangeset/RangeSet))]
                                   (if-not top-level
                                     ranges
                                     (recur
                                      (.-nextSibling top-level)
                                      (if (range-exists? ranges
                                                         (.-from top-level)
                                                         (.-to top-level))
                                        ranges
                                        (do
                                          (.update ranges #js {:add #js [(rangeset/Range.
                                                                          (.-from top-level)
                                                                          (.-to top-level)
                                                                          0)]}))))))))]
             #js {:create (fn [state] (init-layers state nil))
                  :update (fn [layers tr]
                            ;; Need to filter out things as well first.
                            (init-layers
                             (.-state tr)
                             (.map layers (.-changes ^js tr))))})))




(def hide-layer-state
  (.define codemirror-state/StateField
           #js {:create (fn [] (.-none codemirror-view/Decoration))
                :update (fn [hidden tr]
                         hidden
                         #_ (let [forms (.field ^js (.-state tr) track-layers-state)
                                layer-toggles (.field ^js (.-state tr) layer-toggle-state)
                                hidden (.map hidden (.-changes ^js tr))]
                            (reduce (fn [hidden form]
                                      (if (= (:layer form) 1)
                                        (.update hidden #js {:add #js [(.range HideDecoration
                                                                               (:from form)
                                                                               (:to form))]})
                                        hidden))
                                    hidden
                                    forms)))
                :provide (fn [f] (.compute (.-decorations codemirror-view/EditorView) #js [f], (fn [s] (.field ^js s f))))}))


;; This is not the right way to do this, but hacking it until I understand the facets and dispatch and all that.
;; Once I get that figured out. It shouldn't be that hard to add the layers, I don't think.
;; But surprisingly this bad method works fairly well



(defn collect-with-gas [gas rangeset]
  (let [iter (.iter ^js rangeset)]
    (loop [i 0
           coll []]
      (cond
        (= i gas) coll
        (nil? (.-value iter)) coll
        :else (let [from (.-from iter)
                    to (.-to iter)
                    value (.-value iter)]
                (.next iter)
                (recur
                 (inc i)
                 (conj coll {:from from
                             :to to
                             :value value})))))))

(comment

  (.update ranges #js {:add #js [(rangeset/Range.
                                  (.-from top-level)
                                  (.-to top-level)
                                  0)]})


  (collect-with-gas 10
   (.update
    (.update
     (.update  (.-empty rangeset/RangeSet) #js {:add #js [(rangeset/Range. 1 64 0)]})
     #js {:add #js [(rangeset/Range. 199 200 0)]})
    #js {:add #js [(rangeset/Range. 203 207 0)]})
  )


  (def rangeset (.update  (.-empty rangeset/RangeSet) #js {:add #js [(rangeset/Range. 1 64 0)]}))

  (def iter
    (.iter ^js rangeset))
  (do
    (.next iter)
    [(.-value iter)
     (.-from iter)
     (.-to iter)]))

(def layer-gutter
  (gutter/gutter
   #js {:lineMarker (fn [view line _]
                      #_(count (collect-with-gas 10 (.field ^js (.-state view) track-layers-state)))
                     (let [ranges (collect-with-gas 10 (.field ^js (.-state view) track-layers-state))]
                        (when-let [range (first (filter (fn [{:keys [from to]}]
                                                          (= (.-from ^js line) from)) ranges))]
                          (let [marker (MyMarker. (* (+ (.-height line) 5)
                                                     (inc (-
                                                           (.-number (.lineAt (.-doc ^js (.-state view))
                                                                              (:to range) ))
                                                           (.-number (.lineAt (.-doc ^js (.-state view))
                                                                              (:from range))))))
                                                  (:from range))]
                            (set! (.-from marker) (:from range))
                            (set! (.-to  marker) (:to range))
                            marker))))
        :initialSpacer (fn [view] (MySpacer.))}))


(def my-theme
  (.baseTheme codemirror-view/EditorView #js {".cm-gutter" #js {:overflow "visible !important"}}))

(helix/defnc editor [{:keys [text]}]
  (let [start-state (.create codemirror-state/EditorState
                             #js {:doc text
                                  :extensions #js [highlight/defaultHighlightStyle
                                                   (gutter/lineNumbers)
                                                   my-theme
                                                   layer-gutter
                                                   track-layers-state
                                                   (codemirror-view/drawSelection)
                                                   #_(fold/foldGutter)
                                                   layer-toggle-state
                                                   clojure-mode/default-extensions
                                                   hide-layer-state
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
