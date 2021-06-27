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


(def move-layer (.define codemirror-state/StateEffect))

(defclass MyMarker
  (extends gutter/GutterMarker)
  (constructor [this my-height _from _to _value _dispatch]
               (super)
               (set! height my-height)
               (set! startSide _from)
               (set! endSide _to)
               (set! value _value)
               (set! drag {:x (* 10 (dec _value))})
               (set! offset (:x drag))
               (set! dispatch _dispatch))
  (field from)
  (field height)
  (field down)
  (field start 0)
  (field offset 0)
  (field drag 0)
  (field elem)
  (field startSide)
  (field endSide)
  (field value)
  (field dispatch)
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
               (set! (.-drag this) {:x (max 0 (min (+ offset
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
                                          (dispatch #js {:effects (.of move-layer {:from startSide
                                                                                   :to endSide
                                                                                   :value (inc pos)})})
                                          (set! (.-offset this) (* 10 pos))
                                          (set! (.-drag this) {:x offset})
                                          (.setStyle this)
                                          (.removeEventListener js/window "mousemove" moved true)
                                          (set! (.-down this) false))))))


           elem))
  (compare [this other] (and (= startSide (.-startSide ^js other))
                             (= endSide (.-endSide ^js other)))))



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
  (compare [this other] true))

(def toggle-layer (.define codemirror-state/StateEffect))

(declare layer-toggle-state)

(defn create-layer-toggle [view]
  
  (let [dom (js/document.createElement "div")
        update-dom (fn [state dom]
                     (set! (.-innerHTML dom) "")
                     (doseq [[i layer] (map-indexed vector (.field ^js state layer-toggle-state))]
                       (let [link (js/document.createElement "a")]
                         #_(set! (.-innerHTML link) (str layer))
                         (set! (.-cssText (.-style link))
                               (str "border-radius: 20px;
                                     width: 15px;
                                     height: 15px;
                                     display: inline-block;
                                     margin-right: 3px;
                                     margin-top: 2px;"
                                    "background-color: " (get colors i) ";"
                                    (when-not layer
                                      "opacity: 0.4;")))
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
                   (doseq [t (.-transactions ^js update)]
                     (doseq [e (.-effects ^js t)]
                       (when (.is e toggle-layer)
                         (update-dom (.-state update) dom)))))}))



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



(defn find-range [ranges from to]
  (let [found #js {:found false}]
    (.between ^js ranges from to (fn [from* to* value]
                                   (if (= from* from)
                                     (do
                                       (set! (.-found found) value)
                                       false)
                                     true)))
    (.-found found)))



;; If range exists, copy layer (aka value)
;; If not add
;; Check of layer change events and update
;; Make markers pay attention to state


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


(def track-layers-state
  (.define codemirror-state/StateField
           (let [init-layers (fn [state init-ranges]
                               ;; Need to do this properly, probably actually use the cursor?
                               ;; Right now it stops early
                              (let [root (.-node (.cursor (language/syntaxTree state)))]
                                 (loop [top-level (.-firstChild root) ranges (.-empty rangeset/RangeSet)]
                                   
                                   (if-not top-level
                                     ranges
                                     (do
                                       (js/console.log "top-level" top-level)
                                       (recur
                                        (.-nextSibling top-level)
                                        (let [existing-range (or (find-range init-ranges
                                                                             (.-from top-level)
                                                                             (.-to top-level))
                                                                 1)]
                                          (.update ranges #js {:add #js [(rangeset/Range.
                                                                          (.-from top-level)
                                                                          (.-to top-level)
                                                                          (or existing-range 1))]}))))))))]
             #js {:create (fn [state]  (init-layers state (.-empty rangeset/RangeSet)))
                  :update (fn [layers transaction]
                            (let [moved-layers (.map layers (.-changes ^js transaction))
                                  new-layers (reduce (fn [acc effect]
                                                       (if (.is ^js effect move-layer)
                                                         (let [{:keys [from to value]} (.-value ^js effect)]
                                                           (.update acc #js {:filter (fn [from* _ _] (not= from* from))
                                                                             :add #js [(rangeset/Range.
                                                                                        from
                                                                                        to
                                                                                        value)]}))
                                                         acc))
                                                     moved-layers
                                                     (.-effects ^js transaction))]
                              ;; Need to filter out things as well first.
                              (init-layers
                               (.-state transaction)
                               new-layers)))})))




(def hide-layer-state
  (.define codemirror-state/StateField
           #js {:create (fn [] (.-none codemirror-view/Decoration))
                :update (fn [hidden tr]
                          (let [forms (.field ^js (.-state tr) track-layers-state)
                                layer-toggles (.field ^js (.-state tr) layer-toggle-state)
                                hidden (.map hidden (.-changes ^js tr))]
                            #_(println forms layer-toggles hidden)
                            hidden
                            (reduce (fn [hidden form]
                                      (if-not (nth layer-toggles (dec (:value form)))
                                        (.update hidden #js {:add #js [(.range HideDecoration
                                                                               (:from form)
                                                                               (:to form))]})
                                        (.update hidden #js {:filter (fn [from* _ _]
                                                                       (not= (:from form) from*))})))
                                    hidden
                                    (collect-with-gas 10 forms))))
                :provide (fn [f] (.compute (.-decorations codemirror-view/EditorView) #js [f], (fn [s] (.field ^js s f))))}))


;; This is not the right way to do this, but hacking it until I understand the facets and dispatch and all that.
;; Once I get that figured out. It shouldn't be that hard to add the layers, I don't think.
;; But surprisingly this bad method works fairly well




(defn into-rangeset [coll]
  (reduce (fn [acc x]
            (.update acc #js {:add #js [(rangeset/Range. (.-startSide ^js x) (.-endSide ^js x) x)]}))
          (.-empty rangeset/RangeSet)
          coll))



(def layer-gutter
  (gutter/gutter
   #js {:markers (fn [view]
                   (let [layer-toggles (.field ^js (.-state ^js view) layer-toggle-state)
                         ranges (collect-with-gas 10000 (.field ^js (.-state view) track-layers-state))]
                     (into-rangeset
                      (filter identity
                              (for [range ranges]
                                ;; TODO: account for line height
                                (when (nth layer-toggles (dec (:value range)))
                                  (MyMarker. (* 18
                                                (inc (-
                                                      (.-number (.lineAt (.-doc ^js (.-state view))
                                                                         (:to range) ))
                                                      (.-number (.lineAt (.-doc ^js (.-state view))
                                                                         (:from range))))))
                                             (:from range)
                                             (:to range)
                                             (:value range)
                                             (.-dispatch ^js view))))))))
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
                                                   layer-toggle-state
                                                   track-layers-state
                                                   (codemirror-view/drawSelection)
                                                   #_(fold/foldGutter)
                                                  
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
