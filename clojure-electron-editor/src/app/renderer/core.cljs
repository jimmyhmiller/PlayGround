(ns app.renderer.core
  (:require ["@codemirror/closebrackets" :refer [closeBrackets]]
            ["@codemirror/fold" :as fold]
            ["@codemirror/gutter" :refer [lineNumbers]]
            ["@codemirror/highlight" :as highlight]
            ["@codemirror/history" :refer [history historyKeymap]]
            ["@codemirror/state" :refer [EditorState]]
            ["@codemirror/view" :as view :refer [EditorView]]
            [nextjournal.clojure-mode.extensions.eval-region :as eval-region]
            ["lezer" :as lezer]
            ["lezer-generator" :as lg]
            ["lezer-tree" :as lz-tree]
            [applied-science.js-interop :as j]
            [clojure.string :as str]
            [nextjournal.clojure-mode :as cm-clj]
            [nextjournal.clojure-mode.extensions.close-brackets :as close-brackets]
            [nextjournal.clojure-mode.extensions.formatting :as format]
            [nextjournal.clojure-mode.extensions.selection-history :as sel-history]
            [nextjournal.clojure-mode.keymap :as keymap]
            [nextjournal.clojure-mode.live-grammar :as live-grammar]
            [nextjournal.clojure-mode.node :as n]
            [nextjournal.clojure-mode.selections :as sel]
            [nextjournal.clojure-mode.test-utils :as test-utils]
            [reagent.core :as r]
            [reagent.dom :as rdom]
            ["electron" :as electron]
            [cljs.reader]
            [clojure.pprint]
            [app.renderer.new]))




(def theme
  (.theme EditorView
          (j/lit {".cm-content" {:padding "10px 0"}
                  "&.cm-focused" {:outline "none"}
                  ".cm-line" {:padding "0 9px"
                              :line-height "1.6"
                              :font-size "16px"
                              :font-family "var(--code-font)"}
                  ".cm-matchingBracket" {:border-bottom "1px solid var(--teal-color)"
                                         :color "inherit"}
                  ".cm-gutters" {:background "transparent"
                                 :border "none"}
                  ".cm-gutterElement" {:margin-left "5px"}
                  ;; only show cursor when focused
                  ".cm-cursor" {:visibility "hidden"}
                  "&.cm-focused .cm-cursor" {:visibility "visible"}})))

(defonce extensions
  #js[theme
      (history)
      highlight/defaultHighlightStyle
      (view/drawSelection)
      (lineNumbers)
      (fold/foldGutter)
      (.. EditorState -allowMultipleSelections (of true))
      (if false
        ;; use live-reloading grammar
        #js[(cm-clj/syntax live-grammar/parser)
            (.slice cm-clj/default-extensions 1)]
        cm-clj/default-extensions)
      
      (.of view/keymap cm-clj/complete-keymap)
      (.of view/keymap historyKeymap)])


(defonce extensions-viewer
  #js[theme
      (history)
      highlight/defaultHighlightStyle
      (view/drawSelection)
      (fold/foldGutter)
      (.. EditorState -allowMultipleSelections (of true))
      (if false
        ;; use live-reloading grammar
        #js[(cm-clj/syntax live-grammar/parser)
            (.slice cm-clj/default-extensions 1)]
        cm-clj/default-extensions)
      
      (.of view/keymap cm-clj/complete-keymap)
      (.of view/keymap historyKeymap)])

(defn eval-string [expr]
  (.send electron/ipcRenderer "eval" expr))

(j/defn eval-at-cursor [on-result ^:js {:keys [state]}]
  (some->> (eval-region/cursor-node-string state)
           (eval-string))
  true)

(j/defn eval-top-level [on-result ^:js {:keys [state]}]
  (some->> (eval-region/top-level-string state)
           (eval-string))
  true)

(j/defn eval-cell [on-result ^:js {:keys [state]}]
  (-> (str "(do " (.-doc state) " )")
      (eval-string))
  true)

(defn keymap* [modifier]
  {:eval-cell
   [{:key "Mod-Enter"
     :doc "Evaluate cell"}]
   :eval-at-cursor
   [{:key (str modifier "-Enter")
     :doc "Evaluates form at cursor"}]
   :eval-top-level
   [{:key (str modifier "-Shift-Enter")
     :doc "Evaluates top-level form at cursor"}]})

(defn extension [{:keys [modifier
                         on-result]}]
  (.of view/keymap
       (j/lit
        [{:key "Mod-Enter"
          :run (partial eval-cell on-result)}
         {:key (str modifier "-Enter")
          :shift (partial eval-top-level on-result)
          :run (partial eval-at-cursor on-result)}])))




(range 100000)

(defn editor2 [{:keys [eval? source]}]
  (r/with-let [!view (r/atom nil)
               last-result (when eval? (r/atom ::no-result))
               mount! (fn [el]
                        (when el
                          (reset! !view (new EditorView
                                             (j/obj :state
                                                    (test-utils/make-state
                                                     (cond-> #js [extensions-viewer]
                                                       eval? (.concat #js [(extension {:modifier  "Alt"
                                                                                       :on-result (partial reset! last-result)})]))
                                                     source)
                                                    :parent el)))
                          (set! (.-editor js/window) @!view)))]
    [:div {:style {:display "grid"
                   :grid-template-columns "1fr 1fr"}}
     [:div {:class "rounded-md mb-0 text-sm monospace overflow-auto relative border shadow-lg bg-white"
            :ref mount!
            :style {:max-height "100vh"}}]]
    (finally
      (j/call @!view :destroy))))

(defn editor [source {:keys [eval?]}]
  (r/with-let [!view (r/atom nil)
               last-result (when eval? (r/atom ::no-result))
               
               mount! (fn [el]
                        (when el
                          
                          (.on electron/ipcRenderer "eval-result" (fn [event arg]
                                                                    (reset! last-result (try (cljs.reader/read-string arg)
                                                                                             (catch :default e arg)))))
                          (reset! !view (new EditorView
                                             (j/obj :state
                                                    (test-utils/make-state
                                                     (cond-> #js [extensions]
                                                       eval? (.concat #js [(extension {:modifier  "Alt"
                                                                                       :on-result (partial reset! last-result)})]))
                                                     source)
                                                    :parent el)))))]
    [:div {:style {:display "grid"
                   :grid-template-columns "1fr 1fr"}}
     [:div {:class "rounded-md mb-0 text-sm monospace overflow-auto relative border shadow-lg bg-white"
            :ref mount!
            :style {:max-height "100vh"}}]
     (when (and eval? (not= @last-result ::no-result))
       [:div {:style {:max-height "100vh" :overflow "scroll" :font-size 16 :font-family "monospace"}}
        (let [result @last-result]
          (cond (string? result)
                [editor2 {:key result :source result}]
                ;; Should lazily display
                (sequential? result)
                (for [[i entry] (map-indexed vector result)]
                  (cond (map? entry)
                        [:div {:key (str i entry)}
                         [:ul 
                          (for [[k v] entry]
                            ;; need to actually use parser
                            [:li {:key (str i k)
                                  :style {:list-style-type :none}}
                             [:strong {:class "ͼa"} (pr-str k) " "]
                             [:span {:class (if (string? v) "ͼc" "ͼb")}(pr-str v)]])]
                         [:hr {:style {:height 1 :background-color "#dadada" :border :none :margin-right "5%"}}]]
                        
                        :else
                        [editor2 {:key (str i entry) :source (clojure.string/trim (with-out-str (clojure.pprint/pprint entry)))}]))
                :else  [editor2 {:key result :source (pr-str result)}]))
        ])]
    (finally
      (j/call @!view :destroy))))



(defn samples []
  (into [:<>]
        (for [source ["(defn fizz-buzz [n]
  (condp (fn [a b] (zero? (mod b a))) n
    15 \"fizzbuzz\"
    3  \"fizz\"
    5  \"buzz\"
    n))



(take 100
  (repeatedly (fn []
                {:id (+ 10000 (rand-int 1000))
                 :name (first (shuffle [\"Jimmy\" \"Greg\" \"Neil\"]))
                 :likes (rand-int 3000)
                 :keywords (set (take
                                  (rand-int 3)
                                  (shuffle [\"programming\" \"philosophy\" \"society\"])))})))

(comment
  (fizz-buzz 1)
  (fizz-buzz 3)
  (fizz-buzz 5)
  (fizz-buzz 15)
  (fizz-buzz 17)
  (fizz-buzz 42))"]]
          [editor source {:eval? true}])))

(defn linux? []
  (some? (re-find #"(Linux)|(X11)" js/navigator.userAgent)))

(defn mac? []
  (and (not (linux?))
       (some? (re-find #"(Mac)|(iPhone)|(iPad)|(iPod)" js/navigator.platform))))

(defn key-mapping []
  (cond-> {"ArrowUp" "↑"
           "ArrowDown" "↓"
           "ArrowRight" "→"
           "ArrowLeft" "←"
           "Mod" "Ctrl"}
    (mac?)
    (merge {"Alt" "⌥"
            "Shift" "⇧"
            "Enter" "⏎"
            "Ctrl" "⌃"
            "Mod" "⌘"})))

(defn render-key [key]
  (let [keys (into [] (map #(get ((memoize key-mapping)) % %) (str/split key #"-")))]
    (into [:span]
          (map-indexed (fn [i k]
                         [:<>
                          (when-not (zero? i) [:span " + "])
                          [:kbd.kbd k]]) keys))))

(defn key-bindings-table []
  [:table.w-full.text-sm
   [:thead
    [:tr.border-t
     [:th.px-3.py-1.align-top.text-left.text-xs.uppercase.font-normal.black-50 "Command"]
     [:th.px-3.py-1.align-top.text-left.text-xs.uppercase.font-normal.black-50 "Keybinding"]
     [:th.px-3.py-1.align-top.text-left.text-xs.uppercase.font-normal.black-50 "Alternate Binding"]
     [:th.px-3.py-1.align-top.text-left.text-xs.uppercase.font-normal.black-50 {:style {:min-width 290}} "Description"]]]
   (into [:tbody]
         (->> keymap/paredit-keymap*
              (merge (keymap* "Alt"))
              (sort-by first)
              (map (fn [[command [{:keys [key shift doc]} & [{alternate-key :key}]]]]
                     [:<>
                      [:tr.border-t.hover:bg-gray-100
                       [:td.px-3.py-1.align-top.monospace.whitespace-nowrap [:b (name command)]]
                       [:td.px-3.py-1.align-top.text-right.text-sm.whitespace-nowrap (render-key key)]
                       [:td.px-3.py-1.align-top.text-right.text-sm.whitespace-nowrap (some-> alternate-key render-key)]
                       [:td.px-3.py-1.align-top doc]]
                      (when shift
                        [:tr.border-t.hover:bg-gray-100
                         [:td.px-3.py-1.align-top [:b (name shift)]]
                         [:td.px-3.py-1.align-top.text-sm.whitespace-nowrap.text-right
                          (render-key (str "Shift-" key))]
                         [:td.px-3.py-1.align-top.text-sm]
                         [:td.px-3.py-1.align-top]])]))))])

(defn  old-start! []
  (rdom/render [samples] (js/document.getElementById "app-container"))

  (.. (js/document.querySelectorAll "[clojure-mode]")
      (forEach #(when-not (.-firstElementChild %)
                  (rdom/render [editor (str/trim (.-innerHTML %))] %))))

  (let [mapping (key-mapping)]
    (.. (js/document.querySelectorAll ".mod,.alt,.ctrl")
        (forEach #(when-let [k (get mapping (.-innerHTML %))]
                    (set! (.-innerHTML %) k)))))

  #_(rdom/render [key-bindings-table] (js/document.getElementById "docs"))

  (when (linux?)
    (js/twemoji.parse (.-body js/document))))

(defn ^:dev/after-load start! []
  (app.renderer.new/start!))





;; Next steps

;; Cider jack-in (clj only at first)
;; Rewrite in Helix
;; Select-keys Functionality
;; Filter Functionality
;; Map Functionality
;; Lazy Render List?
;; Look at nrepl streaming?
;; Consider making own nrepl client?
;; Consider transit printing?


