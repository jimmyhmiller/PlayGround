(ns examples-live-view.code-view
  (:require [live-view-server.core :as live-view]
            [clojure.repl]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.pprint :as pprint]
            [glow.html]
            [examples-live-view.parse :as parse]
            [glow.core]))


;; Slightly modified from clojure.repl
(defn source-fn
  "Returns a string of the source code for the given symbol, if it can
  find it.  This requires that the symbol resolve to a Var defined in
  a namespace for which the .clj is in the classpath.  Returns nil if
  it can't find the source.  For most REPL usage, 'source' is more
  convenient.

  Example: (source-fn 'filter)"
  [x]
  (when-let [v (resolve x)]
    (when-let [filepath (:file (meta v))]
      (when-let [strm (io/input-stream (io/file filepath))]
        (with-open [rdr (java.io.LineNumberReader. (java.io.InputStreamReader. strm))]
          (dotimes [_ (dec (:line (meta v)))] (.readLine rdr))
          (let [text (StringBuilder.)
                pbr (proxy [java.io.PushbackReader] [rdr]
                      (read [] (let [i (proxy-super read)]
                                 (.append text (char i))
                                 i)))
                read-opts (if (.endsWith ^String filepath "cljc") {:read-cond :allow} {})]
            (if (= :unknown *read-eval*)
              (throw (IllegalStateException. "Unable to read source while *read-eval* is :unknown."))
              (read read-opts (java.io.PushbackReader. pbr)))
            (str text)))))))


(defn pprint-str [coll]
  (let [out (java.io.StringWriter.)]
    (pprint/pprint coll out)
    (str out)))



(def instrumented-vars (atom {}))


(defonce state (atom {:calls {}
                      :namespace-input ""}))

(reset! state {:calls {}
                      :namespace-input ""})


;; Tap has some positive and some negative attributes. Because it
;; intentionally doesn't block on puts, it doesn't have a large probe
;; effect. But it also has a 1024 fixed buffer size, and no watch to
;; batch results. I could probably make my own system fairly easily
;; that overcomes these issues, but for now this works fairly well for
;; most uses cases.
(do
  ;; Little song and dance to make this repl safe.
  (def add-tap-to-state)
  (remove-tap add-tap-to-state)

  (defn add-tap-to-state [{:keys [var-name args results]}]
    (swap! state (fn [state]
                   (-> state
                       (update-in [:calls var-name :number] (fnil inc 0))
                       (update-in [:calls var-name :values] (fnil conj []) {:args args
                                                                            :results results})))))

  (add-tap add-tap-to-state))


(defn stats-handling [v f]
  (fn [& args]
    ;; This ensures we don't infinite loop by adding stats to our own
    ;; function calls.
    (if (= (get live-view/*live-view-context* :app) :code-view)
      (apply f args)
      (let [results (apply f args)]
        (tap> {:args args
               :results results
               :var v
               :var-name (:name (meta v))})
        results))))


;; Need to copy meta information
(defn instrument-var [v]
  (let [old-val @v
        instrumented-f (stats-handling v old-val)]
    (when (and (not (contains? @instrumented-vars v))
               (fn? old-val))
      (alter-var-root v (constantly instrumented-f))
      (swap! instrumented-vars assoc v {:old old-val :new instrumented-f}))))

(defn unstrument-var [v]
  (when-let [val (get-in @instrumented-vars [v :old])]
    (alter-var-root v (constantly val))
    (swap! instrumented-vars dissoc v)))


(defn instrument-ns [namespace-symbol]
  (doseq [[_ var] (ns-publics (find-ns namespace-symbol))]
    (instrument-var var)))

(defn unstrument-ns [namespace-symbol]
  (doseq [[_ var] (ns-publics (find-ns namespace-symbol))]
    (unstrument-var var)))


(defn view-code* [{:keys [var-value var-name ns-name]}]
  [:code.syntax
   [:pre
    (glow.html/hiccup-transform
     (parse/parse
      (let [val @var-value
            type-name (.getName (type val))]
        ;; There is no atom pred??
        (cond (= type-name "clojure.lang.Atom")
              (pprint-str @val)

              (or (number? val) (boolean? val) (string? val))
              (str val)

              (and (string/includes? type-name "$")
                   (fn? @var-value))
              (try (source-fn (symbol (name ns-name) (name var-name)))
                   (catch Exception e
                     (try
                       (clojure.repl/source-fn (symbol (name ns-name) (name var-name)))
                       (catch Exception e nil))))

              (instance? clojure.lang.IObj @var-value)
              (pprint-str @var-value)

              :else (pprint-str (keys (bean @var-value)))))))]])

;; This parser I am using is very slow and makes my views very
;; non-performant. But, a naive memoize gets rid of some of the nice
;; things we get like watching atoms update.
#_(def view-code (memoize view-code*))
(def view-code view-code*)

(defn code->hiccup [code]
  [:code.syntax
   [:pre
    (glow.html/hiccup-transform
     (parse/parse (pprint-str code)))]])



;; This is really ugly right now, need to make prettier.
(defn view-single-var [{:keys [var-name var-calls ns-name only-distinct-calls]}]
  (let [namespace (and ns-name (find-ns (symbol ns-name)))
        var-value (get (ns-publics namespace) var-name)
        var-calls (update var-calls :values (if only-distinct-calls distinct identity))]
    [:div
     [:a {:onclick [:back]} "back"]
     [:h2 (str ns-name "/" (name var-name))]
     (view-code {:var-name var-name :var-value var-value :ns-name ns-name})
     [:p "Distinct" [:input {:type "checkbox"
                             :onchange [:only-distinct-calls]
                             :checked only-distinct-calls}]]
     (for [[i {:keys [args results]}] (map-indexed vector (:values var-calls))]
       [:div
        i
        (code->hiccup args)
        (code->hiccup results)])]))



(defn view [{:keys [calls
                    current-hover
                    current-click
                    namespace-input
                    highlight-var
                    only-distinct-calls
                    filter-called
                    ns-name] :as state}]
  [:body {:style {:background-color "#363638"
                  :color "white"}}
   [:style {:type "text/css"}
    (glow.core/generate-css)]
   [:div
    [:h1 "Code View"]
    (if highlight-var
      (view-single-var {:var-name highlight-var
                        :var-calls (get calls highlight-var)
                        :ns-name ns-name
                        :only-distinct-calls only-distinct-calls})
      [:div
       [:input {:onchange [:namespace-input-change]
                :list "namespaces"
                :style {:width 300}}]
       [:datalist {:id "namespaces"}
        (map (fn [ns]
               [:option {:value ns}])
             (sort (map #(.getName %) (all-ns))))]
       [:button {:onclick [:view]} "view"]
       [:button {:onclick [:inspect]} "inspect"]
       [:p "Only Show Called" [:input {:type "checkbox" :onchange [:filter-called]
                                       :checked filter-called} ]]
       [:div {:style {:display "grid"
                      :grid-template-columns "repeat(3, 1fr)"}}
        (when-let [namespace (and ns-name (find-ns (symbol ns-name)))]
          (for [[var-name var-value]
                (remove (fn [[k v]]
                          (if filter-called
                            (or (not (get-in calls [k :number])) 
                                (zero? (get-in calls [k :number]) ))
                            false))
                        (remove #(string/includes? % "$")
                                (ns-publics namespace)))]
            [:div {:style {:margin 20
                           :background-color "#002b36"
                           :padding 10
                           :max-width "28vw"
                           :position "relative"}}
             [:ul {:style {:position "absolute"
                           :left -30
                           :font-size 30
                           :top -20
                           :line-height 20}}
              (for [i (range (min 10 (get-in calls [var-name :number] 0)))]
                (let [hovered? (or (= current-hover [var-name i])
                                   (= current-click [var-name i]))]
                  [:li {:style {:cursor "pointer"
                                :color (if hovered? "white" "#585858")}
                        :onmouseover [:hover {:i i :var-name var-name}]
                        :onmouseout [:unhover {:i i :var-name var-name}]
                        :onclick [:click {:i i :var-name var-name}]}]))]
             [:div {:style {:overflow "scroll"
                            :max-width "28vw"}}
              [:a {:onclick [:highlight-var {:var-name var-name}]} (name var-name)]
              [:span {:style {:float "right"}}
               (get-in calls [var-name :number] 0)]
              [:div {:style {:borderBottom "2px solid #002b36"
                             :padding-top 10
                             :filter "brightness(80%)"}}]
              (view-code {:var-name var-name :var-value var-value
                          :ns-name ns-name})]
             ;; This hover interface is less than great.
             ;; But it shows that we have this information available.
             (let [shown-value (or current-click current-hover)]
               (if (= (first shown-value) var-name)
                 (let [{:keys [args results]} (get-in calls [var-name :values (- (get-in calls [var-name :number] 0)
                                                                                 (second shown-value)
                                                                                 1)])]
                   [:div {:style {:height 50
                                  :overflow-y "hidden"
                                  :overflow-x "scroll"}}
                    [:code.syntax
                     [:pre  (glow.html/hiccup-transform
                             (parse/parse
                              (str "(" (pr-str var-name) " " (string/join " " (map pr-str args)) ") ;;=> " (pr-str results))))]]])
                 [:div {:style {:height 50}}]))]))]])]])



(defn event-handler [{:keys [action]}]
  #_(println action)
  (let [[action-type payload] action]
    (case action-type
      :hover (swap! state assoc
                    :current-hover [(:var-name payload) (:i payload)]
                    :current-click nil)
      :unhover (swap! state dissoc :current-hover)
      :click (swap! state assoc :current-click [(:var-name payload) (:i payload)])
      :namespace-input-change (swap! state assoc :namespace-input (:value payload))
      :view (do
              (when (and (:ns-name @state)
                         (not= (:ns-name @state) (:namespace-input @state)))
                (unstrument-ns (symbol (:ns-name @state))))
              (swap! state assoc :ns-name (:namespace-input @state)))
      ;; Should make this a toggle
      :inspect (instrument-ns (symbol (:ns-name @state)))
      :highlight-var (swap! state assoc :highlight-var (:var-name payload))
      :back (swap! state dissoc :highlight-var)
      :only-distinct-calls (swap! state update :only-distinct-calls #(not %))
      :filter-called (swap! state update :filter-called #(not %))
      (println "unhandled action" action))))


(defonce live-view-server
  (binding [live-view/*live-view-context* {:app :code-view}]
    (live-view/start-live-view-server
     {:state state
      :view #'view
      :event-handler #'event-handler
      :port 1116
      :skip-frames-allowed? true})))



(comment
  (.stop live-view-server))





 
