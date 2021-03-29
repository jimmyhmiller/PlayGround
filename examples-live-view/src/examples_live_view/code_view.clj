(ns example-live-view.code-view
  (:require [live-view-server.core :as live-view]
            [clojure.repl]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.pprint :as pprint]
            [glow.html]
            [glow.parse]
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



;; Need to pull in source
;; Need to instrument things
;; Need to make a real UI

(defn pprint-str [coll]
  (let [out (java.io.StringWriter.)]
    (pprint/pprint coll out)
   (str out)))



(defn view [_]
  [:body {:style {:background-color "#363638"
                  :color "white"}}
   [:style {:type "text/css"}
    (glow.core/generate-css)]
   [:h1 "Code View"]
   [:div {:style {:display "grid"
                  :grid-template-columns "repeat(3, 1fr)"}}
    (for [[var-name var-value]
          (remove #(string/includes? % "$")
                   (ns-publics *ns*))]
      [:div {:style {:margin 20
                     :background-color "#002b36"
                     :padding 10}}
       (name var-name)
       [:div {:style {:borderBottom "2px solid #002b36"
                      :padding-top 10
                      :filter "brightness(80%)"}}]
       [:code.syntax
        [:pre
         (glow.html/hiccup-transform
          (glow.parse/parse
           (let [type-name (.getName (type (deref var-value)))]
             (cond (= type-name "clojure.lang.Atom")
                   (pprint-str @@var-value)

                   (and (string/includes? type-name "$")
                        (fn? @var-value))
                   (source-fn var-name)

                   (instance? clojure.lang.IObj @var-value)
                   (pprint-str @var-value)

                   :else  (pprint-str (keys (bean @var-value)))))))]]])]])



(defonce state (atom {}))

(defn event-handler [action]
  (println action))


(defonce live-view-server
  (live-view/start-live-view-server
   {:state state
    :view #'view
    :event-handler #'event-handler
    :port 23456}))

