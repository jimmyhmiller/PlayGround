(ns ui-experiment.claude
  (:require
   [clj-http.client :as http]
   [cheshire.core :as json]
   [clojure.string :as str]
   [clojure.java.browse :as browse]
   [clojure.java.shell :as shell]
   [ui-experiment.core :as core]
   [nrepl.core :as nrepl])
  (:import
   [java.net URLEncoder]
   [java.util Base64]
   [java.security MessageDigest SecureRandom]
   [java.net ServerSocket]
   [java.io BufferedReader InputStreamReader PrintWriter]
   [java.lang ProcessBuilder]))

;; Constants for Claude Code OAuth
(def ^:const CLAUDE-CLIENT-ID "9d1c250a-e61b-44d9-88ed-5944d1962f5e")
(def ^:const CLAUDE-BASE-URL "https://api.anthropic.com")
(def ^:const AUTH-URL "https://login.anthropic.com/oauth/authorize")
(def ^:const TOKEN-URL "https://login.anthropic.com/oauth/token")
(def ^:const API-VERSION "2023-06-01")
(def ^:const BETA-VERSION "oauth-2025-04-20")
(def ^:const USER-AGENT "Claude Code/1.0")



(defn load-zsh-env []
  (let [cmd ["zsh" "-i" "-c" "export"]
        pb (ProcessBuilder. cmd)
        process (.start pb)
        output (slurp (.getInputStream process))
        lines (str/split-lines output)]
    (doseq [line lines]
      (when-let [[_ k v] (re-matches #"^([^=]+)=(.*)$" line)]
        (System/setProperty k v)))))

(load-zsh-env)
(defn get-token
  "Gets authentication token from environment or OAuth flow"
  []
  (if-let [env-token (System/getProperty "CLAUDE_CODE_OAUTH_TOKEN")]
    env-token
    (throw (ex-info "need auth" {}))))


(def token (get-token))

;; Claude API client
(defn make-headers [token]
  {"authorization" (str "Bearer " token)
   "anthropic-beta" BETA-VERSION
   "anthropic-version" API-VERSION
   "content-type" "application/json"})



(defn send!
  "Creates a message using Claude API with smart defaults"
  [{:keys [model messages max-tokens system tools temperature]
            :or {model "claude-sonnet-4-20250514"
                 max-tokens 1000
                 system "You are Claude Code, Anthropic's official CLI for Claude."}}]
  (let [params (cond-> {:model model
                        :max_tokens max-tokens
                        :messages (or messages [])
                        :system system}
                 tools (assoc :tools tools)
                 temperature (assoc :temperature temperature))
        url (str CLAUDE-BASE-URL "/v1/messages")
        headers (make-headers token)
        response (http/post url
                            {:headers headers
                             :body (json/generate-string params)
                             :throw-exceptions false})]
    (if (= 200 (:status response))
      (json/parse-string (:body response) true)
      (throw (ex-info "API request failed"
                      {:status (:status response)
                       :body (:body response)})))))



(defmulti execute-tool
  "Execute a;;;;;;;; tool based on its name"
  (fn [tool-name _params] (keyword (str/replace tool-name "_" "-"))))

(def nrepl-conn (delay (nrepl/connect :port 7888)))

(def *eval-history (atom []))

(defmethod execute-tool :draw-rect [_tool-name params]
  (core/on-ui
    (core/add-rectangle! (:x params) (:y params) (:width params) (:height params) (core/color 0xFFFF0000)))
  (str "Drew red rectangle at (" (:x params) "," (:y params) ") with size " (:width params) "x" (:height params)))

(defmethod execute-tool :eval-code [_tool-name params]
  (let [code (:code params)
        ns-name (:namespace params)
        timestamp (java.util.Date.)
        client (nrepl/client @nrepl-conn 1000)
        response (nrepl/message client {:op "eval" 
                                       :code code
                                       :ns ns-name})
        responses (doall response)
        errors (filter :err responses)
        values (nrepl/response-values responses)
        exception (first (filter :ex responses))
        result (cond
                 exception (str "ERROR - Exception: " (:ex exception) "\n" 
                               (when-let [root-ex (:root-ex exception)]
                                 (str "Root cause: " root-ex)))
                 (seq errors) (str "ERROR - " (str/join "\n" (map :err errors)))
                 (seq values) (str "SUCCESS - Result: " (str/join "\n" values))
                 :else "SUCCESS - Result: nil")]
    (swap! *eval-history conj {:timestamp timestamp
                               :type :eval-code
                               :namespace ns-name
                               :code code
                               :result result})
    result))

(defmethod execute-tool :list-namespaces [_tool-name _params]
  (let [client (nrepl/client @nrepl-conn 1000)
        response (nrepl/message client {:op "eval" 
                                       :code "(map str (all-ns))"})
        results (nrepl/response-values response)]
    (if (seq results)
      (first results)
      "[]")))

(defmethod execute-tool :list-namespace-members [_tool-name params]
  (let [ns-name (:namespace params)
        client (nrepl/client @nrepl-conn 1000)
        code (str "(map (fn [[k v]] "
                  "  {:name (str k) "
                  "   :type (cond "
                  "           (fn? @v) \"function\" "
                  "           (instance? clojure.lang.Atom @v) \"atom\" "
                  "           (instance? clojure.lang.Ref @v) \"ref\" "
                  "           (instance? clojure.lang.Agent @v) \"agent\" "
                  "           :else \"var\") "
                  "   :value (if (fn? @v) \"<function>\" (pr-str @v))}) "
                  "  (ns-publics '" ns-name "))")
        response (nrepl/message client {:op "eval" :code code})
        results (nrepl/response-values response)]
    (if (seq results)
      (first results)
      "[]")))

(defmethod execute-tool :draw-skija [_tool-name params]
  (let [code (:code params)
        timestamp (java.util.Date.)
        full-code (str "(ui-experiment.core/on-ui "
                       "  (ui-experiment.core/add-draw-fn! "
                       "    (fn [^io.github.humbleui.skija.Canvas canvas] " code ")))")
        client (nrepl/client @nrepl-conn 1000)
        response (nrepl/message client {:op "eval" 
                                       :code full-code
                                       :ns "ui-experiment.claude"})
        responses (doall response)
        errors (filter :err responses)
        values (nrepl/response-values responses)
        exception (first (filter :ex responses))
        initial-result (cond
                        exception (str "ERROR - Exception: " (:ex exception) "\n" 
                                      (when-let [root-ex (:root-ex exception)]
                                        (str "Root cause: " root-ex)))
                        (seq errors) (str "ERROR - " (str/join "\n" (map :err errors)))
                        (seq values) "SUCCESS - Added drawing function to queue"
                        :else "SUCCESS - Added drawing function to queue")]
    ;; If successfully added, wait a bit and check for drawing errors
    (if (str/starts-with? initial-result "SUCCESS")
      (do
        ;; Wait for potential drawing to happen
        (Thread/sleep 100)
        ;; Check for any new drawing errors
        (let [draw-errors (core/get-and-clear-draw-errors!)
              final-result (if (seq draw-errors)
                            (str "DRAWING ERROR - Function was added but failed during execution:\n"
                                 (str/join "\n" (map #(str "- " (:error-message %)) draw-errors)))
                            initial-result)]
          (swap! *eval-history conj {:timestamp timestamp
                                     :type :draw-skija
                                     :code code
                                     :full-code full-code
                                     :result final-result
                                     :draw-errors draw-errors})
          final-result))
      (do
        (swap! *eval-history conj {:timestamp timestamp
                                   :type :draw-skija
                                   :code code
                                   :full-code full-code
                                   :result initial-result})
        initial-result))))

(defmethod execute-tool :inspect-skija-classes [_tool-name params]
  (let [class-name (:class params)
        client (nrepl/client @nrepl-conn 1000)
        code (str "(let [cls (Class/forName \"" class-name "\")] "
                  "  {:methods (map #(str (.getName %) "
                  "                       \" (\" "
                  "                       (clojure.string/join \", \" (map str (.getParameterTypes %))) "
                  "                       \") -> \" "
                  "                       (.getReturnType %)) "
                  "                 (.getMethods cls))})")
        response (nrepl/message client {:op "eval" :code code})
        results (nrepl/response-values response)]
    (if (seq results)
      (first results)
      "Class not found")))

(defmethod execute-tool :show-eval-history [_tool-name params]
  (let [limit (or (:limit params) 10)
        recent-history (take-last limit @*eval-history)]
    (if (empty? recent-history)
      "No evaluation history found"
      (json/generate-string 
        {:total-count (count @*eval-history)
         :showing (count recent-history)
         :history recent-history}
        {:pretty true}))))

(defmethod execute-tool :check-draw-errors [_tool-name _params]
  (let [errors (core/get-and-clear-draw-errors!)]
    (if (empty? errors)
      "No drawing errors found"
      (json/generate-string 
        {:error-count (count errors)
         :errors errors}
        {:pretty true}))))

(defmethod execute-tool :force-draw [_tool-name _params]
  (core/force-draw!)
  "Triggered a draw cycle")

(defmethod execute-tool :default [tool-name _params]
  (str "Unknown tool: " tool-name))

;; Tool result processing
(defn process-tool-use
  "Process a single tool use content block and return tool result"
  [tool-use]
  (let [{:keys [name input id]} tool-use
        result (execute-tool name input)]
    {:type "tool_result"
     :tool_use_id id
     :content (if (string? result)
                result
                (json/generate-string result))}))

(defn process-tool-uses
  "Process all tool uses in a message content and return tool results"
  [content]
  (->> content
       (filter #(= (:type %) "tool_use"))
       (map process-tool-use)))

(defn create-tool-result-message
  "Create an assistant message with tool results"
  [tool-results]
  {:role "user"
   :content tool-results})

(defn process-response
  "Process Claude's response and execute any tool uses, returning messages to continue conversation"
  [response]
  (let [content (get-in response [:content])
        tool-uses (filter #(= (:type %) "tool_use") content)]
    (if (empty? tool-uses)
      [] ; No tools to execute
      (let [tool-results (process-tool-uses content)]
        [(create-tool-result-message tool-results)]))))

;; Message helpers for threading
(defn text [content]
  "Create a user text message"
  {:role "user"
   :content [{:type "text"
              :text content}]})

(defn with-tools [tools]
  "Add tools to the conversation"
  (fn [messages]
    {:messages messages
     :tools tools}))

(defn chat!
  "Send messages and get response - designed for threading"
  [params-or-messages]
  (if (vector? params-or-messages)
    (send! {:messages params-or-messages})
    (send! params-or-messages)))

(defn chat-with-tools!
  "Send messages with tools and automatically process tool uses"
  [{:keys [messages tools]}]
  (loop [current-messages messages
         all-responses []]
    (let [response (send! {:messages current-messages :tools tools})
          _ (println "\n=== Assistant Response ===")
          content (:content response)
          _ (doseq [item content]
              (case (:type item)
                "text" (println "Text:" (:text item))
                "tool_use" (do 
                            (println "\nTool Call:" (:name item))
                            (println "  ID:" (:id item))
                            (println "  Input:" (json/generate-string (:input item) {:pretty true})))
                (println "Unknown content type:" item)))
          new-responses (conj all-responses response)
          tool-result-messages (process-response response)
          assistant-message {:role "assistant"
                            :content (:content response)}]
      (when (seq tool-result-messages)
        (println "\n=== Tool Execution Results ===")
        (doseq [msg tool-result-messages]
          (doseq [content-item (:content msg)]
            (println "Tool Result ID:" (:tool_use_id content-item))
            (println "  Result:" (:content content-item)))))
      (if (empty? tool-result-messages)
        new-responses
        (recur (concat current-messages [assistant-message] tool-result-messages)
               new-responses)))))

(defn conversation
  "Start a conversation with automatic tool processing"
  [initial-message tools]
  (chat-with-tools! [(text initial-message)] tools))

(defn list-models
  "Fetches available models from the API"
  [token]
  (let [url (str CLAUDE-BASE-URL "/v1/models")
        headers (dissoc (make-headers token) "content-type")
        response (http/get url
                          {:headers headers
                           :throw-exceptions false})]
    (if (= 200 (:status response))
      (let [data (json/parse-string (:body response) true)]
        (:data data))
      (throw (ex-info "Failed to fetch models"
                     {:status (:status response)
                      :body (:body response)})))))


(def rect-tool {:name "draw_rect"
      :description "draw rectangle"
      :input_schema {:type "object"
                     :properties {:height {:type "integer"}
                                  :width {:type "integer"}
                                  :x {:type "integer"}
                                  :y {:type "integer"}}
                     :required ["height" "width" "x" "y"]}})

(def eval-tool {:name "eval_code"
                :description "Evaluate Clojure code via nREPL in a specific namespace"
                :input_schema {:type "object"
                               :properties {:code {:type "string"
                                                  :description "The Clojure code to evaluate"}
                                          :namespace {:type "string"
                                                     :description "The namespace to evaluate the code in (e.g. 'ui-experiment.core')"}}
                               :required ["code" "namespace"]}})

(def list-namespaces-tool {:name "list_namespaces"
                           :description "List all available namespaces in the running Clojure application"
                           :input_schema {:type "object"
                                         :properties {}
                                         :required []}})

(def list-namespace-members-tool {:name "list_namespace_members"
                                  :description "List all public members (functions, atoms, vars) in a specific namespace"
                                  :input_schema {:type "object"
                                                :properties {:namespace {:type "string"
                                                                        :description "The namespace to inspect (e.g. 'ui-experiment.core')"}}
                                                :required ["namespace"]}})

(def draw-skija-tool 
  {:name "draw_skija"
   :description "Add a custom drawing function using Skija Canvas API. Your code will be wrapped in:
(ui-experiment.core/on-ui 
  (ui-experiment.core/add-draw-fn! 
    (fn [^io.github.humbleui.skija.Canvas canvas] YOUR_CODE_HERE)))

This ensures it runs on the UI thread and gets added to the drawing queue.

Available Skija imports (already available):
- io.github.humbleui.skija.Canvas, Paint, Rect, RRect, Path
- Access ui-experiment.core/color function for colors

Common Skija drawing operations:
- (.drawRect canvas rect paint) - Draw a rectangle
- (.drawCircle canvas x y radius paint) - Draw a circle  
- (.drawLine canvas x1 y1 x2 y2 paint) - Draw a line
- (.drawString canvas text x y paint) - Draw text
- (.drawPath canvas path paint) - Draw a path
- (.drawRRect canvas rrect paint) - Draw a rounded rectangle
- (.drawOval canvas rect paint) - Draw an oval
- (.drawArc canvas rect startAngle sweepAngle useCenter paint) - Draw an arc

Paint configuration:
- (io.github.humbleui.skija.Paint.) - Create a new paint
- (.setColor paint color) - Set color (use (ui-experiment.core/color 0xAARRGGBB))
- (.setStroke paint true) - Set stroke style
- (.setStrokeWidth paint width) - Set stroke width
- (.setAntiAlias paint true) - Enable anti-aliasing

Text drawing (Font API):
- (io.github.humbleui.skija.Font.) - Create default font
- (.makeWithSize font size) - Set font size
- (.setTypeface font typeface) - Set typeface
- (.drawString canvas text x y font paint) - Draw text with font and paint
- (.measureTextWidth font text) - Get text width for positioning
- Font and Typeface classes are imported and available

Helper classes:
- (io.github.humbleui.types.Rect/makeXYWH x y width height) - Create a rectangle
- (io.github.humbleui.skija.RRect/makeXYWH x y width height radius) - Create a rounded rectangle
- (io.github.humbleui.skija.Path.) - Create a new path for complex shapes"
   :input_schema {:type "object"
                  :properties {:code {:type "string"
                                     :description "Clojure code to execute inside (fn [canvas] ...). All Skija classes are available with full package names."}}
                  :required ["code"]}})

(def inspect-skija-tool
  {:name "inspect_skija_classes"
   :description "Inspect available methods on Skija classes to understand what drawing operations are possible"
   :input_schema {:type "object"
                  :properties {:class {:type "string"
                                      :description "Full class name to inspect, e.g. 'io.github.humbleui.skija.Canvas' or 'io.github.humbleui.skija.Paint'"}}
                  :required ["class"]}})

(def show-eval-history-tool
  {:name "show_eval_history"
   :description "Show the history of all code that has been evaluated, including eval-code and draw-skija operations"
   :input_schema {:type "object"
                  :properties {:limit {:type "integer"
                                      :description "Number of recent entries to show (default 10)"}}
                  :required []}})

(def check-draw-errors-tool
  {:name "check_draw_errors"
   :description "Check for any drawing errors that occurred during rendering and get detailed error information"
   :input_schema {:type "object"
                  :properties {}
                  :required []}})

(def force-draw-tool
  {:name "force_draw"
   :description "Force a drawing cycle to happen immediately, useful for testing if draw functions work"
   :input_schema {:type "object"
                  :properties {}
                  :required []}})


(def example (atom {}))

(comment
  (chat-with-tools!
   {:messages [(text "draw a rect of any size at any position")]
    :tools [rect-tool]})
  
  (chat-with-tools!
   {:messages [(text "list all available namespaces")]
    :tools [list-namespaces-tool]})
  
  (chat-with-tools!
   {:messages [(text "show me what's in the ui-experiment.core namespace")]
    :tools [list-namespace-members-tool]})
  
  (chat-with-tools!
   {:messages [(text "in the ui-experiment.claude namespace, there is an atom named example, please assoc :success true in it")]
    :tools [eval-tool]})
  
  (chat-with-tools!
   {:messages [(text "find the rectangles atom in ui-experiment.core and clear it, then draw 3 new rectangles")]
    :tools [list-namespace-members-tool eval-tool rect-tool]})
  
  (chat-with-tools!
   {:messages [(text "draw a purple circle at position 200,200 with radius 75")]
    :tools [draw-skija-tool]})
  
  (chat-with-tools!
   {:messages [(text "inspect what methods are available on the Font class")]
    :tools [inspect-skija-tool]})
  
  (chat-with-tools!
   {:messages [(text "clear the draw queue and then draw a smiley face using circles and arcs")]
    :tools [eval-tool draw-skija-tool list-namespace-members-tool inspect-skija-tool]})


  (chat-with-tools!
   {:messages [(text (str "Please draw a little table for this data " (prn-str [{:x 1 :y 2} {:x 3 :y 4}])))]
    :tools [eval-tool draw-skija-tool list-namespace-members-tool inspect-skija-tool]})


  )



