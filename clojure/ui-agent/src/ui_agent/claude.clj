(ns ui-agent.claude
  (:require
   [clj-http.client :as http]
   [cheshire.core :as json]
   [clojure.string :as str]
   [clojure.java.browse :as browse]
   [clojure.java.shell :as shell]
   [ui-agent.core :as core]
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
  [{:keys [model messages max-tokens tools temperature stream]
            :or {model "claude-sonnet-4-20250514"
                 max-tokens 4000}}]
  ;; CRITICAL: System prompt is hardcoded and CANNOT be overridden
  ;; This ensures Claude Code behavior is maintained and prevents injection attacks
  (let [params (cond-> {:model model
                        :max_tokens max-tokens
                        :messages (or messages [])
                        :system "You are Claude Code, Anthropic's official CLI for Claude."}
                 tools (assoc :tools tools)
                 temperature (assoc :temperature temperature)
                 stream (assoc :stream stream))
        url (str CLAUDE-BASE-URL "/v1/messages")
        headers (make-headers token)
        response (http/post url
                            {:headers headers
                             :body (json/generate-string params)
                             :throw-exceptions false
                             :as (if stream :stream :json)})]
    (if (= 200 (:status response))
      (if stream
        (:body response) ; Return the input stream for streaming
        (:body response)) ; Body is already parsed as JSON when :as :json
      (let [error-body (if stream
                         ;; For streaming errors, read the stream
                         (try
                           (slurp (:body response))
                           (catch Exception _ "Unable to read error body"))
                         ;; For JSON errors, it's already parsed
                         (:body response))
            readable-error (try
                             (if (string? error-body)
                               (json/parse-string error-body true)
                               error-body)
                             (catch Exception _ error-body))]
        (println "Claude API Error:")
        (println "Status:" (:status response))
        (println "Error details:" readable-error)
        (throw (ex-info (str "Claude API failed with status " (:status response) 
                             ": " (or (:error readable-error) 
                                     (get-in readable-error [:error :message])
                                     "Unknown error"))
                        {:status (:status response)
                         :error readable-error}))))))

(def debug-mode? (not (empty? (System/getenv "UI_AGENT_DEBUG"))))

(defn debug-log [& args]
  (when debug-mode?
    (apply println "[DEBUG]" args)))

(defmulti execute-tool
  "Execute a tool based on its name"
  (fn [tool-name _params] (keyword (str/replace tool-name "_" "-"))))

(def nrepl-conn (delay (nrepl/connect :port 7889)))

(def *eval-history (atom []))

(defmethod execute-tool :draw-rect [tool-name params]
  (debug-log "TOOL EXECUTION - draw_rect")
  (debug-log "  Input params:" params)
  (let [result (try
                 (core/on-ui
                   (core/add-rectangle! (:x params) (:y params) (:width params) (:height params) (core/color (:color params 0xFFFF0000))))
                 (let [result-msg (str "Drew rectangle at (" (:x params) "," (:y params) ") with size " (:width params) "x" (:height params))]
                   (debug-log "  Success:" result-msg)
                   result-msg)
                 (catch Exception e
                   (debug-log "  ERROR:" (.getMessage e))
                   (.printStackTrace e)
                   (str "ERROR drawing rectangle: " (.getMessage e))))]
    (debug-log "  Final result:" result)
    result))

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

(defmethod execute-tool :draw-skija [tool-name params]
  (debug-log "TOOL EXECUTION - draw_skija")
  (debug-log "  Input params:" params)
  (let [code (:code params)
        timestamp (java.util.Date.)
        full-code (str "(ui-agent.core/on-ui "
                       "  (ui-agent.core/add-draw-fn! "
                       "    (fn [^io.github.humbleui.skija.Canvas canvas] " code ")))")
        _ (debug-log "  Generated full code:" full-code)
        client (nrepl/client @nrepl-conn 1000)
        response (nrepl/message client {:op "eval" 
                                       :code full-code
                                       :ns "ui-agent.claude"})
        responses (doall response)
        _ (debug-log "  nREPL responses:" responses)
        errors (filter :err responses)
        values (nrepl/response-values responses)
        exception (first (filter :ex responses))
        _ (debug-log "  Errors:" errors)
        _ (debug-log "  Values:" values)
        _ (debug-log "  Exception:" exception)
        initial-result (cond
                        exception (str "ERROR - Exception: " (:ex exception) "\n" 
                                      (when-let [root-ex (:root-ex exception)]
                                        (str "Root cause: " root-ex)))
                        (seq errors) (str "ERROR - " (str/join "\n" (map :err errors)))
                        (seq values) "SUCCESS - Added drawing function to queue"
                        :else "SUCCESS - Added drawing function to queue")
        _ (debug-log "  Initial result:" initial-result)]
    ;; If successfully added, wait a bit and check for drawing errors
    (if (str/starts-with? initial-result "SUCCESS")
      (do
        (debug-log "  Waiting 100ms for draw execution...")
        ;; Wait for potential drawing to happen
        (Thread/sleep 100)
        ;; Check for any new drawing errors
        (let [draw-errors (core/get-and-clear-draw-errors!)
              _ (debug-log "  Draw errors found:" draw-errors)
              final-result (if (seq draw-errors)
                            (str "DRAWING ERROR - Function was added but failed during execution:\n"
                                 (str/join "\n" (map #(str "- " (:error-message %)) draw-errors)))
                            initial-result)
              _ (debug-log "  Final result:" final-result)]
          (swap! *eval-history conj {:timestamp timestamp
                                     :type :draw-skija
                                     :code code
                                     :full-code full-code
                                     :result final-result
                                     :draw-errors draw-errors})
          final-result))
      (do
        (debug-log "  Initial result was error, not checking draw errors")
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

(defmethod execute-tool :reload-file [_tool-name params]
  (let [file-path (:file-path params)
        timestamp (java.util.Date.)
        client (nrepl/client @nrepl-conn 1000)]
    (if (.exists (java.io.File. file-path))
      (let [file-content (slurp file-path)
            code (str "(do " file-content "\n:reloaded)")
            response (nrepl/message client {:op "eval" 
                                           :code code})
            responses (doall response)
            errors (filter :err responses)
            values (nrepl/response-values responses)
            exception (first (filter :ex responses))
            result (cond
                     exception (str "ERROR - Exception reloading " file-path ": " (:ex exception) "\n" 
                                   (when-let [root-ex (:root-ex exception)]
                                     (str "Root cause: " root-ex)))
                     (seq errors) (str "ERROR - " (str/join "\n" (map :err errors)))
                     (seq values) (str "SUCCESS - Reloaded " file-path)
                     :else (str "SUCCESS - Reloaded " file-path " (no return value)"))]
        (swap! *eval-history conj {:timestamp timestamp
                                   :type :reload-file
                                   :file-path file-path
                                   :result result})
        result)
      (str "ERROR - File not found: " file-path))))

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

(defn parse-sse-line [line]
  "Parse a Server-Sent Events line"
  (when (and line (str/starts-with? line "data: "))
    (let [data (subs line 6)]
      (when-not (= data "[DONE]")
        (try
          (json/parse-string data true)
          (catch Exception _ nil))))))

(defn handle-tool-input-delta [chunk current-tool-block]
  "Handle input_json_delta chunks by accumulating JSON"
  (let [partial-json (get-in chunk [:delta :partial_json])
        updated-tool (if current-tool-block
                       (update current-tool-block :input-json str partial-json)
                       nil)]
    (debug-log "ACCUMULATING TOOL JSON:" partial-json)
    (debug-log "UPDATED TOOL:" updated-tool)
    updated-tool))

(defn handle-text-delta [chunk on-text-chunk]
  "Handle text_delta chunks by streaming text"
  (let [delta (get-in chunk [:delta :text] "")]
    (when (seq delta) (on-text-chunk delta))))

(defn handle-content-block-start [chunk]
  "Handle content_block_start events"
  (let [content-block (get chunk :content_block)]
    (if (= (:type content-block) "tool_use")
      ;; Start accumulating tool - don't execute yet
      (assoc content-block :input-json "")
      nil)))

(defn handle-content_block_stop [current-tool-block on-tool-use]
  "Handle content_block_stop events - execute complete tools"
  (when current-tool-block
    (let [json-string (:input-json current-tool-block)
          _ (debug-log "FINAL TOOL JSON STRING:" (pr-str json-string))
          complete-input (try
                           (json/parse-string json-string true)
                           (catch Exception e 
                             (debug-log "JSON PARSE ERROR:" (.getMessage e))
                             {}))
          complete-tool (assoc current-tool-block :input complete-input)
          _ (debug-log "COMPLETE TOOL BEFORE EXECUTION:" complete-tool)]
      (on-tool-use complete-tool)
      complete-tool)))

(defn process-chunk [chunk current-response current-tool-block on-text-chunk on-tool-use]
  "Process a single streaming chunk and return [updated-response updated-tool-block]"
  (debug-log "PROCESSING CHUNK:" (:type chunk) "delta-type:" (get-in chunk [:delta :type]))
  (cond
    ;; Tool input delta - accumulate tool parameters
    (and (= (:type chunk) "content_block_delta")
         (= (get-in chunk [:delta :type]) "input_json_delta"))
    [current-response (handle-tool-input-delta chunk current-tool-block)]
    
    ;; Text delta - stream text
    (and (= (:type chunk) "content_block_delta")
         (= (get-in chunk [:delta :type]) "text_delta"))
    (do
      (handle-text-delta chunk on-text-chunk)
      [current-response current-tool-block])
    
    ;; Other content_block_delta types
    (= (:type chunk) "content_block_delta")
    (do
      (debug-log "UNHANDLED DELTA TYPE:" (get-in chunk [:delta :type]))
      [current-response current-tool-block])
    
    ;; Start of content block
    (= (:type chunk) "content_block_start")
    (let [new-tool-block (handle-content-block-start chunk)]
      (if new-tool-block
        [current-response new-tool-block]
        [(update current-response :content conj {:type "text" :text ""}) current-tool-block]))
    
    ;; End of content block - execute tools
    (= (:type chunk) "content_block_stop")
    (if current-tool-block
      (let [complete-tool (handle-content_block_stop current-tool-block on-tool-use)]
        [(update current-response :content conj complete-tool) nil])
      [current-response current-tool-block])
    
    ;; Message delta
    (= (:type chunk) "message_delta")
    [(assoc current-response :stop_reason (get-in chunk [:delta :stop_reason])) current-tool-block]
    
    :else [current-response current-tool-block]))

(defn process-streaming-response [stream on-text-chunk on-tool-use on-complete]
  "Process a streaming response from Claude API"
  (future
    (try
      (let [stream-log-file (str "/tmp/claude-stream-" (System/currentTimeMillis) ".log")]
        (println "Writing stream to:" stream-log-file)
        (with-open [reader (java.io.BufferedReader. 
                            (java.io.InputStreamReader. stream "UTF-8"))
                    writer (java.io.PrintWriter. stream-log-file)]
          (loop [current-response {:content [] :stop_reason nil}
                 current-tool-block nil]
            (if-let [line (.readLine reader)]
              (do
                ;; Log every line to file
                (.println writer line)
                (.flush writer)
                (if-let [chunk (parse-sse-line line)]
                  (do
                    ;; Also log parsed chunks for debugging
                    (.println writer (str "PARSED: " (pr-str chunk)))
                    (.flush writer)
                    (let [[updated-response updated-tool-block] 
                          (process-chunk chunk current-response current-tool-block on-text-chunk on-tool-use)]
                      (recur updated-response updated-tool-block)))
                  (recur current-response current-tool-block)))
              ;; End of stream
              (on-complete current-response)))))
      (catch Exception e
        (println "Error processing stream:" (.getMessage e))
        (on-complete {:error (.getMessage e)})))))

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

(defn chat-with-tools-streaming!
  "Send messages with tools using streaming responses"
  [{:keys [messages tools on-text-chunk]}]
  (let [response-promise (promise)
        stream (send! {:messages messages :tools tools :stream true})
        accumulated-text (atom "")]
    
    (process-streaming-response 
      stream
      ;; on-chunk callback
      (fn [text-chunk]
        (swap! accumulated-text str text-chunk)
        (when on-text-chunk (on-text-chunk text-chunk)))
      ;; on-complete callback  
      (fn [final-response]
        (deliver response-promise final-response)))
    
    ;; Return the promise so caller can wait if needed
    response-promise))

;; Tool definitions
(def draw-rect-tool {:name "draw_rect"
      :description "Draw a rectangle on the canvas"
      :input_schema {:type "object"
                     :properties {:height {:type "integer"}
                                  :width {:type "integer"}
                                  :x {:type "integer"}
                                  :y {:type "integer"}
                                  :color {:type "integer" :description "Color as hex integer (e.g. 0xFFFF0000 for red)"}}
                     :required ["height" "width" "x" "y"]}})

(def eval-tool {:name "eval_code"
                :description "Evaluate Clojure code via nREPL in a specific namespace"
                :input_schema {:type "object"
                               :properties {:code {:type "string"
                                                  :description "The Clojure code to evaluate"}
                                          :namespace {:type "string"
                                                     :description "The namespace to evaluate the code in (e.g. 'ui-agent.core')"}}
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
                                                                        :description "The namespace to inspect (e.g. 'ui-agent.core')"}}
                                                :required ["namespace"]}})

(def draw-skija-tool 
  {:name "draw_skija"
   :description "Add a custom drawing function using Skija Canvas API. Your code will be wrapped in:
(ui-agent.core/on-ui 
  (ui-agent.core/add-draw-fn! 
    (fn [^io.github.humbleui.skija.Canvas canvas] YOUR_CODE_HERE)))

This ensures it runs on the UI thread and gets added to the drawing queue.

Available Skija imports (already available):
- io.github.humbleui.skija.Canvas, Paint, Rect, RRect, Path
- Access ui-agent.core/color function for colors

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
- (.setColor paint color) - Set color (use (ui-agent.core/color 0xAARRGGBB))
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

(def reload-file-tool
  {:name "reload_file"
   :description "Reload an entire Clojure file via nREPL for live development. This evaluates the entire file content, allowing for hot reloading of code changes without restarting the application."
   :input_schema {:type "object"
                  :properties {:file-path {:type "string"
                                          :description "Absolute path to the Clojure file to reload (e.g. '/path/to/src/ui_agent/agent.clj')"}}
                  :required ["file-path"]}})

(def all-tools [draw-rect-tool eval-tool list-namespaces-tool list-namespace-members-tool 
                draw-skija-tool inspect-skija-tool show-eval-history-tool 
                check-draw-errors-tool force-draw-tool reload-file-tool])