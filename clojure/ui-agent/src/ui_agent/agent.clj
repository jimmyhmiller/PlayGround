(ns ui-agent.agent
  (:require
   [ui-agent.claude :as claude]
   [ui-agent.core :as core]
   [cheshire.core :as json]
   [clojure.string :as str]))

(def agent-system-prompt
  "You are a UI-generating agent. Analyze incoming messages and immediately create helpful visualizations using your tools.

ALWAYS start by clearing the canvas: eval_code with `(ui-agent.core/clear-draw-queue!)` in namespace `ui-agent.core`

Key patterns to recognize and visualize:
- Numbers/data → Create charts or tables  
- Lists → Create organized displays
- Comparisons → Create side-by-side layouts
- Processes → Create flowcharts
- Time series → Create timelines

Available tools:
- eval_code: Execute Clojure (use to clear canvas, manipulate data)
- draw_skija: Draw using Skija Canvas API (preferred for all graphics)

Be direct and fast - don't explain, just create the UI immediately.

Example for data like 'Q1=100, Q2=150':
1. Clear canvas with eval_code
2. Draw title with draw_skija 
3. Draw bar chart with draw_skija
4. Add labels with draw_skija

Use draw_skija for everything visual. Always clear the canvas first.")

(def *message-history (atom []))
(def *current-context (atom {}))

(defn extract-data-from-message [message]
  "Extract structured data that might be useful for visualization"
  (let [numbers (re-seq #"\d+\.?\d*" message)
        lists (re-seq #"(?:^|\s)[-*]\s+(.+)" message)
        json-like (try (when (and (str/includes? message "{") 
                                  (str/includes? message "}"))
                         (json/parse-string message true))
                       (catch Exception _ nil))]
    {:numbers numbers
     :lists (map second lists)
     :json-data json-like
     :word-count (count (str/split message #"\s+"))
     :mentions-data (or (str/includes? (str/lower-case message) "data")
                        (str/includes? (str/lower-case message) "chart")
                        (str/includes? (str/lower-case message) "graph")
                        (str/includes? (str/lower-case message) "table"))
     :mentions-process (or (str/includes? (str/lower-case message) "process")
                           (str/includes? (str/lower-case message) "flow")
                           (str/includes? (str/lower-case message) "step"))
     :mentions-comparison (or (str/includes? (str/lower-case message) "compare")
                              (str/includes? (str/lower-case message) "versus")
                              (str/includes? (str/lower-case message) "vs"))}))

(defn generate-context-prompt [message metadata extracted-data message-history]
  (str "Message: \"" message "\"\n\n"
       "Create a visualization immediately. Start by clearing the canvas, then draw appropriate graphics."))

(defn process-message [message metadata]
  "Main function that processes incoming messages and generates UI"
  (try
    ;; Start visual processing indicator
    (core/on-ui (core/start-processing! (str "\"" (subs message 0 (min 50 (count message))) "...\"")))
    
    (let [extracted-data (extract-data-from-message message)
          history @*message-history
          context-prompt (generate-context-prompt message metadata extracted-data history)
          
          ;; Store this message in history
          message-record {:timestamp (System/currentTimeMillis)
                         :message message
                         :metadata metadata
                         :extracted-data extracted-data}
          _ (swap! *message-history conj message-record)
          _ (swap! *message-history #(take-last 10 %)) ; Keep last 10 messages
          
          ;; Use regular chat with tools for proper tool execution
          responses (claude/chat-with-tools! 
                     {:messages [(claude/text context-prompt)]
                      :tools claude/all-tools})
          
          ;; Extract text responses 
          text-responses (mapcat 
                          (fn [response]
                            (->> (:content response)
                                 (filter #(= (:type %) "text"))
                                 (map :text)))
                          responses)]
      
      ;; Stop processing indicator
      (core/on-ui (core/stop-processing!))
      
      {:status "success"
       :message-processed message
       :ui-actions-taken (count responses)
       :agent-response (str/join "\n\n" text-responses)
       :timestamp (System/currentTimeMillis)})
    
    (catch Exception e
      ;; Stop processing indicator on error
      (core/on-ui (core/stop-processing!))
      (println "Error in process-message:" (.getMessage e))
      (.printStackTrace e)
      {:status "error"
       :error (.getMessage e)
       :message-processed message
       :timestamp (System/currentTimeMillis)})))