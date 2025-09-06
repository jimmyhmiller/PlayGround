(ns ui-agent.agent
  (:require
   [ui-agent.claude :as claude]
   [ui-agent.core :as core]
   [cheshire.core :as json]
   [clojure.string :as str]))


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
          _ (claude/debug-log "AGENT PROCESSING MESSAGE:" message)
          _ (claude/debug-log "EXTRACTED DATA:" extracted-data) 
          _ (claude/debug-log "CONTEXT PROMPT:" context-prompt)
          
          ;; Store this message in history
          message-record {:timestamp (System/currentTimeMillis)
                         :message message
                         :metadata metadata
                         :extracted-data extracted-data}
          _ (swap! *message-history conj message-record)
          _ (swap! *message-history #(take-last 10 %)) ; Keep last 10 messages
          
          ;; Use streaming for immediate response
          accumulated-text (atom "")
          tool-results (atom [])
          tools-executed (atom 0)
          stream (claude/send! {:messages [(claude/text context-prompt)]
                               :tools claude/all-tools
                               :stream true})
          
          ;; Process streaming response
          _ (claude/process-streaming-response 
              stream
              ;; on-chunk callback for text
              (fn [text-chunk]
                (swap! accumulated-text str text-chunk)
                (print text-chunk)
                (flush)
                (claude/debug-log "STREAMING TEXT CHUNK:" text-chunk))
              ;; on-tool-use callback for complete tool blocks
              (fn [tool-use]
                (println "\nExecuting tool:" (:name tool-use))
                (claude/debug-log "STREAMING TOOL USE:" tool-use)
                (let [tool-result (claude/execute-tool (:name tool-use) (:input tool-use))
                      tool-result-message {:type "tool_result"
                                          :tool_use_id (:id tool-use)
                                          :content (if (string? tool-result)
                                                     tool-result
                                                     (cheshire.core/generate-string tool-result))}]
                  (claude/debug-log "TOOL RESULT:" tool-result)
                  (swap! tool-results conj tool-result-message))
                (swap! tools-executed inc))
              ;; on-complete callback
              (fn [final-response]
                (println "\nStreaming complete")
                (claude/debug-log "STREAMING FINAL RESPONSE:" final-response)
                ;; If we have tool results, send them back to Claude
                (when (seq @tool-results)
                  (println "\nSending tool results back to Claude...")
                  (claude/debug-log "TOOL RESULTS TO SEND:" @tool-results)
                  (try
                    (let [;; Clean up the assistant message content to only include valid API fields
                          clean-content (->> (:content final-response)
                                            (map (fn [item]
                                                  (if (= (:type item) "tool_use")
                                                    ;; Select only valid tool_use fields for the API
                                                    (select-keys item [:type :id :name :input])
                                                    item)))
                                            ;; Filter out empty text blocks
                                            (filter (fn [item]
                                                     (or (not= (:type item) "text")
                                                         (and (= (:type item) "text")
                                                              (not (empty? (str/trim (:text item)))))))))
                          assistant-message {:role "assistant" 
                                           :content clean-content}
                          tool-result-message {:role "user"
                                              :content @tool-results}
                          follow-up-stream (claude/send! {:messages [(claude/text context-prompt)
                                                                    assistant-message
                                                                    tool-result-message]
                                                         :tools claude/all-tools
                                                         :stream true})]
                      ;; Process the follow-up streaming response
                      (claude/process-streaming-response
                        follow-up-stream
                        ;; on-chunk callback for follow-up text
                        (fn [text-chunk]
                          (swap! accumulated-text str text-chunk)
                          (print text-chunk)
                          (flush)
                          (claude/debug-log "FOLLOW-UP TEXT CHUNK:" text-chunk))
                        ;; on-tool-use callback for follow-up tools (recursive)
                        (fn [tool-use]
                          (println "\nExecuting follow-up tool:" (:name tool-use))
                          (claude/debug-log "FOLLOW-UP TOOL USE:" tool-use)
                          (let [tool-result (claude/execute-tool (:name tool-use) (:input tool-use))]
                            (claude/debug-log "FOLLOW-UP TOOL RESULT:" tool-result))
                          (swap! tools-executed inc))
                        ;; on-complete callback for follow-up
                        (fn [follow-up-final-response]
                          (println "\nFollow-up streaming complete")
                          (claude/debug-log "FOLLOW-UP FINAL RESPONSE:" follow-up-final-response))))
                    (catch Exception e
                      (println "Error sending tool results:" (.getMessage e))
                      (claude/debug-log "TOOL RESULT SEND ERROR:" e))))))]
      
      ;; Stop processing indicator
      (core/on-ui (core/stop-processing!))
      
      (let [final-result {:status "success"
                          :message-processed message
                          :tools-executed @tools-executed
                          :agent-response @accumulated-text
                          :streaming true
                          :timestamp (System/currentTimeMillis)}]
        (claude/debug-log "AGENT FINAL RESULT:" final-result)
        final-result))
    
    (catch Exception e
      ;; Stop processing indicator on error
      (core/on-ui (core/stop-processing!))
      (println "Error in process-message:" (.getMessage e))
      (.printStackTrace e)
      {:status "error"
       :error (.getMessage e)
       :message-processed message
       :timestamp (System/currentTimeMillis)})))