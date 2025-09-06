(ns ui-agent.claude-streaming
  (:require
   [cheshire.core :as json]
   [clojure.string :as str]))

(defn parse-sse-line [line]
  "Parse a Server-Sent Events line"
  (when (and line (str/starts-with? line "data: "))
    (let [data (subs line 6)]
      (when-not (= data "[DONE]")
        (try
          (json/parse-string data true)
          (catch Exception _ nil))))))

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
                          (cond
                            ;; Text delta - just stream the text
                            (= (:type chunk) "content_block_delta")
                            (let [delta (get-in chunk [:delta :text] "")]
                              (when (seq delta) (on-text-chunk delta))
                              [current-response current-tool-block])
                            
                            ;; Tool input delta - accumulate tool parameters
                            (and (= (:type chunk) "content_block_delta") 
                                 (get-in chunk [:delta :partial_json]))
                            (let [partial-json (get-in chunk [:delta :partial_json])
                                  updated-tool (if current-tool-block
                                                 (update current-tool-block :input-json str partial-json)
                                                 nil)]
                              [current-response updated-tool])
                            
                            ;; Start of content block
                            (= (:type chunk) "content_block_start")
                            (let [content-block (get chunk :content_block)]
                              (if (= (:type content-block) "tool_use")
                                ;; Start accumulating tool - don't execute yet
                                [current-response (assoc content-block :input-json "")]
                                [(update current-response :content conj {:type "text" :text ""}) current-tool-block]))
                            
                            ;; End of content block 
                            (= (:type chunk) "content_block_stop")
                            (if current-tool-block
                              ;; Tool block complete - now we can execute it
                              (let [complete-input (try
                                                      (json/parse-string (:input-json current-tool-block) true)
                                                      (catch Exception _ {}))
                                    complete-tool (assoc current-tool-block :input complete-input)]
                                (on-tool-use complete-tool)
                                [(update current-response :content conj complete-tool) nil])
                              [current-response current-tool-block])
                            
                            ;; Message delta
                            (= (:type chunk) "message_delta")
                            [(assoc current-response :stop_reason (get-in chunk [:delta :stop_reason])) current-tool-block]
                            
                            :else [current-response current-tool-block])]
                      (recur updated-response updated-tool-block)))
                  (recur current-response current-tool-block)))
              ;; End of stream
              (on-complete current-response)))))
      (catch Exception e
        (println "Error processing stream:" (.getMessage e))
        (on-complete {:error (.getMessage e)})))))