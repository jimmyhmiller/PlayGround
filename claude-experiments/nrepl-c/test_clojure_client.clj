#!/usr/bin/env clojure

;; Test nREPL server with Clojure nREPL client
(require '[clojure.java.shell :as shell])

;; Add nREPL dependency
(require '[clojure.java.io :as io])

(try
  (require '[nrepl.core :as nrepl])
  (catch Exception e
    (println "Installing nREPL dependency...")
    (shell/sh "clojure" "-Sdeps" "{:deps {nrepl/nrepl {:mvn/version \"1.0.0\"}}}" "-e" "nil")
    (require '[nrepl.core :as nrepl])))

(defn test-nrepl-server []
  (println "Testing nREPL server with Clojure client...")
  (println "Connecting to localhost:7888...")

  (with-open [conn (nrepl/connect :port 7888)]
    (println "Connected!")

    ;; Test describe
    (println "\n1. Testing describe operation:")
    (let [client (nrepl/client conn 1000)
          response (nrepl/message client {:op "describe"})]
      (doseq [msg response]
        (println "  " msg)))

    ;; Test clone session
    (println "\n2. Testing clone operation:")
    (let [client (nrepl/client conn 1000)
          response (nrepl/message client {:op "clone"})]
      (doseq [msg response]
        (println "  " msg))
      (let [session-id (-> response first :new-session)]
        (println "  Created session:" session-id)

        ;; Test eval with session
        (println "\n3. Testing eval with session:")
        (let [eval-response (nrepl/message client {:op "eval"
                                                     :code "(+ 1 2)"
                                                     :session session-id})]
          (doseq [msg eval-response]
            (println "  " msg)))

        ;; Test close session
        (println "\n4. Testing close operation:")
        (let [close-response (nrepl/message client {:op "close"
                                                      :session session-id})]
          (doseq [msg close-response]
            (println "  " msg)))))

    (println "\n✓ All Clojure client tests completed!")))

(test-nrepl-server)
