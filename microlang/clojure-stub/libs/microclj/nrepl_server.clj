;; microclj's nREPL server — the wire protocol is the REAL nrepl/bencode.clj
;; (vendored, unmodified) over the in-language java.net/java.io layer; this
;; file is the op dispatch (clone/describe/eval/load-file/close/…), speaking
;; the standard nREPL message shapes so real clients (nrepl.core, CIDER,
;; Calva) connect as to any nREPL endpoint.
;;
;; v1 serves connections SERIALLY on the calling thread: evals re-enter the
;; live compiler through the eval bridge, which is single-threaded state.
;; (Real transport.clj + concurrent sessions need the executor/interruption
;; surface — the next milestone.)

(ns microclj.nrepl-server
  (:require [nrepl.bencode :as bencode]))

(def -session-counter (atom 0))
(def -sessions (atom ()))

;; bencode hands us byte arrays for every string payload — stringify shallowly.
(defn -bstr [v]
  (if (%num-eq (type-of v) 'Vector) (String. v "UTF-8") v))

(defn -req-str [req k]
  (let [v (get req k)] (if (nil? v) nil (-bstr v))))

(defn -send! [out msg]
  (bencode/write-bencode out msg)
  (.flush out))

;; every response carries the request's id + session (nREPL protocol).
(defn -respond [out req m]
  (let [base m
        base (if-let [id (-req-str req "id")] (assoc base "id" id) base)
        base (if-let [s (-req-str req "session")] (assoc base "session" s) base)]
    (-send! out base)))

(defn -new-session! []
  (let [id (str "microclj-session-" (swap! -session-counter inc))]
    (swap! -sessions conj id)
    id))

(def -describe-ops
  {"clone" {} "describe" {} "eval" {} "load-file" {} "close" {}
   "ls-sessions" {} "interrupt" {}})

(defn -op-clone [req out]
  (-respond out req {"new-session" (-new-session!) "status" ["done"]}))

(defn -op-describe [req out]
  (-respond out req
            {"versions" {"nrepl" {"major" 1 "minor" 3 "incremental" 1
                                  "version-string" "1.3.1"}
                         "clojure" {"major" 1 "minor" 12 "incremental" 0
                                    "version-string" "1.12.0-microclj"}}
             "ops" -describe-ops
             "aux" {}
             "status" ["done"]}))

(defn -op-eval [req out]
  (let [code (or (-req-str req "code") (-req-str req "file"))
        w (java.io.StringWriter.)
        ew (java.io.StringWriter.)]
    (try
      ;; wrap in `do`: clients send whole buffers; a top-level do evals as a
      ;; sequence of top-level forms, so `(ns …)` in the buffer switches the
      ;; session's namespace. *out*/*err* are captured and streamed back.
      (let [v (binding [*out* w *err* ew]
                (eval (read-string (str "(do " code "\n)"))))]
        (let [os (.toString w)] (if (= os "") nil (-respond out req {"out" os})))
        (let [es (.toString ew)] (if (= es "") nil (-respond out req {"err" es})))
        (-respond out req {"value" (pr-str v) "ns" (name (%current-ns))})
        (-respond out req {"status" ["done"]}))
      (catch :default e
        (let [os (.toString w)] (if (= os "") nil (-respond out req {"out" os})))
        (-respond out req {"err" (str (.getMessage e) "\n")})
        (-respond out req {"ex" (pr-str e) "root-ex" (pr-str e)
                           "ns" (name (%current-ns))
                           "status" ["eval-error" "done"]})))))

(defn -op-close [req out]
  (-respond out req {"status" ["done" "session-closed"]}))

(defn -op-ls-sessions [req out]
  (-respond out req {"sessions" (vec (deref -sessions)) "status" ["done"]}))

(defn -op-interrupt [req out]
  ;; serial server: nothing is running concurrently to interrupt
  (-respond out req {"status" ["done" "interrupt-id-mismatch"]}))

(defn -dispatch [req out]
  (let [op (-req-str req "op")]
    (cond
      (= op "clone") (-op-clone req out)
      (= op "describe") (-op-describe req out)
      (= op "eval") (-op-eval req out)
      (= op "load-file") (-op-eval req out)
      (= op "close") (-op-close req out)
      (= op "ls-sessions") (-op-ls-sessions req out)
      (= op "interrupt") (-op-interrupt req out)
      :else (-respond out req {"status" ["error" "unknown-op" "done"]}))))

(defn -handle-connection [sock]
  (let [in (java.io.PushbackInputStream. (.getInputStream sock))
        out (.getOutputStream sock)]
    (try
      (loop []
        (let [msg (bencode/read-nrepl-message in)]
          (-dispatch msg out)
          (recur)))
      ;; EOF / disconnect / bad frame ends this connection, never the server
      (catch :default e nil))
    (.close sock)))

(defn start-server!
  "Serve nREPL on 127.0.0.1:port (0 = pick a free port), forever."
  [port]
  (let [ss (java.net.ServerSocket. port)]
    (println (str "nREPL server started on port " (.getLocalPort ss)
                  " on host 127.0.0.1 - nrepl://127.0.0.1:" (.getLocalPort ss)))
    (loop []
      (let [sock (.accept ss)]
        (-handle-connection sock)
        (recur)))))
