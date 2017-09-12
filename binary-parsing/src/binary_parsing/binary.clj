(import '[java.nio.channels FileChannel]
        '[java.nio.file StandardOpenOption]
        '[java.nio.file FileSystems]
        '[java.nio.file Paths]
        '[java.nio.file OpenOption]
        '[java.nio ByteBuffer])

(def file (Paths/get (java.net.URI. "file:///Users/jimmy.miller/Desktop/largedump.hprof")))

(def channel (FileChannel/open file (into-array OpenOption [StandardOpenOption/READ])))

(defn read-until [pred channel]
  (let [buf (ByteBuffer/allocate 1)]
    (.read channel buf)
    (if-not (pred (.get buf 0))
      (recur pred channel)
      channel)))

(defn skip [n channel] 
  (.position channel (+ (.position channel) n)))

(defn get-bytes [n channel]
  (let [buf (ByteBuffer/allocate n)]
    (.read channel buf)
    buf))

(defn get-int [channel]
  (.getInt (get-bytes 4 channel) 0))

(defn skip-to-first-tag [channel]
  (->> channel
      (read-until zero?)
      (skip 4)
      (skip 8)))

(defn skip-to-next-tag [channel]
  (skip 4 channel)
  (let [length (get-int channel)]
    (skip length channel)
    length))

(defn get-tag [channel]
  (.get (get-bytes 1 channel) 0))

(defn parse-file [channel]
  (try
    (.position channel 0)
    (skip-to-first-tag channel)
    (loop [c channel]
      (when (< (.position c) 19555451801)
        (get-tag c)
        (skip-to-next-tag c)
        (recur c)))
    (catch Exception e (throw e))))

(defn skip-around [channel]
  (loop [c channel]
    (skip 1000 c)
    (recur c)))


(comment
  (skip-around channel)

  (time (parse-file channel))

  (.position channel 0))
