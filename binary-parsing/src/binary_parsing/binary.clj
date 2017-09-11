(import '[java.nio.channels FileChannel]
        '[java.nio.file StandardOpenOption]
        '[java.nio.file FileSystems]
        '[java.nio.file Paths]
        '[java.nio.file OpenOption]
        '[java.nio ByteBuffer])

(def file (Paths/get (java.net.URI. "file:///Users/jimmyhmiller/Desktop/test.txt")))

(def channel (FileChannel/open file (into-array OpenOption [StandardOpenOption/READ])))

(defn read-until [pred channel]
  (let [buf (ByteBuffer/allocate 1)]
    (.read channel buf)
    (when-not (pred (.get buf 0))
      (recur pred channel))))

(defn skip [n channel] 
  (.position channel (+ (.position channel) n)))

(defn get-bytes [n channel]
  (let [buf (ByteBuffer/allocate n)]
    (.read channel buf)
    buf))

(defn get-int [channel]
  (.getInt (get-bytes 4 channel)))

(defn skip-to-first-tag [channel]
  (-> channel
      (read-until zero?)
      (skip 4)
      (skip 8)))

(defn seek-to-next-tag [channel]
  (skip 4)
  (let [length (get-int channel)]
    (skip length channel)
    length))





