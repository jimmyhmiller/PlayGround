;; java.io modeled in the language — byte streams over raw arrays, declared
;; with `defclass` like everything else in the JVM layer (host_jvm.clj). This
;; is the substrate the REAL nrepl/bencode.clj reads and writes through.
;;
;; Byte conventions (matching Java exactly):
;;   * a byte[] is a raw array of SIGNED ints (-128..=127) — `%str->bytes` /
;;     `%bytes->str` and `unchecked-byte` produce/consume that range;
;;   * `.read` returns UNSIGNED 0..255, or -1 at end of stream;
;;   * `.write` of an int truncates to a signed byte.
;;
;; The classes load into clojure.core (the file runs during the core phase);
;; the `clojure.java.io` namespace at the bottom holds the user-facing fns.

;; ─────────────── ByteArrayOutputStream ───────────────
;; append arr[off..off+len) (signed bytes) onto the growable array `buf`
(defn -write-bytes-into [buf arr off len]
  (let [off (if (nil? off) 0 off)
        len (if (nil? len) (%alength arr) len)]
    (loop [i 0]
      (if (%lt i len)
        (do (-al-add! buf (unchecked-byte (%aget arr (%add off i))))
            (recur (%add i 1)))
        nil))))
;; field 0 = a growable raw array of signed bytes
(defn -baos-write-array [o arr off len]
  (-write-bytes-into (field o 0) arr off len))

(defclass java.io.ByteArrayOutputStream
  (:tag ByteArrayOutputStream)
  (:ctor ([] (%make-record 'ByteArrayOutputStream (list (array))))
         ([sz] (%make-record 'ByteArrayOutputStream (list (array)))))
  (:method write
    ([o b] (if (number? b)
             (-al-add! (field o 0) (unchecked-byte b))
             (-baos-write-array o b nil nil))
           nil)
    ([o arr off len] (-baos-write-array o arr off len) nil))
  (:method toByteArray [o] (%aclone (field o 0)))
  (:method toString ([o] (%bytes->str (field o 0))) ([o charset] (%bytes->str (field o 0))))
  (:method size [o] (%alength (field o 0)))
  (:method reset [o] (%aclear (field o 0)) nil)
  (:method flush [o] nil)
  (:method close [o] nil))

;; ─────────────── ByteArrayInputStream ───────────────
;; fields: [bytes pos-atom]
(defn -bais-read-1 [o]
  (let [arr (field o 0)
        pos (field o 1)
        i (deref pos)]
    (if (%lt i (%alength arr))
      (do (reset! pos (%add i 1))
          (%bit-and (%aget arr i) 255))   ;; unsigned, as InputStream.read
      -1)))

;; generic 3-arg read over any stream with a 1-arg .read: fill buf[off..off+len),
;; return the count actually read, or -1 if at end of stream immediately.
(defn -stream-read-into [in buf off len]
  (loop [i 0]
    (if (%lt i len)
      (let [b (.read in)]
        (if (neg? b)
          (if (%num-eq i 0) -1 i)
          (do (%cell-set! buf (%add off i) (unchecked-byte b))
              (recur (%add i 1)))))
      len)))

(defclass java.io.ByteArrayInputStream
  (:tag ByteArrayInputStream)
  (:ctor ([bytes] (%make-record 'ByteArrayInputStream (list bytes (atom 0)))))
  (:method read
    ([o] (-bais-read-1 o))
    ([o buf off len] (-stream-read-into o buf off len)))
  (:method available [o] (%sub (%alength (field o 0)) (deref (field o 1))))
  (:method close [o] nil))

;; ─────────────── PushbackInputStream ───────────────
;; fields: [underlying pushback-atom] — one byte of pushback is what
;; bencode/nREPL need (they unread at most one delimiter probe).
(defclass java.io.PushbackInputStream
  (:tag PushbackInputStream)
  (:ctor ([in] (%make-record 'PushbackInputStream (list in (atom nil))))
         ([in sz] (%make-record 'PushbackInputStream (list in (atom nil)))))
  (:method read
    ([o] (let [pb (field o 1)
               b (deref pb)]
           (if (nil? b)
             (.read (field o 0))
             (do (reset! pb nil) (%bit-and b 255)))))
    ([o buf off len] (-stream-read-into o buf off len)))
  (:method unread [o b] (reset! (field o 1) b) nil)
  (:method close [o] (.close (field o 0)) nil))

;; interface entries so `extend-type InputStream/OutputStream` (bencode does
;; the former) lands on the concrete stream tags.
(defclass java.io.InputStream
  (:kind :interface)
  (:tag PushbackInputStream)
  (:extend-tags PushbackInputStream ByteArrayInputStream))
(defclass java.io.OutputStream
  (:kind :interface)
  (:tag ByteArrayOutputStream)
  (:extend-tags ByteArrayOutputStream))

;; ─────────────── writers (*out* / *err* targets) ───────────────
;; The dynamic vars *out*/*err* (core.clj) hold Writer-shaped objects: `.write`
;; a string, `.flush`. The bootstrap defaults are plain records of the tags
;; below, so these classes give them their methods; StringWriter is what
;; with-out-str / an nREPL eval capture binds.

(defclass java.io.StringWriter
  (:tag StringWriter)
  (:ctor ([] (%make-record 'StringWriter (list (array)))))
  ;; field 0 = an array of string chunks, joined at .toString (O(1) appends)
  (:method write [w s] (-al-add! (field w 0) (str s)) nil)
  (:method append [w s] (-al-add! (field w 0) (str s)) w)
  (:method toString [w]
    (loop [i 0 acc ""]
      (if (%lt i (%alength (field w 0)))
        (recur (%add i 1) (%str-cat acc (%aget (field w 0) i)))
        acc)))
  (:method flush [w] nil)
  (:method close [w] nil))

(defclass microclj.io.StdoutWriter
  (:tag StdoutWriter)
  (:method write [w s] (%print s) nil)
  (:method flush [w] nil)
  (:method close [w] nil))

(defclass microclj.io.StderrWriter
  (:tag StderrWriter)
  (:method write [w s] (%err-print s) nil)
  (:method flush [w] nil)
  (:method close [w] nil))

;; ─────────────── java.net sockets (over the %tcp-* prims) ───────────────
;; A Socket record holds the prim handle; its streams read/write bytes with
;; Java's conventions (unsigned reads, -1 at EOF). Writes buffer per stream
;; and go to the wire on `.flush` (one syscall per message, not per byte).

(defclass java.net.SocketInputStream
  (:tag SocketInputStream)
  (:ctor ([h] (%make-record 'SocketInputStream (list h))))
  (:method read
    ([o] (%tcp-read (field o 0)))
    ([o buf off len] (-stream-read-into o buf off len)))
  (:method close [o] nil))

(defclass java.net.SocketOutputStream
  (:tag SocketOutputStream)
  (:ctor ([h] (%make-record 'SocketOutputStream (list h (array)))))
  (:method write
    ([o b] (if (number? b)
             (-al-add! (field o 1) (unchecked-byte b))
             (-write-bytes-into (field o 1) b nil nil))
           nil)
    ([o arr off len] (-write-bytes-into (field o 1) arr off len) nil))
  (:method flush [o]
    (let [buf (field o 1)]
      (if (%lt 0 (%alength buf))
        (do (%tcp-write (field o 0) buf) (%aclear buf))
        nil)))
  (:method close [o] (.flush o) nil))

(defclass java.net.Socket
  (:tag Socket)
  (:ctor ([h] (%make-record 'Socket (list h (atom false)))))
  (:method getInputStream [s] (java.net.SocketInputStream. (field s 0)))
  (:method getOutputStream [s] (java.net.SocketOutputStream. (field s 0)))
  (:method isClosed [s] (deref (field s 1)))
  (:method close [s] (reset! (field s 1) true) (%tcp-close (field s 0)) nil))

(defclass java.net.ServerSocket
  (:tag ServerSocket)
  (:ctor ([port] (%make-record 'ServerSocket (list (%tcp-listen port) (atom false)))))
  (:method accept [s] (java.net.Socket. (%tcp-accept (field s 0))))
  (:method getLocalPort [s] (%tcp-local-port (field s 0)))
  (:method isClosed [s] (deref (field s 1)))
  (:method close [s] (reset! (field s 1) true) (%tcp-close (field s 0)) nil))

;; ─────────────── clojure.java.io ───────────────
(ns clojure.java.io)

;; copy an InputStream's remaining bytes to an output stream (the one shape
;; nrepl's stack uses; readers/writers arrive with the socket layer).
(defn copy [in out]
  (loop []
    (let [b (.read in)]
      (if (neg? b)
        nil
        (do (.write out b) (recur))))))
