(ns binary-parsing.core
  (:require [gloss.core :refer [compile-frame defcodec] :as gloss]
            [gloss.io :refer 
             [encode decode to-byte-buffer decode-stream decode-stream-headers]]
            [nio.core :as nio]
            [clojure.java.io :as io]
            [manifold.stream :as m-stream]
            [clojure.core.async :refer [<!!]])
  (:import [java.nio ByteBuffer]))



(defn byte-seq [rdr]
  (when-let [ch (.read rdr)]
    (cons (unchecked-byte ch) (lazy-seq (byte-seq rdr)))))


(def filename "/Users/jimmy.miller/Desktop/largedump.hprof")
(def stream (io/input-stream filename))
(def data (byte-seq stream))





(defn drop-until [pred coll]
  (drop 1 (drop-while pred coll)))

(set! *print-length* 10)

(defn decode-int [bytes]
  (decode :int32 (to-byte-buffer (take 4 bytes))))


(defn seek-to-first-tag [data]
  (->> data
       (drop-until (complement zero?))
       (drop 4)
       (drop 8)))

(defn get-tag [data]
  (first data))

(defn seek-to-next-tag [data]
  (let [length (decode-int (drop 5 data))]
    (drop length (drop 9 data))))


(defn get-tags
  ([data]
   (get-tags data {}))
  ([data tags]
   (if (empty? data)
     tags
     (recur (seek-to-next-tag data) (update tags (get-tag data) (fnil inc 0))))))


(def tags 
  (->> data
       seek-to-first-tag
       get-tags))

tags


(defcodec tags (gloss/enum :byte :none :str :load-class :unload-class))

(defcodec tag-info (gloss/ordered-map :tag tags
                                      :time :int32
                                      :length :int32))



(defmulti frame-for-tag (fn [{:keys [tag]}] tag))

(defmethod frame-for-tag :str [{:keys [length]}]
  (compile-frame (gloss/ordered-map :id :int64
                                    :str (gloss/string :utf-8 :length (- length 8)))))

(defmethod frame-for-tag :default [obj] 
  (compile-frame :byte))

(defcodec intro 
  (gloss/ordered-map :intro (gloss/string :utf-8 :delimiters ["\0"]) 
                     :size :int32 
                     :date :int64))


(defcodec objs (gloss/header tag-info frame-for-tag :tag))



(def result-stream (apply decode-stream-headers 
                          (map to-byte-buffer data) 
                          (cons intro (repeat objs))))




@(m-stream/take! result-stream)



