(ns muse-tools.core
  (:require [clojure.java.io :as io]
            [cheshire.core :as json]
            [clojure.string :as string])
  (:import [java.util.zip ZipOutputStream ZipEntry]
           [java.time Instant]
           [java.time.temporal ChronoUnit]))

(defn now []
  (.toString (.truncatedTo (Instant/now) ChronoUnit/MINUTES)))

(defn uuid-ish []
  (string/replace
   (str (java.util.UUID/randomUUID))
   "-" ""))


(defn make-board-id []
  (str "board_" (uuid-ish)))

(defn make-pdf-id []
  (str "pdf_" (uuid-ish)))

(defn make-board-with-pdf [{:keys [pdf-name label]}]
  (let [id (make-board-id)
        pdf-id (make-pdf-id)]
    {:documents
     {id
      {:cards [{:document_id pdf-id
                :position_x 95,
                :position_y 88,
                :size_height 150,
                :size_width 200,
                :z 2}],
       :created_at (now)
       :ink_models {:0 {}},
       :label label
       :should_auto_resize_card true,
       :snapshot_scale 0.30674847960472107,
       :type "board"},
      pdf-id
      {:created_at (now)
       :ink_models {},
       :label label
       :original_file pdf-name
       :type "pdf"}},
     :root id}))

(defn slurp-bytes
  "Slurp the bytes from a slurpable thing"
  [x]
  (with-open [out (java.io.ByteArrayOutputStream.)]
    (clojure.java.io/copy (clojure.java.io/input-stream x) out)
    (.toByteArray out)))


(do
  (def file (io/file "/Users/jimmyhmiller/Downloads/test.muse"))
  (def out (ZipOutputStream. (io/output-stream file)))

  (.putNextEntry out (ZipEntry. "Board 2021-09-06 11.03.15/"))
  (.putNextEntry out (ZipEntry. "Board 2021-09-06 11.03.15/contents.json"))

  (def contents (.getBytes (json/generate-string (make-board-with-pdf {:pdf-name"mypdf.pdf" :label "Hello World!"}))))
  (.write out contents 0 (alength contents))
  (.closeEntry out)
  (.putNextEntry out (ZipEntry. "Board 2021-09-06 11.03.15/metadata.json"))
  (def metadata (.getBytes (slurp "/Users/jimmyhmiller/Downloads/Board 2021-09-05 21.03.15/metadata.json")))
  (.write out metadata 0 (alength metadata))
  (.closeEntry out)

  (.putNextEntry out (ZipEntry. "Board 2021-09-06 11.03.15/files/"))
  (.putNextEntry out (ZipEntry. "Board 2021-09-06 11.03.15/files/mypdf.pdf"))
  (def pdf (slurp-bytes "/Users/jimmyhmiller/Downloads/Realism and Reason/files/pdf_a9d11ad6cd8341a9a9fc3aa035b046ea_original.pdf"))
  (.write out pdf 0 (alength pdf))
  (.closeEntry out)

  (.close out))





