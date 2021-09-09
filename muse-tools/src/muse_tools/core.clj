(ns muse-tools.core
  (:require [clojure.java.io :as io]
            [cheshire.core :as json]
            [clojure.string :as string])
  (:import [java.util.zip ZipOutputStream ZipEntry]
           [java.time Instant]
           [java.time.temporal ChronoUnit]))


(def metadata
  {:bundle_version 6,
   :created_at "2021-09-06T01:03:19Z",
   :device "iPad8,11",
   :muse_build "5321",
   :muse_bundle_id "com.musesoftware.museios-appstore",
   :muse_version "1.8.1",
   :os_version "15.0",
   :resolution [2048 2732]})

(defn now []
  (.toString (.truncatedTo (Instant/now) ChronoUnit/MINUTES)))

(defn uuid-ish []
  (string/replace
   (str (java.util.UUID/randomUUID))
   "-" ""))

(defn slurp-bytes
  "Slurp the bytes from a slurpable thing"
  [x]
  (with-open [out (java.io.ByteArrayOutputStream.)]
    (clojure.java.io/copy (clojure.java.io/input-stream x) out)
    (.toByteArray out)))


(defn make-board-id []
  (str "board_" (uuid-ish)))

(defn make-pdf-id []
  (str "pdf_" (uuid-ish)))

(defn add-board [context {:keys [label cards id parent-id position size]}]
  (let [board-id (or id (make-board-id))]
    (cond-> context
      true
      (assoc-in [:muse :documents board-id]
                {:cards cards
                 :created_at (now)
                 :ink_models {:0 {}}
                 :label label
                 :should_auto_resize_card true,
                 ;; Probably need to figure this out
                 :snapshot_scale 0.15
                 :type "board"})

      parent-id
      ;; probably should extract
      (update-in [:muse :documents parent-id :cards]
                 conj
                 {:document_id board-id
                  :position_x (:x position)
                  :position_y (:y position)
                  :size_height (or (:height size) 150),
                  :size_width (or (:width size) 200),
                  :z 2}))))


(defn add-root [context {:keys [id label]}]
  (let [board-name (-> (str "Board_" (now) "/")
                       (string/replace ":" ".")
                       (string/replace "Z" ""))]
    (.putNextEntry (:zip context) (ZipEntry. board-name))
    (-> context
        (assoc-in [:muse :root] id)
        (assoc-in [:muse :documents id] {:label label :cards []})
        (assoc-in [:meta :board-name] board-name))))


(defn initialize-zip [{:keys [output-path]}]
  (let [file (io/file output-path)
        zip  (ZipOutputStream. (io/output-stream file))]
    zip))

(defn initial-context [{:keys [output-path]}]
  {:muse
   {:documents {}
    :root nil}
   :zip (initialize-zip {:output-path output-path})})


(defn add-pdf [context {:keys [label file id parent-id position size]}]
  (let [pdf-id (or id (make-pdf-id))]
    (.putNextEntry (:zip context)
                   (ZipEntry. (str (get-in context [:meta :board-name]) "/files/" (.getName file))))
    (let [data (slurp-bytes file)]
      (.write (:zip context) data 0 (alength data)))
    (.closeEntry (:zip context))
    (cond-> context
      true
      (assoc-in [:muse :documents pdf-id]
                {:created_at (now)
                 :ink_models {},
                 :label label
                 :original_file (.getName file)
                 :type "pdf"})
      parent-id
      (update-in [:muse :documents parent-id :cards]
                 conj
                 {:document_id pdf-id
                  :position_x (:x position)
                  :position_y (:y position)
                  :size_height (or (:height size) 150),
                  :size_width (or (:width size) 200),
                  :z 2}))))


(defn grid-locations [coll {:keys [margin elem-width elem-height width-max]}]
  (map (fn [elem position]
         (assoc elem :position position))
       coll
       (for [y (range margin 1000000(+ elem-height margin))
             x (range margin width-max (+ elem-width margin))]
         {:x x :y y})))


(defn add-pdf-wrapper [context {:keys [label file parent-id position]}]
  (let [board-id (make-board-id)]
    (-> context
        (add-board {:id board-id :label label :parent-id parent-id :position position})
        (add-pdf {:label label :file file :position {:x 50 :y 110} :parent-id board-id}))))


(defn make-pdf-grid [context parent-id [label pdfs]]
  (reduce (fn [context {:keys [label file position]}]
            (add-pdf-wrapper context {:label label
                                      :file file
                                      :parent-id parent-id
                                      :position position}))
          context
          (grid-locations pdfs {:margin 80 :elem-width 185 :elem-height 250 :width-max 1400})))

(defn finalize-context [context]
  (let [contents (.getBytes (json/generate-string (:muse context)))
        metadata (.getBytes (json/generate-string metadata))]
    (.putNextEntry (:zip context) (ZipEntry. (str (get-in context [:meta :board-name]) "/contents.json")))
    ;; Pull to a function
    (.write (:zip context) contents 0 (alength contents))
    (.closeEntry (:zip context))
    (.putNextEntry (:zip context) (ZipEntry. (str (get-in context [:meta :board-name]) "/metadata.json")))
    (.write (:zip context) metadata 0 (alength metadata))
    (.closeEntry (:zip context))
    (.close (:zip context))))


(defn get-article-name [file]
  (-> file
      (.getName)
      (string/split #"-")
      last
      (string/replace ".pdf" "")))

(def worrydream-files-by-author
  (->> (file-seq (io/file "/Users/jimmyhmiller/Desktop/worrydream/worrydream.com/refs"))
       (remove  #(.isDirectory %))
       (filter (file-name-f #(string/ends-with? % ".pdf")))
       (map (fn [file] {:file file :label (.getName file)}))
       (group-by (comp extract-author-from-name :label))
       (sort-by (comp count second))
       reverse
       (map (fn [[author pdfs]]
              [author (map (fn [{:keys [file] :as context}]
                             (assoc context :label (get-article-name file)))
                           pdfs)])
            )
       #_(take 2)
       #_(take-while #(> (count (second %)) 1))
       #_(into {})))


;; This is close, but sizing isn't working properly. Probably need to play with how that works
(defn make-pdf-grid-all-authors [context author-groups]
  (reduce (fn [context {:keys [author pdfs position]}]
            (let [board-id (make-board-id)]
              (-> context
                  (add-board {:id board-id
                              :label author
                              :parent-id  (get-in context [:muse :root])
                              ;; Need to change position
                              :position position})
                  (make-pdf-grid board-id [author pdfs]))))
          context
          (grid-locations
           (map (fn [[author pdfs]]
                  {:author author
                   :pdfs pdfs})
                author-groups)
           {:margin 80 :elem-width 250 :elem-height 150 :width-max 1400})))



(defn produce-whole-future-of-coding-board []
  (let [root (make-board-id)]
    (-> (initial-context {:output-path "/Users/jimmyhmiller/Downloads/test2.muse"})
        (add-root {:id root :label "Future of Coding"})
        (make-pdf-grid-all-authors worrydream-files-by-author)
        (finalize-context))))

(produce-whole-future-sof-coding-board)


(defn file-name-f [f]
  (fn [file]
    (f (.getName file))))

(defn extract-author-from-name [file-name]
  (string/trim (first (string/split file-name #"-"))))





(comment


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

    (.close out)))





