(ns notre-dame.core
  (:require [cheshire.core :as json]
            [hickory.core :as html]
            [meander.epsilon :as m]
            [clojure.string :as string]
            [clojure.java.io :as io]
            [notre-dame.utils :as utils]))


(def initial-url "https://ndpr.nd.edu/news/archives/")
(def base-url "https://ndpr.nd.edu")

(defn fetch-page [link]
  (Thread/sleep 100)
  (html/as-hiccup
   (html/parse
    (slurp link))))


(defn archive-links [initial-page]
  (m/search initial-page
    (m/$ [:span {:class "published_year"} 
        [:a {:href ?href} . _ ...]])
    (str base-url ?href)))


(defn fetch-links [links]
  (doall
   (map fetch-page links)))


(defn find-article-links [archive-link]
  (m/search archive-link
    (m/$ [:h1 {:class "entry-title"}
        [:a {:href ?href} . _ ...]])
    (str base-url ?href)))

(defn find-pagination-links [archive-index]
  (m/match archive-index
    (m/$ [:div {:class "pagination"} . _ _ . [:span _ ?first] . _ ... . [:a _ ?last] _ _])
    (range (Integer/parseInt ?first) (inc (Integer/parseInt ?last)))))

(defn all-pages [archive-link pages]
  (for [page pages]
    (str archive-link "page/" page "/")))



(defn isbn? [x]
  (and (string? x)
       (some? (re-find #"ISBN.*" x))))

(defn extract-isbn [isbn]
  (-> (or (nth (re-find #"(ISBN|IBSN).*?([0-9-]+)" isbn) 2) "")
      (string/replace "-" "")))



(defn find-isbn [article-page]
  (extract-isbn
   (apply str
          (m/search article-page
            (m/$ [:p {} . _ ... . (m/$ (pred string? ?string)) . _ ... ])
            ?string))))


(comment

  (def results
    (let [initial-page (fetch-page initial-url)
          index-links (archive-links initial-page)
          index-link-data (fetch-links index-links)
          index-numbered-pages (map find-pagination-links index-link-data)
          all-search-listing-urls (mapcat all-pages index-links index-numbered-pages)
          all-search-listings (map fetch-page all-search-listing-urls)
          all-article-page-urls (mapcat find-article-links all-search-listings)
          all-article-pages  (map-indexed (fn [index page]
                                            (println index page)
                                            (let [page-data (fetch-page page)
                                                  isbn (find-isbn page-data)]
                                              (println isbn) 
                                              [page page-data isbn])) 
                                          all-article-page-urls)]
      (doall all-article-pages))))



(defn fetch-gen-lib-search-page [isbn]
  (let [page (fetch-page (str  "http://libgen.lc/search.php?req=" isbn))
        href (m/find page
               (m/$ [:a {:href ?href} "[1]"])
               ?href)
        extension (m/find page
                     (m/$ [:td {:nowrap ""} (m/and (m/or "pdf" "epub") ?extension)])
                    ?extension)]
    {:href href
     :extension extension}))

;; stupid bug in stupid library
(defn parse-download-page [download-page]
  (html/as-hiccup (last (second download-page))))

(defn download-page-info [{:keys [href extension] :as info}]
  (try
    (if-not href
      info
      (let [page (parse-download-page (fetch-page href))
            img (m/find page
                        (m/$ [:img {:src ?image}])
                        ?image)
            title (m/find page
                          (m/$ [:td {} (m/re #"Title: (.*)" [_ ?title]) & _])
                          ?title)
            href (m/find page
                         (m/$ [:a {:href ?href} [:h2 {} "GET"]])
                         ?href)
            description (m/find page
                                (m/$ [:td {:colspan "2"} (m/pred string? !description) &
                                      (m/gather (m/pred string? !description))])
                                (string/join "\n" !description))]

        (merge info
               {:image img
                :title title
                :download href
                :description description})))
    (catch Exception e
      (println "error fetching" info)
      info)))


(def state (atom {:n 0
                  :items []}))

(count
 (filter :href
         (:items @state)))

(comment)


(def task
  (future
    (doall
     (map (fn [{:keys [link isbn]}]
            (let [item (merge {:isbn isbn
                               :ndpr link} 
                              (download-page-info (fetch-gen-lib-search-page isbn)))]
              (swap! state (fn [state] (-> state
                                           (update :n inc)
                                           (update :items conj item))))))
          (map (fn [[link isbn]]
                 {:link (last
                         (re-matches #"[0-9]+ (.*)" link))
                  :isbn isbn})
               (partition 2
                          (string/split-lines
                           (slurp "/Users/jimmyhmiller/Documents/misc/scraping/isbn.txt"))))))))


(spit "/Users/jimmyhmiller/Documents/misc/scraping/partial.html"
      (utils/to-html
       (m/rewrite (filterv :download (:items @state))
         [{:image !image
           :href !href
           :ndpr !ndpr
           :title !title
           :download !download
           :description !description} ...]

         [:html
          [:head
           [:link {:rel "stylesheet" :href "index.css" :type "text/css"}]]
          [:body
           [:div {:class "container"} .
            [:div {:class "item"}
             [:h3 {:class "heading"} [:a {:href !download} !title]]
             [:img {:class "image" :src (m/app str "http://booksdl.org/" !image)}]
             [:p {:class "review"} [:a {:href !ndpr} "Review"]]
             [:p {:class "description"} !description]] ...]]])))

(add-watch state :logger
           (fn [_ _ _ new-state]
             (println (str "fetching " (:n new-state)))))


(first (filterv :download (:items @state)))

(spit "/Users/jimmyhmiller/Documents/misc/scraping/partial.json"
      (json/generate-string @state))

  ;; a mess






(m/search (parse-download-page download-info)
  (m/$ [:td {:colspan "2"} & ?children])
  ?children)

(def info (fetch-gen-lib-search-page "9780231171489"))

(download-page-info
 (fetch-gen-lib-search-page "9780231171489"))

(def download-page
  (fetch-page
   (m/find gen-lib-page
     (m/$ [:a {:href ?href} "[1]"])
     ?href)))





  ;; need to grab extension and title
  ;; Maybe grab the description and notre dame review
  ;; Then I could browse easier?





(defn copy-uri-to-file [file uri]
  (with-open [in (clojure.java.io/input-stream uri)
              out (clojure.java.io/output-stream file)]
    (clojure.java.io/copy in out)))

(copy-uri-to-file "/Users/jimmyhmiller/Desktop/book.pdf"
                  (m/find (parse-download-page download-page)
                    (m/$ [:a {:href ?href} [:h2 {} "GET"]])
                    ?href))

(def gen-lib-page
  (fetch-page "http://libgen.lc/search.php?req=9780231171489"))



(def example-page (fetch-page "https://ndpr.nd.edu/news/richard-wollheim-on-the-art-of-painting-art-as-representation-and-expression/"))




(find-article-links
 (fetch-page "https://ndpr.nd.edu/news/archives/2019/page/1/"))

  
  
