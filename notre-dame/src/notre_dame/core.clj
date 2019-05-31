(ns notre-dame.core
  (:require [cheshire.core :as json]
            [hickory.core :as html]
            [meander.match.delta :as m]
            [clojure.string :as string]))



(def initial-url "https://ndpr.nd.edu/news/archives/")
(def base-url "https://ndpr.nd.edu")

(defn fetch-page [link]
  (Thread/sleep 1000)
  (html/as-hiccup
   (html/parse
    (slurp link))))


(defn archive-links [initial-page]
  (m/search initial-page
    ($ [:span {:class "published_year"} 
        [:a {:href ?href} . _ ...]])
    (str base-url ?href)))


(defn fetch-links [links]
  (doall
   (map fetch-page links)))


(defn find-article-links [archive-link]
  (m/search archive-link
    ($ [:h1 {:class "entry-title"}
        [:a {:href ?href} . _ ...]])
    (str base-url ?href)))

(defn find-pagination-links [archive-index]
  (m/match archive-index
    ($ [:div {:class "pagination"} . _ _ . [:span _ ?first] . _ ... . [:a _ ?last] _ _])
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

(extract-isbn " 3, Kenneth Reinhard and Susan Spitzer (trs.), Columbia University Press, 2018, 261pp., $30.00 (hbk), ISBN 9780231171489.")

(defn find-isbn [article-page]
  (extract-isbn
   (apply str
          (m/search article-page
            ($ [:p {} . _ ... . ($ (pred string? ?string)) . _ ... ])
            ?string))))



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
    (doall all-article-pages)))

(def example-page (fetch-page "https://ndpr.nd.edu/news/form-matter-substance/"))


(find-article-links
 (fetch-page "https://ndpr.nd.edu/news/archives/2019/page/1/"))

