(ns markdown-to-blog.core
  (:require [markdown-to-hiccup.core :as mark]
            [clojure.string :as string]
            [hiccup.core :as hiccup]))


(def markdown
  (slurp "/Users/jimmyhmiller/Documents/Code/PlayGround/writings/programming/Card Driven Development.md"))


(defmulti transform
  (fn [data] (when (vector? data)
               (first data))))

(defn heading [size [_ _ content]] 
  [:Heading {:text content
             :size size}])

(defmethod transform :h1 [[_ _ content]]
  [:Title {:text content}])

(defmethod transform :h2 [tag]
  (heading 2 tag))

(defmethod transform :h3 [tag]
  (heading 3 tag))

(defmethod transform :h4 [tag]
  (heading 4 tag))

(defmethod transform :h5 [tag]
  (heading 5 tag))

(defmethod transform :h5 [tag]
  (heading 6 tag))

(defmethod transform :blockquote [[_ attr content]]
  [:BlockQuote attr content])

(defn indent-lines [code]
  (->> (string/split code #"\n")
       (string/join "\n    ")
       (str "    ")))

(defmethod transform :pre [[_ _ [tag attr source] :as content]]
  (if (not= tag :code)
    (throw (ex-info "Pre without conde" content))
    [(keyword (string/capitalize (:class attr))) {} (str "\n  {`\n" (indent-lines source) "\n  `}\n")]))

(defmethod transform :code [[_ attr body :as content]]
  (if (not (contains? attr :class))
    [:Term {} body]
    content))

(defmethod transform :div [[_ _ & content]]
  [:GlobalLayout {} content])

(defmethod transform nil [x]
  x)

(defmethod transform :default [x]
  x)

(defn convert-md [content]
  (->>
   (mark/md->hiccup content)
   (mark/component)
   (clojure.walk/postwalk transform)
   (hiccup/html)))

(convert-md markdown)

(comment
  (add-watch (var convert-md)
             :print-it
             (fn [_ _ _ _]
               (clojure.pprint/pprint (convert-md markdown)))))
