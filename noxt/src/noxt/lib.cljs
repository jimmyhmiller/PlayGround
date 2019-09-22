(ns noxt.lib
  (:require [goog.object]
            [cljs.loader :as loader]
            [react]
            [clojure.string :as string]))

(def LinkContext
  (react/createContext {:component-registry {}}))

(defn find-page-main [page]
  (reduce (fn [obj prop] (goog.object/get obj prop))
          js/window
          (concat (string/split (name page) #"\." ) ["main"])))


(defn load-page [page cb]
  (loader/load page
               (fn []
                 (cb page (find-page-main page)))))


(defn switch-route [component-registry page on-change]
  (load-page page
             (fn [new-page component]
               (swap! component-registry assoc page component)
               (js/history.pushState #js{} "ignored" (if (= page :index) "/" (str "/" (name page))))
               (on-change))))

(defn Link [{:keys [page]} text]
  (let [{:keys [component-registry on-change]} (react/useContext LinkContext)]
    [:a {:href (str "/" (if (= page :index) "" (name page)))
         :on-click (fn [e]
                     (.preventDefault e)
                     (switch-route component-registry page on-change))}
     text]))
