(ns noxt.lib
  (:require [goog.object]
            [react]
            [clojure.string :as string]))

(def LinkContext
  (react/createContext {:component-registry {}
                        :on-change nil
                        :load-page nil}))


(defn switch-route [component-registry page on-change load-page]
  (load-page page
             (fn [new-page component]
               (swap! component-registry assoc page component)
               (js/history.pushState #js{} "ignored" (if (= page :index) "/" (str "/" (name page))))
               (on-change))))

(defn Link [{:keys [page]} text]
  (let [{:keys [component-registry on-change load-page]} (react/useContext LinkContext)]
    [:a {:href (str "/" (if (= page :index) "" (name page)))
         :on-click (fn [e]
                     (.preventDefault e)
                     (switch-route component-registry page on-change load-page))}
     text]))
