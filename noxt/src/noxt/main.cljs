(ns noxt.main
  (:require [uix.core.alpha :as uix :include-macros true]
            [uix.dom.alpha :as uix.dom]
            [cljs.loader :as loader]
            [goog.object]
            [clojure.string :as string]
            [noxt.index])
  (:import [goog.events EventType]))

(enable-console-print!)

(defn find-page-main [page]
  (reduce (fn [obj prop] (goog.object/get obj prop))
          js/window
          (concat (string/split (name page) #"\." ) ["main"])))

;; I can't resolve methods dynamically.
;; Do I somehow make it not munge these names?
;; Do I generate this source with all the page names?
;; ^export might be the right approach.
;; It will require some preprocessing the source
(defn load-page [page cb]
  (loader/load page
               (fn []
                 (cb page (find-page-main page)))))


(defn app []
  (let [page (uix/state :index)
        component-registry (uix/state {:index noxt.index/main})]
    [:<>
     [(get @component-registry @page)]
     [:a {:href "#" :on-click (fn [e]
                                (.preventDefault e)
                                (load-page :noxt.about (fn [new-page component]
                                                         (swap! component-registry assoc new-page component)
                                                         (swap! page new-page))))}
      "Load About"]]))

(uix.dom/render [app] (js/document.getElementById "app"))


(loader/set-loaded! :main)


(comment

  (defonce page (atom "index"))

  (events/listen (gdom/getElement "button") EventType.CLICK
                 (fn [e]
                   (if (= "index" @page)
                     (loader/load :noxt.about
                                  (fn []
                                    ((resolve 'noxt.about/main))
                                    (reset! page "about")))
                     (do (noxt.index/main)
                         (reset! page "index"))))))
