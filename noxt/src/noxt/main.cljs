(ns noxt.main
  (:require [uix.core.alpha :as uix :include-macros true]
            [uix.dom.alpha :as uix.dom]
            [uix.compiler.alpha :as compiler]
            [react]
            [cljs.loader :as loader]
            [goog.object]
            [noxt.lib]
            [clojure.string :as string]
            ["react-dom/server" :as react-dom])
  (:import [goog.events EventType]))

(enable-console-print!)

(defn current-location []
  (if (= js/window.location.pathname "/") 
    :index
    (keyword (subs js/window.location.pathname 1))))

(defn current-page []
  (let [{:keys [registry-provider]} (react/useContext noxt.lib/LinkContext)]))

(defn app []
  (let [current-page (uix/state (current-location))
        component-registry (uix/state {})]
    (uix/with-effect [@current-page]
      (noxt.lib/load-page @current-page 
                          (fn [new-page component]
                            (swap! component-registry assoc new-page component))))

    (uix/with-effect []
      (let [f (fn [e]
                (reset! current-page (current-location)))]
        (js/window.addEventListener "popstate" f)
        (fn []
          (js/window.removeEventListener "popstate" f))))

    [:> (.-Provider noxt.lib/LinkContext) {:value {:component-registry component-registry
                                                   :on-change #(reset! current-page (current-location))}}
     (let [Comp (get @component-registry @current-page)]
       (if Comp [Comp] nil))]))

(uix.dom/render [app] (js/document.getElementById "app"))


(loader/set-loaded! :main)