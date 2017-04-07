(ns insta-instaparse.core
  (:require [om.core :as om :include-macros true]
            [om.dom :as dom :include-macros true]
            [cljs.pprint :as pprint]
            [instaparse.core :as insta]))



(def trans
  (partial insta/transform {:class-name str}))

(defn parser [grammar]
  (insta/parser grammar
                :auto-whitespace :standard))


(defn parse [grammar text]
  (->> ((parser grammar) text)
       (trans)))


(enable-console-print!)

(defonce app-state (atom {:text "aaaabbbb"
                          :grammar "S = AB*
                          AB = A B
                          A = 'a'+
                          B = 'b'+"}))


(def textarea-style
  (clj->js {:width "45%"
            :height "500px"}))

(defn root-component [app owner]
  (reify
    om/IRender
    (render [_]
            (dom/div nil
                     (dom/textarea #js {:onChange (fn [e] (om/update! app [:text] (.. e -target -value))) :style textarea-style :value (:text app)})
                     (dom/textarea #js {:onChange (fn [e] (om/update! app [:grammar] (.. e -target -value)))  :style textarea-style :value (:grammar app)})
                     (dom/div nil (dom/button #js {:onClick (fn [e] (om/update! app [:parsed-text] (parse (:grammar app) (:text app))))} "parse"))
                     (dom/div nil (dom/pre nil
                                           (with-out-str
                                             (pprint/pprint
                                              (:parsed-text app)))))))))

(om/root
 root-component
 app-state
 {:target (. js/document (getElementById "app"))})
