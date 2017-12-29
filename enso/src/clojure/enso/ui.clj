(ns enso.ui
  (:require [enso.commands :as commands]
            [enso.parse :as parse]
            [seesaw.border :refer [empty-border]]
            [seesaw.color :refer [color]]
            [seesaw.core :refer [frame horizontal-panel label show! vertical-panel]]
            [seesaw.font :refer [font]]
            [seesaw.bind :as bind])
  (:import java.awt.Component))

(def ^:const green "#9fbe57")

(defn left-align [c]
  (doto c
    (.setAlignmentX Component/LEFT_ALIGNMENT)))

(defn create-help-label [text]
  (label
   :text text
   :background green
   :minimum-size [275 :by 100]
   :foreground :white
   :h-text-position :left
   :halign :left
   :v-text-position :top
   :valign :top
   :font "Gentium Plus-32"
   :border (empty-border :thickness 5)))

(def standard-help-text 
  "Welcome to Enso! Enter a command, or type \"help\" for assistance.")

(defn make-input-container []
  (left-align (horizontal-panel)))

(defn create-input-label []
  (label
   :halign :left
   :h-text-position :left
   :background "#000"
   :foreground "#fff"
   :font "Gentium Plus-64"
   :border (empty-border :thickness 5)))

(defn generic-label
  ([text color font-size font-style]
   (label
    :halign :left
    :h-text-position :left
    :text text
    :background "#000"
    :foreground color
    :font (font :nme "Gentium Plus" :style font-style :size font-size)))
  ([text color font-size]
   (generic-label text color font-size #{})))

(defn match-label [font-size match]
  (if (string? match)
    (generic-label match "#ABC26B" font-size #{:italic})
    (generic-label (last match) :white font-size)))

(defn draw-window [content]
  (let [window
        (doto (frame :undecorated? true :size [1000 :by 1000] :content content)
          (.setOpacity (float 0.0))
          (.setLocation 0 20)
          (.setAlwaysOnTop true)
          (.setBackground (color 0 0 0 0)))]
    (.putClientProperty (.getRootPane window) "Window.shadow" Boolean/FALSE)
    (show! window)
    window))

(def type->style 
  {:selected {:highlight :white
              :base green
              :arg :gray
              :size 64}
   :not-selected {:highlight "#c8d79e"
                  :base green
                  :arg :gray
                  :size 32}})

(defn generic-label-for-type [text type prop]
  (let [styles (type->style type)]
    (generic-label text (prop styles) (:size styles))))

(defmulti render-parsed-command (fn [parsed type] (first parsed)))

(defmethod render-parsed-command :line [[_ & children] type]
  (map #(render-parsed-command % type) children))

(defmethod render-parsed-command :match [[_ value] type]
  (generic-label-for-type value type :highlight))

(defmethod render-parsed-command :unmatch [[_ value] type]
  (generic-label-for-type value type :base))

(defmethod render-parsed-command :arg [[_ value] type]
  (generic-label-for-type value type :arg))

(defmethod render-parsed-command :default [_ _] nil)

(defn top-match-label [text]
  (let [commands (commands/get-commands-with-suggestions text)
        [command suggestion] (first commands)]
    (render-parsed-command (parse/parse text command suggestion) :selected)))

(defn other-match-labels [text]
  (let [commands (rest (commands/get-commands-with-suggestions text))]
    (->> commands
         (map (fn [[command suggestion]]
            (parse/parse text command suggestion)))
         (map #(render-parsed-command % :not-selected))
         (into []))))

(defn create-horizontal-panels [n]
  (doall (repeatedly n #(left-align (horizontal-panel)))))

(defn make-container [command-containers static-containers]
  (->> command-containers
       (concat static-containers)
       (into [])
       (vertical-panel :items)
       (left-align)))

;TODO: make this not ugly
(defn bind-input [state input-container]
  (bind/bind state
             (bind/transform #(top-match-label (:command-text %)))
             (bind/property input-container :items)))

;TODO: make this not ugly
(defn bind-auto-complete [state command-containers]
  (doall (for [i (range 10)]
           (bind/bind state
                      (bind/transform #(get (other-match-labels (:command-text %)) i []))
                      (bind/property (nth command-containers i) :items)))))

(defn bind-visibility [state window]
  (bind/subscribe state
                  #(if (:active %) 
                     (.setOpacity window (float 0.85))
                     (.setOpacity window (float 0.0)))))


