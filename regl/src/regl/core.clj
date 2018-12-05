(ns regl.core
  (:require [fn-fx.fx-dom :as dom]
            [fn-fx.controls :as ui]
            [fn-fx.diff :refer [component defui render should-update?]]
            [fn-fx.util :refer [run-later]]
            [glow.core :as highlight]
            [glow.colorschemes :as colorschemes]
            [clojure.repl :as repl])
  (:import [javafx.scene.control TreeItem]
           [javafx.beans.value ChangeListener]))



(defn generate-code [text]
  (str "<html><head><style type=\"text/css\">\n
      body { background: #fff; font-size: 16px }\n"
       (highlight/generate-css (assoc colorschemes/solarized-dark :background "white"))
       "\n</style>
      </head>
      <body>\n"
       (highlight/highlight-html text)
       "\n</body>
      </html>"))

(def state (atom {:html ""
                  :items [{:text "hello"}
                          {:text "thing"}]}))


(def ui (agent nil))

(def tree
  (let [root (TreeItem. "asdf")]
    (.add (.getChildren root) (TreeItem. "asadfadf"))
    root))

(defui MainComponent 
  (should-update? [_ _ _] false)
  (render [this args]
          (ui/stage
           :title "Hello World!"
           :shown true
           :min-width 300
           :min-height 300
           :scene (ui/scene
                   :root (ui/h-box
                          :children [(ui/list-view :id "list" 
                                                   :items ["map" "filter" "clojure.repl/source"])
                                     (ui/web-view :id "web-browser")])))))


(defn load-html [html]
  (let [engine (.getEngine (.lookup (.getScene @(:root @ui)) "#web-browser"))]
    (run-later
     (.loadContent engine html))))

(defn handler [{:keys [event] :as e}]
  (if (= event :list-changed)
    (let [html (generate-code (repl/source-fn (symbol (:new e))))]
      (run-later
       (load-html html)
       (swap! state 
              assoc 
              :html html
              :fn-name (:new e)))))
  (println "Received Event: " e))

(defn set-list-selection-handler []
  (run-later
   (let [list-view (.lookup (.getScene @(:root @ui)) "#list")]
     (.. list-view 
         (getSelectionModel)
         (selectedItemProperty)
         (addListener (reify ChangeListener
                        (changed [this observable old new]
                          (handler {:event :list-changed
                                    :old old
                                    :new new}))))))))



(defn update-app [state]
  (send ui
        (fn [old-ui]
          (dom/update-app old-ui (main-component state)))))

(defn startup [state]
  (send ui (fn [_]
             (dom/app (main-component @state) handler)))
  (await ui)
  (set-list-selection-handler))

(startup state)
(update-app @state)

(remove-watch #'main-component :ui)

(add-watch #'main-component :ui
           (fn [_ _ _ ns]
             (println "changed!")
             (update-app ns)))

(add-watch state :ui
           (fn [_ _ _ ns]
             (println "changed!")
             (update-app ns)))
