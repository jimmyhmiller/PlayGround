(ns test-react-hook-form.main
  (:require [helix.core :as h]
            [helix.hooks :as hooks]
            [helix.dom :as d]
            ["react-dom/client" :as rdom]
            ["react-hook-form" :as form]))


(h/defnc app []
  (let [form (form/useForm)
        register (.-register form)
        handle-submit (.-handleSubmit form)
        on-submit (fn [data] (println data))]
    (d/form {:onSubmit (handle-submit on-submit)}
            (d/input {:& (register "name")})
            (d/input {:type "submit"}))))
 

(defn init []
  ;; start your app with your favorite React renderer
  (-> (rdom/createRoot (js/document.getElementById "app"))
      (.render (h/$ app))))
