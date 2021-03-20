(ns example-live-view.todo
  (:require [live-view-server.core :as live-view]
            [clojure.string :as string]))


(def state (atom {:todos (sorted-map)
                  :counter 0
                  :input ""
                  :filter :all
                  :editing nil
                  :editing-input ""}))

@state

(reset! state {:todos (sorted-map)
                  :counter 0
                  :input ""
                  :filter :all
                  :editing nil
                  :editing-input ""})


(defn add-todo [text]
  (let [{:keys [counter]} (swap! state update :counter inc)]
    (swap! state assoc-in [:todos counter]
           {:id counter :title text :done false})
    (swap! state assoc :input "")))

(defn toggle [id] (swap! state update-in [:todos id :done] not))
(defn save [id title] (swap! state assoc-in [:todos id :title] title))
(defn delete [id] (swap! state update :todos dissoc id))

(defn mmap [m f a] (->> m (f a) (into (empty m))))
(defn complete-all [v] (swap! state update :todos mmap map #(assoc-in % [1 :done] v)))
(defn clear-done [] (swap! state update :todos mmap remove #(get-in % [1 :done])))

(defonce init (do
                (add-todo "Rename Cloact to Reagent")
                (add-todo "Add undo demo")
                (add-todo "Make all rendering async")
                (add-todo "Allow any arguments to component functions")
                (complete-all true)))

(defn todo-input [{:keys [input on-save on-stop id class placeholder]}]
  [:input {:type "text"
           :value input
           :id id
           :class class
           :placeholder placeholder
           :onblur [:save]
           :onchange [:change-input]
           :onkeydown [:input-key-press]}])


(defn todo-edit [{:keys [input on-save on-stop id class placeholder]}]
  [:input {:type "text"
           :autofocus "autofocus"
           :value input
           :id id
           :class class
           :placeholder placeholder
           :onblur [:save-edit {:id id}]
           :onchange [:change-edit-input ]
           :onkeydown [:edit-input-key-press {:id id}]}])


(defn todo-stats [{:keys [filt active done]}]
  (let [props-for (fn [name]
                    {:class (if (= name filt) "selected")
                     :onclick [:filter {:name name}]})]
    [:div
     [:span#todo-count
      [:strong active] " " (case active 1 "item" "items") " left"]
     [:ul#filters
      [:li [:a (props-for :all) "All"]]
      [:li [:a (props-for :active) "Active"]]
      [:li [:a (props-for :done) "Completed"]]]
     (when (pos? done)
       [:button#clear-completed {:onclick [:clear-done]}
        "Clear completed " done])]))



(defn todo-item [{:keys [editing id done title input]}]
  ;; Fix
  (let [editing (= editing id)]
    [:li {:class (str (if done "completed ")
                      (if editing "editing"))}
     [:div.view
      [:input.toggle {:type "checkbox" :checked done
                      :onchange [:toggle-todo {:id id}]}]
      [:label {:ondoubleclick [:editing {:id id}]} title]
      [:button.destroy {:onclick [:delete {:id id}]}]]
     (when editing
       (todo-edit {:class "edit"
                   :input input
                   :id id}))]))

(defn todo-app [state]
  (let [filt (:filter state)]
    (let [items (vals (:todos state))
          done (->> items (filter :done) count)
          active (- (count items) done)]
      [:body
       [:div
        [:link {:rel "stylesheet" :href "todos.css"}]
        [:section#todoapp
         [:header#header
          [:h1 "todos"]
          (todo-input {:input (:input state)
                       :id "new-todo"
                       :placeholder "What needs to be done?"
                       :on-save add-todo})]
         (when (-> items count pos?)
           [:div
            [:section#main
             [:input#toggle-all {:type "checkbox" :checked (zero? active)
                                 :onchange [:complete-all]}]
             [:label {:for "toggle-all"} "Mark all as complete"]
             [:ul#todo-list
              (for [todo (filter (case filt
                                   :active (complement :done)
                                   :done :done
                                   :all identity) items)]
                ^{:key (:id todo)} (todo-item (assoc todo
                                                     :editing (:editing state)
                                                     :input (:editing-input state))))]]
            [:footer#footer
             (todo-stats {:active active :done done :filt filt})]])]
        [:footer#info
         [:p "Double-click to edit a todo"]]]])))


(defn view [state]
  (todo-app state))



(defn event-handler [{:keys [action]}]
  (prn action)
  (let [[action-type payload] action]
    (case action-type
      :delete (delete (:id payload))
      :toggle-todo (toggle (:id payload))
      :complete-all (complete-all (:value payload))
      :change-input (swap! state assoc :input (:value payload))
      :input-key-press (when (= (:keycode payload) 13)
                         (add-todo (:input @state)))
      :editing (swap! state assoc
                      :editing (:id payload)
                      :editing-input (get-in @state [:todos (:id payload) :title]))
      :change-edit-input (swap! state assoc :editing-input (:value payload))
      :edit-input-key-press (when (= (:keycode payload) 13)
                              (swap! state assoc :editing nil))
      :filter (swap! state assoc :filter (:name payload))
      :clear-done (clear-done)
      :save-edit (do
                   (swap! state (fn [state]
                                  (-> state
                                      (assoc-in [:todos (:id payload) :title]
                                                (:editing-input state))
                                      (assoc :editing nil
                                             :editing-input "")))))
      (println "not handled" action))))
@state
(def live-view-server
  (live-view/start-live-view-server
   {:state state
    :view #'view
    :event-handler #'event-handler
    :port 56789}))


(comment

  (.stop live-view-server))



;; Need to let state be a var and be able to re-eval
