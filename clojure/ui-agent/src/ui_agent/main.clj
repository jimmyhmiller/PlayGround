(ns ui-agent.main
  (:require
   [ui-agent.core :as core]
   [ui-agent.server :as server])
  (:gen-class))

(defn -main [& args]
  (println "Starting UI Agent...")
  
  ;; Start nREPL first
  (core/start-nrepl!)
  
  ;; Start the HTTP server in a background thread
  (let [port (if (seq args) 
               (Integer/parseInt (first args))
               8080)]
    (future (server/start-server! port)))
  
  (println "UI Agent starting...")
  (println "- nREPL server running on port 7889") 
  (println "- HTTP server running on port 8080")
  (println "- Send messages to: http://localhost:8080/message")
  
  ;; Initialize UI system on main thread (required for OpenGL)
  (println "Initializing UI system on main thread...")
  (.set (org.lwjgl.glfw.GLFWErrorCallback/createPrint System/err))
  (org.lwjgl.glfw.GLFW/glfwInit)
  (org.lwjgl.glfw.GLFW/glfwWindowHint org.lwjgl.glfw.GLFW/GLFW_VISIBLE org.lwjgl.glfw.GLFW/GLFW_FALSE)
  (org.lwjgl.glfw.GLFW/glfwWindowHint org.lwjgl.glfw.GLFW/GLFW_RESIZABLE org.lwjgl.glfw.GLFW/GLFW_TRUE)
  
  ;; Create initial window
  (core/create-window!)
  
  (println "UI Agent fully started!")
  
  ;; Run render loop on main thread (this blocks)
  (core/render-loop!)
  
  ;; Cleanup on exit
  (core/close-window!)
  (org.lwjgl.glfw.GLFW/glfwTerminate)
  (.free (org.lwjgl.glfw.GLFW/glfwSetErrorCallback nil))
  (shutdown-agents))