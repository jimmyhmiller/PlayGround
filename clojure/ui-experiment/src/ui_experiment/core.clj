(ns ui-experiment.core
  (:require
   [nrepl.server :as nrepl])
  (:import
   [io.github.humbleui.skija BackendRenderTarget Canvas ColorSpace DirectContext FramebufferFormat Paint Surface SurfaceColorFormat SurfaceOrigin Font Typeface]
   [io.github.humbleui.types Rect]
   [org.lwjgl.glfw Callbacks GLFW GLFWErrorCallback]
   [org.lwjgl.opengl GL GL11]
   [org.lwjgl.system MemoryUtil]
   [java.util.concurrent ConcurrentLinkedQueue]))

(defn color [^long l]
  (.intValue (Long/valueOf l)))

(def *rect-color (atom (color 0xFFCC3300)))

(def *rectangles (atom []))

(def *draw-queue (atom []))

(def *draw-errors (atom []))

(defn add-rectangle! [x y width height color]
  (swap! *rectangles conj {:x x :y y :width width :height height :color color}))

(defn add-draw-fn! [draw-fn]
  (swap! *draw-queue conj draw-fn))

(defn clear-draw-queue! []
  (reset! *draw-queue []))

(defn get-and-clear-draw-errors! []
  (let [errors @*draw-errors]
    (reset! *draw-errors [])
    errors))

(defn force-draw! []
  (let [window-ref (atom nil)]
    ;; This is a hack to trigger a draw cycle
    (on-ui
      (println "Forcing draw cycle..."))))

(defn draw [^Canvas canvas]
  ;; Draw rectangles for backwards compatibility
  (doseq [{:keys [x y width height color]} @*rectangles]
    (try
      (let [paint (doto (Paint.) (.setColor color))]
        (.drawRect canvas (Rect/makeXYWH x y width height) paint))
      (catch Exception e
        (println "Error drawing rectangle:" (.getMessage e)))))
  ;; Draw arbitrary functions, removing any that error
  (let [current-queue @*draw-queue
        working-functions (atom [])]
    (doseq [[idx draw-fn] (map-indexed vector current-queue)]
      (try
        (draw-fn canvas)
        (swap! working-functions conj draw-fn)
        (catch Exception e
          (let [error-info {:timestamp (java.util.Date.)
                           :function-index idx
                           :error-message (.getMessage e)
                           :stack-trace (with-out-str (.printStackTrace e))}]
            (swap! *draw-errors conj error-info)
            (println "Error in drawing function, removing from queue:" (.getMessage e))))))
    (reset! *draw-queue @working-functions)))

(defn display-scale [window]
  (let [x (make-array Float/TYPE 1)
        y (make-array Float/TYPE 1)]
    (GLFW/glfwGetWindowContentScale window x y)
    [(first x) (first y)]))

(defonce ^ConcurrentLinkedQueue task-queue (ConcurrentLinkedQueue.))

(defn invoke-on-ui-thread [task]
  (.add task-queue task))

(defmacro on-ui [& body]
  `(invoke-on-ui-thread
     (fn []
       ~@body)))

(defn drain-ui-thread-tasks! []
  (loop []
    (when-let [task (.poll task-queue)]
      (try
        (task)
        (catch Exception e
          (println "Error in UI task:" (.getMessage e))
          (.printStackTrace e)))
      (recur))))

(defn start-nrepl! []
  (future
    (nrepl/start-server
     :port 7888
     :handler (nrepl/default-handler))))

(def *window-state (atom {:window nil
                          :context nil
                          :surface nil
                          :target nil
                          :canvas nil}))

(defn close-window! []
  (when-let [{:keys [window context surface target]} @*window-state]
    (when window
      (Callbacks/glfwFreeCallbacks window)
      (GLFW/glfwHideWindow window)
      (GLFW/glfwDestroyWindow window))
    (when surface (.close surface))
    (when target (.close target))
    (when context (.close context))
    (reset! *window-state {:window nil :context nil :surface nil :target nil :canvas nil})))

(defn create-window! []
  (let [width 640
        height 480
        window (GLFW/glfwCreateWindow width height "Skija LWJGL Demo" MemoryUtil/NULL MemoryUtil/NULL)]
    (GLFW/glfwMakeContextCurrent window)
    (GLFW/glfwSwapInterval 1)
    (GLFW/glfwShowWindow window)
    
    ;; Set up key callback for Command+N
    (GLFW/glfwSetKeyCallback window
      (reify org.lwjgl.glfw.GLFWKeyCallbackI
        (invoke [this window key scancode action mods]
          (when (and (= action GLFW/GLFW_PRESS)
                     (= key GLFW/GLFW_KEY_N)
                     (not= 0 (bit-and mods GLFW/GLFW_MOD_SUPER)))
            (on-ui (create-window!))))))
    
    ;; Set up close callback
    (GLFW/glfwSetWindowCloseCallback window
      (reify org.lwjgl.glfw.GLFWWindowCloseCallbackI
        (invoke [this window]
          (GLFW/glfwSetWindowShouldClose window true))))
    
    (GL/createCapabilities)
    
    (let [context (DirectContext/makeGL)
          fb-id   (GL11/glGetInteger 0x8CA6)
          [scale-x scale-y] (display-scale window)
          target  (BackendRenderTarget/makeGL (* scale-x width) (* scale-y height) 0 8 fb-id FramebufferFormat/GR_GL_RGBA8)
          surface (Surface/makeFromBackendRenderTarget context target SurfaceOrigin/BOTTOM_LEFT SurfaceColorFormat/RGBA_8888 (ColorSpace/getSRGB))
          canvas  (.getCanvas surface)]
      (.scale canvas scale-x scale-y)
      (reset! *window-state {:window window :context context :surface surface :target target :canvas canvas})
      {:window window :context context :surface surface :target target :canvas canvas})))

(defn render-loop! []
  (loop []
    (when-let [{:keys [window canvas context]} @*window-state]
      (when (and window (not (GLFW/glfwWindowShouldClose window)))
        (drain-ui-thread-tasks!)
        
        (.clear canvas (color 0xFFFFFFFF))
        (let [layer (.save canvas)]
          (draw canvas)
          (.restoreToCount canvas layer))
        (.flush context)
        (GLFW/glfwSwapBuffers window)
        (GLFW/glfwPollEvents)
        (recur)))
    ;; If window was closed, clean up and wait for a new one
    (close-window!)
    (Thread/sleep 100)
    (recur)))

(defn -main [& args]
  (start-nrepl!) 

  (.set (GLFWErrorCallback/createPrint System/err))
  (GLFW/glfwInit)
  (GLFW/glfwWindowHint GLFW/GLFW_VISIBLE GLFW/GLFW_FALSE)
  (GLFW/glfwWindowHint GLFW/GLFW_RESIZABLE GLFW/GLFW_TRUE)
  
  ;; Create initial window
  (create-window!)
  
  ;; Start render loop (this will handle window reopening)
  (render-loop!)
  
  ;; Cleanup on exit
  (close-window!)
  (GLFW/glfwTerminate)
  (.free (GLFW/glfwSetErrorCallback nil))
  (shutdown-agents))

(comment
  (on-ui
    (reset! ui-experiment.core/*rect-color (ui-experiment.core/color 0xFF33CC33)))
  
  ;; Add some rectangles
  (on-ui
    (add-rectangle! 100 100 50 50 (color 0xFFFF0000)))  ; Red rectangle
  
  (on-ui
    (add-rectangle! 200 200 75 75 (color 0xFF00FF00)))  ; Green rectangle
  
  (on-ui
    (add-rectangle! 300 150 100 60 (color 0xFF0000FF))) ; Blue rectangle
  
  ;; Clear all rectangles
  (on-ui
    (reset! ui-experiment.core/*rectangles []))
  
  ;; Add a custom drawing function for a circle
  (on-ui
    (add-draw-fn! 
     (fn [^Canvas canvas]
       (let [paint (doto (Paint.) 
                     (.setColor (color 0xFFFF00FF)))]
         (.drawCircle canvas 400 300 50 paint)))))
  
  ;; Add a custom drawing function for text
  (on-ui
    (add-draw-fn!
     (fn [^Canvas canvas]
       (let [paint (doto (Paint.)
                     (.setColor (color 0xFF000000)))
             font (doto (Font.) (.makeWithSize 24))]
         (.drawString canvas "Hello Skija!" 50 50 font paint)))))
  
  ;; Clear the draw queue
  (on-ui
    (clear-draw-queue!)))
