(ns lwjgl.main
  (:require
   [nrepl.server :as nrepl]
   [cider.nrepl :as cider.nrepl])
  (:import
   [java.nio DoubleBuffer]
   [org.jetbrains.skija
    Font
    FontStyle
    FontMgr 
    Typeface
    BackendRenderTarget Canvas ColorSpace DirectContext FramebufferFormat Paint Rect Surface SurfaceColorFormat SurfaceOrigin]
   [org.lwjgl.glfw Callbacks GLFW GLFWErrorCallback GLFWWindowSizeCallback]
   [org.lwjgl.opengl GL GL11]
   [org.lwjgl.system MemoryUtil]))


(set! *warn-on-reflection* true)

(defn color [^long l]
  (.intValue (Long/valueOf l)))

(def *rect-color (atom (color 0xFF000000)))


(def mgr (FontMgr/getDefault))
(def text-face-bold (.matchFamiliesStyle ^FontMgr  mgr (into-array String ["Ubuntu Mono"]) FontStyle/NORMAL))
(def text-font-36 (Font. ^Typeface text-face-bold (float 36)))



(def last-position (atom {:x 100 :y 100}))



(defn draw [window ^Canvas canvas]
  (.clear canvas (color 0xFF002b36))
  (let [paint (doto (Paint.) (.setColor (color 0xFFfdf6e3)))]
    (if (not (zero? (GLFW/glfwGetMouseButton window GLFW/GLFW_MOUSE_BUTTON_1)))
      (let [xarr (double-array [0]) 
            yarr (double-array [0])]
        (GLFW/glfwGetCursorPos (long window) xarr yarr)
        (reset! last-position {:x (* 2 (aget xarr 0)) :y (* 2 (aget yarr 0)) })
        (.drawString canvas "double : Int -> Int" 
                     (* 2 (aget xarr 0)) (* 2 (aget yarr 0))
                     text-font-36 paint))
      (let [{:keys [x y]} @last-position]
        (.drawString canvas "double : Int -> Int" 
                     x y
                     text-font-36 paint)))
    ))




(defn get-canvas-and-context [width height]
  (let [context (DirectContext/makeGL)
        fb-id   (GL11/glGetInteger 0x8CA6)
        target  (BackendRenderTarget/makeGL (* 2 width) (* 2 height) 0 8 fb-id FramebufferFormat/GR_GL_RGBA8)
        surface (Surface/makeFromBackendRenderTarget context target SurfaceOrigin/BOTTOM_LEFT SurfaceColorFormat/RGBA_8888 (ColorSpace/getSRGB))
        canvas  (.getCanvas surface)]

    {:context context :canvas canvas}))



;; Need to listen for events




(comment
  (GLFW/glfwSetCursor (long 1) (long 2)))

(defn -main [& args]
  (.set (GLFWErrorCallback/createPrint System/err))
  (GLFW/glfwInit)
  (GLFW/glfwWindowHint GLFW/GLFW_VISIBLE GLFW/GLFW_FALSE)
  (GLFW/glfwWindowHint GLFW/GLFW_RESIZABLE GLFW/GLFW_TRUE)
  (let [width 1024
        height 768
        window (GLFW/glfwCreateWindow width height "Skija LWJGL Demo" MemoryUtil/NULL MemoryUtil/NULL)]
    (def window window)
    (GLFW/glfwMakeContextCurrent window)
    (GLFW/glfwSwapInterval 1)

    (GLFW/glfwSetInputMode window GLFW/GLFW_CURSOR GLFW/GLFW_CURSOR_NORMAL)
    (GLFW/glfwSetCursor window (GLFW/glfwCreateStandardCursor GLFW/GLFW_CROSSHAIR_CURSOR))
    (GLFW/glfwGetMouseButton window GLFW/GLFW_MOUSE_BUTTON_1)
    
    (GLFW/glfwShowWindow window)  
    (GL/createCapabilities)

  

    (doto (Thread. #(clojure.main/main))
      (.start))

    (nrepl/start-server :port 7888 :handler cider.nrepl/cider-nrepl-handler)
    (println "nREPL server started at locahost:7888")

    (let [dimensions (atom {:width width :height height})
          {:keys [canvas context]} (get-canvas-and-context width height)]
      (GLFW/glfwSetWindowSizeCallback window (proxy [GLFWWindowSizeCallback] []
                                               (invoke [window width height]
                                                 (reset! dimensions {:width (long width) :height (long height)}))))
      (loop [^Canvas canvas canvas
             ^DirectContext context context
             width (long width)
             height (long height)]
        (let [current-dims @dimensions]
          (if (or (not= width (:width current-dims))
                  (not= height (:height current-dims)))
            (let [{:keys [width height]} current-dims
                  {:keys [canvas context]} (get-canvas-and-context width height)]
              (recur canvas context (long width) (long height)))
            (when (not (GLFW/glfwWindowShouldClose window))
              (.clear canvas (color 0xFFFFFFFF))
              (let [layer (.save canvas)]
                (#'draw window canvas)
                (.restoreToCount canvas layer))
              (.flush context)
              (GLFW/glfwSwapBuffers window)
              (GLFW/glfwPollEvents)
              (recur canvas context (long width) (long height))))))

      (Callbacks/glfwFreeCallbacks window)
      (GLFW/glfwHideWindow window)
      (GLFW/glfwDestroyWindow window)
      (GLFW/glfwPollEvents)
      (GLFW/glfwTerminate)
      (.free (GLFW/glfwSetErrorCallback nil))
      (shutdown-agents)
      )))

(comment
  (reset! lwjgl.main/*rect-color (lwjgl.main/color 0xFF33CC33)))



