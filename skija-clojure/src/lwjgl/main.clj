(ns lwjgl.main
  (:require
   [nrepl.server :as nrepl]
   [cider.nrepl :as cider.nrepl]
   [clojure.string :as string])
  (:import
   [java.nio DoubleBuffer]
   [org.jetbrains.skija.paragraph TextStyle ParagraphBuilder Paragraph ParagraphStyle FontCollection]
   [org.jetbrains.skija
    Font
    FontStyle
    FontMgr
    Typeface
    BackendRenderTarget Canvas ColorSpace DirectContext FramebufferFormat Paint Rect Surface SurfaceColorFormat SurfaceOrigin]
   [org.lwjgl.glfw Callbacks GLFW GLFWErrorCallback GLFWWindowSizeCallback  GLFWKeyCallback]
   [org.lwjgl.opengl GL GL11]
   [org.lwjgl.system MemoryUtil]))


(set! *warn-on-reflection* true)

(defn color [^long l]
  (.intValue (Long/valueOf l)))

(def *rect-color (atom (color 0xFF000000)))


(def mgr (FontMgr/getDefault))
(def fc (FontCollection.))
(.setDefaultFontManager ^FontCollection fc mgr)
(def text-face-bold (.matchFamiliesStyle ^FontMgr  mgr (into-array String ["Ubuntu Mono"]) FontStyle/NORMAL))
(def text-font-36 ^Font (Font. ^Typeface text-face-bold (float 36)))


(def colors
  {:yellow    0xffb58900
   :orange    0xffcb4b16
   :red       0xffdc322f
   :magenta   0xffd33682
   :violet    0xff6c71c4
   :blue      0xff268bd2
   :cyan      0xff2aa198
   :green     0xff859900
   :neutral   0xfffdf6e3 })


(defn set-color-alpha [color number-float]
  (bit-or (bit-and color 0x00FFFFFF)
          (bit-shift-left (int (* 255 number-float)) 24)))



(defn style-over-steps [start-size end-size steps text-color]
  (map-indexed (fn [i size]
                 (-> (TextStyle.)
                     (.setFontFamilies (into-array String ["Ubuntu Mono"]))
                     (.setFontSize (int size))
                     (.setColor (color (set-color-alpha (colors text-color)  (/ i steps))))))
               (range start-size (inc end-size) (/ (- end-size start-size) (dec steps)))))


(def text-styles
  (atom
   (into {}
         (map (fn [[name color-value]]
                [name (-> (TextStyle.)
                          (.setFontSize 36)
                          (.setColor (color color-value))
                          (.setFontFamilies (into-array String ["Ubuntu Mono"]))
                          )])
              colors))))





(swap! text-styles assoc :cyan (first (drop 0 (style-over-steps 10 36 10 :cyan))))



(defn draw-color-dsl [^Canvas canvas text x y]
  (let [pb (doto (ParagraphBuilder. (-> (ParagraphStyle.)
                                        (.setTextStyle (@text-styles :neutral)))
                                    fc))]
    (doseq [segment text]
      (let [current-text (if (vector? segment) (second segment) segment)]
        (when (vector? segment)
          (.pushStyle pb (@text-styles (first segment))))
        (.addText pb current-text)
        (when (vector? segment)
          (.popStyle pb))))

    (let [p (.build pb)]
      (.layout p 400)
      (.paint p canvas x y))))



(def last-position (atom {:x 100
                          :y 100}))








;; Need to make it so you actually only drag if in the rect of the paragraph.
;; That shouldn't be too hard.
;; Or am I dragging the whole canvas instead?

(def dion-attack-example
  [[:blue "Attack Kind"] " :: enum" "\n"
   [:cyan "  Fire\n" ]
   [:cyan "  Water\n"]
   [:cyan "  Earth\n"]
   [:cyan "  Air\n"]
   [:cyan "  Light\n"]
   [:cyan "  Dark\n"]
   [:cyan "  Poison\n"]])

(def double-example
  [[:cyan "double"] " : " [:blue "Int"] " -> " [:blue "Int"] "\n"
   [:cyan "double"] " x = x " [:green "*"] " 2"])

(defn draw [window ^Canvas canvas]
  (.clear canvas (color 0xFF002b36))
  (let [paint (doto (Paint.) (.setColor (color (colors :cyan))))
        text dion-attack-example]
    (if (not (zero? (GLFW/glfwGetMouseButton window GLFW/GLFW_MOUSE_BUTTON_1)))
      (let [xarr (double-array [0])
            yarr (double-array [0])]
        (GLFW/glfwGetCursorPos (long window) xarr yarr)
        (reset! last-position {:x (+ (* 2 (aget xarr 0)) 20) :y (- (* 2 (aget yarr 0)) 30)})
        (draw-color-dsl canvas
                        text
                        (+ (* 2 (aget xarr 0)) 20)
                        (- (* 2 (aget yarr 0)) 30)))
      (let [{:keys [x y]} @last-position]
        (draw-color-dsl canvas
                        text
                        x y)))))




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
      (GLFW/glfwSetKeyCallback window (proxy [GLFWKeyCallback] []
                                        (invoke [window key scancode action mods]
                                          (future
                                            (when (zero? action)
                                              (cond (= key 83)
                                                    (let [steps 20]
                                                      (loop [i 0]
                                                        (when (< i steps)
                                                          (swap! text-styles assoc :cyan (first (drop i (style-over-steps 16 36 steps :cyan))))
                                                          (Thread/sleep 5)
                                                          (recur (inc i)))))
                                                    (= key 72)
                                                    (swap! text-styles assoc :cyan (first (style-over-steps 16 36 10 :cyan)))
                                                    :else nil))))))
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
