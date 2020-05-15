(ns clojure-asm.core
  (:import [org.objectweb.asm Opcodes Type ClassWriter]
           [org.objectweb.asm.commons Method GeneratorAdapter]))



(def writer (ClassWriter. ClassWriter/COMPUTE_FRAMES))

(def init (Method/getMethod "void <init>()"))

(def my-method (Method/getMethod "int addSome()"))

(def class-name "jimmy/Hello2")


(.visit writer Opcodes/V1_8 Opcodes/ACC_PUBLIC class-name nil "java/lang/Object" nil)

(def gen (GeneratorAdapter. Opcodes/ACC_PUBLIC init nil nil writer))

;; Generates the default constructor
(.visitCode gen)
(.loadThis gen)
(.invokeConstructor gen (Type/getType Object) init)
(.returnValue gen)
(.endMethod gen)


(def gen2 (GeneratorAdapter. Opcodes/ACC_PUBLIC my-method nil nil writer))

(.visitCode gen2)
(.push gen2 (int 2))
(.push gen2 (int 2))
(.math gen2 GeneratorAdapter/ADD Type/INT_TYPE)
(.returnValue gen2)
(.endMethod gen2)

(.visitEnd writer)

(def bytes (.toByteArray writer))

(def class-loader (clojure.lang.DynamicClassLoader.))

(def klass (.defineClass class-loader (.replace class-name \/ \.) bytes nil))


(import '[jimmy Hello2])

(.addSome (Hello2.))

