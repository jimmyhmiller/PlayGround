 (ns clojure-asm.core
   (:import (org.objectweb.asm Opcodes Type ClassWriter)
            (org.objectweb.asm.commons Method GeneratorAdapter)))


(set! *warn-on-reflection* true)


;; For some reason my repl is messing up in this file but if I run 
;; (ns clojure-asm.core) manually in the repl it starts working?


;; Simple functional language
;; Support arithmetic
;; Support making functions
;; Support java interop
;; Support ADTs
;; Type check it
;; Support functions

;; I love the idea of a function, dynamic, jvm, non-lisp language. I
;; think this is a niche that is missing. But I could also see that
;; being a compile to js language.

;; What I think would be really interesting as well is a rust like
;; language with GC on the jvm. That seems not too difficult to start with.


(def INIT (Method/getMethod "void <init>()"))

(defn generate-code! [^GeneratorAdapter gen {:keys [type] :as code}]
  (case type
    :var
    (.loadArg ^GeneratorAdapter gen (:index code))
    :math
    (.math ^GeneratorAdapter gen GeneratorAdapter/ADD Type/INT_TYPE)
    :return-value
    (.returnValue ^GeneratorAdapter gen)))


(defn generate-default-constructor [^ClassWriter writer]
  (let [gen (GeneratorAdapter. Opcodes/ACC_PUBLIC INIT nil nil writer)]
    (.visitCode gen)
    (.loadThis gen)
    (.invokeConstructor gen (Type/getType Object) INIT)
    (.returnValue gen)
    (.endMethod gen)))


(defn generate-static-method [writer name {:keys [args return-type code] :as description}]
  (let [method (Method. name return-type (into-array Type (map :type args)))
        gen (GeneratorAdapter. (int (+ Opcodes/ACC_PUBLIC Opcodes/ACC_STATIC)) method nil nil writer)]
    (run! (fn [line] (generate-code! gen line)) code)
    (.endMethod ^GeneratorAdapter gen)))

(defn make-fn [fn-description]
  (let [class-name (:name fn-description)
        writer (ClassWriter. ClassWriter/COMPUTE_FRAMES)]
    (.visit ^ClassWriter writer Opcodes/V1_8 Opcodes/ACC_PUBLIC class-name nil "java/lang/Object" nil)
    (generate-default-constructor writer)
    (generate-static-method writer "invoke" fn-description)
    (.visitEnd writer)
    (.defineClass ^clojure.lang.DynamicClassLoader 
                  (clojure.lang.DynamicClassLoader.)
                  (.replace ^String class-name \/ \.) (.toByteArray ^ClassWriter  writer) nil)
    class-name))


(def fn-example
  {:type :fn
   :name "add"
   :args [{:type Type/INT_TYPE
           :name 'x}
          {:type Type/INT_TYPE
           :name 'y}]
   :code [{:type :var
           :index (int 0)}
          {:type :var
           :index (int 1)}
          {:type :math
           :op GeneratorAdapter/ADD
           :op-type Type/INT_TYPE}
          {:type :return-value}]
   :return-type Type/INT_TYPE})

(def fn-name (make-fn fn-example))

(add/invoke 2 3)


;; I should spend some time thinking about what this language should really look like.
;; I think given the jvm, it shouldn't be too hard to bootstrap the whole thing.
;; Which would be really really cool.

(comment
  (do
    (def writer (ClassWriter. ClassWriter/COMPUTE_FRAMES))


    (def class-name "jimmy/Hello2")


    (.visit ^ClassWriter  writer Opcodes/V1_8 Opcodes/ACC_PUBLIC class-name nil "java/lang/Object" nil)

    (def gen (GeneratorAdapter. Opcodes/ACC_PUBLIC INIT nil nil writer))

    ;; Generates the default constructor
    (.visitCode ^GeneratorAdapter gen)
    (.loadThis ^GeneratorAdapter gen)
    (.invokeConstructor ^GeneratorAdapter gen (Type/getType Object) INIT)
    (.returnValue ^GeneratorAdapter gen)
    (.endMethod ^GeneratorAdapter gen)

    (def my-method (Method/getMethod "int addSome(int, int)"))
    (def gen2 (GeneratorAdapter. (+ Opcodes/ACC_PUBLIC Opcodes/ACC_STATIC) my-method nil nil writer))

    (.visitCode ^GeneratorAdapter gen2)
    (.loadArg ^GeneratorAdapter gen2 (int 0))
    (.loadArg ^GeneratorAdapter gen2 (int 1))
    (.math ^GeneratorAdapter gen2  GeneratorAdapter/ADD Type/INT_TYPE)
    (.returnValue ^GeneratorAdapter gen2)
    (.endMethod ^GeneratorAdapter gen2)

    (.visitEnd ^ClassWriter writer)

    (def bytes (.toByteArray ^ClassWriter  writer))

    (def class-loader (clojure.lang.DynamicClassLoader.))

    (def klass (.defineClass ^clojure.lang.DynamicClassLoader 
                             (clojure.lang.DynamicClassLoader.)
                             (.replace ^String class-name \/ \.) bytes nil)))


  (import '[jimmy Hello2])

  (Hello2/addSome (int 1) (int 3))


  (decompiler/disassemble (.addSome (Hello2.))))
