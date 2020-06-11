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
    (.math gen (:op code) (:op-type code))
    :get-static-field
    (.getStatic gen (:owner code) (:name code) (:result-type code))
    :invoke-static
    (.invokeStatic gen (:owner code) (:method code))
    :invoke-virtual
    (.invokeVirtual gen (:owner code) (:method code))
    :invoke-constructor
    (.invokeConstructor gen (:owner code) (:method code))
    :new
    (.newInstance gen (:owner code))
    :store-local
    (.storeLocal gen (:index code) (:local-type code))
    :load-local
    (.loadLocal gen (:index code) (:local-type code))
    :push-int
    (.push gen (int (:value code)))
    :put-field
    (.putField gen (:owner code) (:name code) (:field-type code))
    :dup
    (.dup gen)
    :pop
    (.pop gen)
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

(defn initialize-class [^ClassWriter writer class-name]
  (.visit writer Opcodes/V1_8 Opcodes/ACC_PUBLIC class-name nil "java/lang/Object" nil))

(defn make-fn [fn-description]
  (let [class-name (:name fn-description)
        writer (ClassWriter. ClassWriter/COMPUTE_FRAMES)]
    (initialize-class writer class-name)
    (generate-default-constructor writer)
    (generate-static-method writer "invoke" fn-description)
    (.visitEnd writer)
    (.defineClass ^clojure.lang.DynamicClassLoader 
                  (clojure.lang.DynamicClassLoader.)
                  (.replace ^String class-name \/ \.) (.toByteArray ^ClassWriter  writer) nil)
    class-name))

(defn resolve-type-of-field [[_ class field]]
  (Type/getType ^java.lang.Class (.getGenericType (.getDeclaredField (Class/forName class) field)))) 

(comment
  (.visitField writer Opcodes/ACC_PUBLIC "tag" (.getDescriptor Type/INT_TYPE) nil (int 0))
  (.visitEnd writer))



(defn make-variant-factory [^ClassWriter writer class-name [i {:keys [name]}]]
  (let [method (Method. name (Type/getObjectType class-name) (into-array Type []))
        gen (GeneratorAdapter. (int (+ Opcodes/ACC_PUBLIC Opcodes/ACC_STATIC)) method nil nil writer)]
    (run! (fn [line] (generate-code! gen line))
          [{:type :new
            :owner (Type/getObjectType class-name)}
           {:type :dup}
           {:type :invoke-constructor
            :owner (Type/getObjectType class-name)
            :method INIT}
           {:type :store-local
            :index (int 0)
            :local-type  (Type/getObjectType class-name)}
           {:type :load-local
            :index (into 0)
            :local-type  (Type/getObjectType class-name)}
           {:type :push-int
            :value (int i)}
           {:type :put-field
            :name "tag"
            :owner (Type/getObjectType class-name)
            :field-type Type/INT_TYPE}
           {:type :load-local
            :index (into 0)
            :local-type (Type/getObjectType class-name)}
           {:type :return-value}])
    (.endMethod ^GeneratorAdapter gen)))

(defn make-variant-factories [^ClassWriter writer class-name variants]
  (run! (partial make-variant-factory writer class-name) (map-indexed vector variants)))


(defn make-adt [{:keys [name variants]}]
  (let [class-name name
        writer (ClassWriter. ClassWriter/COMPUTE_FRAMES)]
    (initialize-class writer class-name)
    (generate-default-constructor writer)
    (.visitField writer Opcodes/ACC_PUBLIC "tag" (.getDescriptor Type/INT_TYPE) nil (int 0))
    (.visitEnd writer)
    (make-variant-factories writer class-name variants)
    (.defineClass ^clojure.lang.DynamicClassLoader 
                  (clojure.lang.DynamicClassLoader.)
                  (.replace ^String class-name \/ \.) (.toByteArray ^ClassWriter  writer) nil)
    class-name))



(def adt-example1
  {:type :adt
   :name "Color"
   :variants [{:name "Blue"}
              {:name "Green"}
              {:name "Yellow"}]})

(make-adt adt-example1)

;; Next I need to allow variants to have fields and make classes for them.
;; Then I need to make this factory methods take those classes
;; After that I need to think about constructor/matching, and interop

(.tag (Color/Yellow))

(defn make-struct-field [^ClassWriter writer {:keys [name type]}]
  (.visitField writer Opcodes/ACC_PUBLIC name (.getDescriptor ^Type type) nil nil)
  (.visitEnd writer))

(defn make-struct [{:keys [name fields]}]
  (let [class-name name
        writer (ClassWriter. ClassWriter/COMPUTE_FRAMES)]
    (initialize-class writer class-name)
    (generate-default-constructor writer)
    (run! (partial make-struct-field writer) fields)
    (.defineClass ^clojure.lang.DynamicClassLoader 
                  (clojure.lang.DynamicClassLoader.)
                  (.replace ^String class-name \/ \.) (.toByteArray ^ClassWriter  writer) nil)
    class-name))



(def struct-example1
  {:type :struct
   :name "Point"
   :fields [{:name "x"
             :type Type/INT_TYPE}
            {:name "y"
             :type Type/INT_TYPE}]})

(make-struct struct-example1)





(defn resolve-type [expr]
  (case expr
    :string (Type/getType String)
    :void Type/VOID_TYPE
    :int Type/INT_TYPE
    :long Type/LONG_TYPE
    ;; need to add handling Class
    (throw (ex-info "Unknown type" {:expr expr}))))

(defn build-method [[_ {:keys [args name return]} & code]]
  {:type :fn
   :name name
   :args (mapv (fn [[name type]] {:type (resolve-type type)
                                  :name name}) args)
   :code (build-method-code args code)
   :return-type (resolve-type return)})

(defn build-method-code 
  ([args code]
   (build-method-code args code ()))
  ([args code stack]
   (if (empty? code)
     (mapv :code (reverse stack))
     (recur args
            (rest code)
            (cons
             (let [expr (first code)]
               (case (first expr)
                 :var {:code {:type :var
                              :index (int (second expr))}
                       :type (resolve-type (second (nth args (second expr))))}
                 
                 :get-static-field (let [type (resolve-type-of-field expr)] 
                                     {:code {:type :get-static-field
                                             :owner (Type/getType (Class/forName (second expr)))
                                             :name (nth expr 2)
                                             :result-type type}
                                      :type type})
                 :invoke-virtual (let [{:keys [type args name]} (second expr)] 
                                   {:code
                                    {:type :invoke-virtual
                                     :owner (:type (nth stack args))
                                     :method (Method. name 
                                                      (resolve-type type)
                                                      (into-array Type (map :type (take args stack))))}
                                    :type (resolve-type type)})
                 :return-value {:code {:type :return-value}}))
             stack)))))


(def example
  [:method {:name "add2" :args [['x :string]] :return :void}
   [:get-static-field "java.lang.System" "out"]
   [:var (int 0)]
   [:invoke-virtual {:name "println" :type :void :args 1}]
   [:return-value]])


(make-fn (build-method example))

;; Maybe something like this for our higher level thing
[:method "add2" [x :string] :void
 [:println :java.lang.System/out x :void]]



(add2/invoke "Hello!")

(resolve-type :void)

(def fn-example
  {:type :fn
   :name "add"
   :args [{:type Type/INT_TYPE
           :name 'x}
          {:type (Type/getType String)
           :name 'y}]
   :code [{:type :var
           :index (int 0)}
          {:type :get-static-field
           :owner (Type/getType System)
           :name "out"
           :result-type (Type/getType (class System/out))}
          {:type :var
           :index (int 1)}
          {:type :invoke-virtual
           :owner (Type/getType (class System/out))
           :method (Method. "println" Type/VOID_TYPE (into-array Type [(Type/getType String)]))}
          {:type :var
           :index (int 1)}
          {:type :invoke-static
           :owner (Type/getType Integer)
           :method (Method. "parseInt" Type/INT_TYPE (into-array Type [(Type/getType String)]))}
          {:type :math
           :op GeneratorAdapter/ADD
           :op-type Type/INT_TYPE}
          {:type :return-value}]
   :return-type Type/INT_TYPE})

(def fn-name (make-fn fn-example))

(add/invoke 2 "3")

;; Next step is probably to add some layer on top of this one.
;; Then do the translation.



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

(require '[clj-java-decompiler.core :as decompiler])
(decompiler/disassemble (Color.))
