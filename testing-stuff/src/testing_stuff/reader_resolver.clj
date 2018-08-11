(ns testing-stuff.reader-resolver
  (:import (clojure.lang Compiler)))

(def aliases (atom {}))

(defn add-alias [alias sym]
  (swap! aliases assoc alias sym))

(def my-reader-resolver
  "Quick POC resolver. Probably not right."
  (reify clojure.lang.LispReader$Resolver
    (currentNS [this] (ns-name *ns*))
    (resolveAlias [this sym] (get @aliases sym (get (ns-aliases (.currentNS this)) sym)))
    (resolveClass [this sym] (when-let [klass (.getMapping (the-ns (.currentNS this)) sym)]
                               (when (class? klass)
                                 (symbol (.getName klass))))) 
    (resolveVar [this sym] (clojure.lang.Symbol/intern (name (.currentNS this)) (name sym)))))

(alter-var-root #'*reader-resolver* (constantly my-reader-resolver))

(add-alias 'user 'com.my-company.really.long.name.user)

::user/thing ; => :com.my-company.really.long.name.user/thing


