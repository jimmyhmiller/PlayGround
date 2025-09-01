# Clojure Conventions

## Namespaces and Requires
- Always put all requires in the `ns` declaration at the top of the file
- Never use local `require` statements inside functions
- Use `:as` aliases for namespaces to avoid repetition

## Example
```clojure
;; GOOD
(ns my-app.core
  (:require
   [other.namespace :as other]))

(defn my-function []
  (other/some-function))

;; BAD
(defn my-function []
  (require '[other.namespace :as other])
  (other/some-function))
```