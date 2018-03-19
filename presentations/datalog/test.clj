

(require '[clojure.spec.alpha :as s])
(require '[clojure.test.check.generators])
(require '[clojure.spec.test.alpha :as stest])

(s/def ::country #{"US"})
(s/def ::end-date nil?)
(s/def ::begin-date nil?)
(s/def ::primary nil?)
(s/def ::type (s/nilable #{"Group"}))
(s/def ::locale nil?)
(s/def ::name string?)
(s/def ::sort-name string?)
(s/def
 ::aliases
 (s/coll-of
  (s/keys
   :req-un
   [::begin-date
    ::end-date
    ::locale
    ::name
    ::primary
    ::sort-name
    ::type]) :max-count 3))
(s/def ::id string?)
(s/def ::area (s/keys :req-un [::id ::name ::sort-name]))
(s/def ::begin-area (s/keys :req-un [::id ::name ::sort-name]))
(s/def ::score string?)
(s/def ::disambiguation string?)
(s/def ::end string?)
(s/def ::begin string?)
(s/def ::ended (s/nilable boolean?))
(s/def ::life-span (s/keys :req-un [::ended] :opt-un [::begin ::end]))
(s/def ::count pos-int?)
(s/def ::tags (s/coll-of (s/keys :req-un [::count ::name]) :max-count 3))
(s/def
 ::artists
 (s/coll-of
  (s/keys
   :req-un
   [::area
    ::country
    ::id
    ::life-span
    ::name
    ::score
    ::sort-name
    ::type]
   :opt-un
   [::aliases ::begin-area ::disambiguation ::tags]) :max-count 3))
(s/def ::offset integer?)
(s/def ::created string?)
(s/def ::artist (s/keys :req-un [::artists ::count ::created ::offset]))


(s/exercise ::artist 10)
