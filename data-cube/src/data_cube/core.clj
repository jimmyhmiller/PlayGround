(ns data-cube.core
  (:import [java.util.concurrent ConcurrentHashMap]
           [com.urbanairship.datacube
            DbHarness$CommitType
            IdService Dimension
            Rollup DataCube
            DataCubeIo
            SyncLevel
            WriteBuilder 
            ReadBuilder]
           [com.urbanairship.datacube.idservices
            HBaseIdService
            MapIdService
            CachingIdService]
           [com.urbanairship.datacube.dbharnesses
            MapDbHarness]
           [com.urbanairship.datacube.ops
            LongOp
            IntOp]
           [com.urbanairship.datacube.bucketers 
            StringToBytesBucketer
            HourDayMonthBucketer
            BigEndianIntBucketer
            BigEndianLongBucketer
            BooleanBucketer
            TagsBucketer]
           [org.joda.time DateTime DateTimeZone]))


;; https://github.com/urbanairship/datacube

(def id-service (CachingIdService. 5 ^IdService (MapIdService.) "id-service"))

(def backingMap (ConcurrentHashMap.))

(def dbHarness (MapDbHarness. backingMap LongOp/DESERIALIZER DbHarness$CommitType/READ_COMBINE_CAS id-service))

(def hour-day-month-bucketer (HourDayMonthBucketer.))

(def time (Dimension. "time" hour-day-month-bucketer false 8))

(def zipcode (Dimension. "zipcode" (StringToBytesBucketer.) true 5))

(def keywords (Dimension. "keywords" (TagsBucketer.) true 7))
(def keywords2 (Dimension. "keywords2" (TagsBucketer.) true 7))

(def hourAndZipRollup (Rollup. zipcode time HourDayMonthBucketer/hours))
(def dayAndZipRollup (Rollup. zipcode time HourDayMonthBucketer/days))
(def hourRollup (Rollup. time HourDayMonthBucketer/hours))
(def dayRollup (Rollup. time HourDayMonthBucketer/days))
(def hourAndZipRollup (Rollup. zipcode time HourDayMonthBucketer/hours))
(def zipRollup (Rollup. zipcode))
(def zipKeywordsRollUp (Rollup. zipcode keywords))
(def keywordsRollup (Rollup. keywords))
(def keywordsKeywords2Rollup (Rollup. keywords keywords2))


(def dimensions [time zipcode keywords keywords2])

(def rollups [hourAndZipRollup dayAndZipRollup hourRollup dayRollup zipRollup zipKeywordsRollUp
              keywordsKeywords2Rollup keywordsRollup])

(def cube (DataCube. dimensions rollups))

(def cubeIo (DataCubeIo. cube dbHarness 1 Long/MAX_VALUE SyncLevel/FULL_SYNC "Thing" false))


(def now (DateTime. DateTimeZone/UTC))


(.writeSync cubeIo (LongOp. 10)
            (.. (WriteBuilder.)
                (at time now)
                (at zipcode "46203")
                (at keywords ["a" "b" "c"])
                (at keywords2 ["a" "b" "c"])))

(def differentHour (.withHourOfDay now (mod (inc (.getHourOfDay now)) 24)))

(.writeSync cubeIo (LongOp. 10) 
            (.. (WriteBuilder.)
                (at time differentHour)
                (at zipcode "46203")
                (at keywords ["a" "b"])
                (at keywords2 ["a" "b"])))


(defn cardinality [cube a]
  (.getLong
   (.get (.get cubeIo (.. (ReadBuilder. cube)
                          (at keywords a))))))

(defn intersection [cube a b]
  (.getLong
   (.get (.get cubeIo (.. (ReadBuilder. cube)
                          (at keywords a)
                          (at keywords2 b))))))

(defn union [cube a b]
  (- (+ (cardinality cube a) (cardinality cube b)) (intersection cube a b)))



(intersection cube "a" "c")

(union cube "a" "b")


(.getLong
 (.get (.get cubeIo (.. (ReadBuilder. cube)
                        (at keywords "a")
                        (at keywords2 "c")))))


(.get cubeIo (.. (ReadBuilder. cube)
                 (at keywords "b")
                 (at keywords2 "c")))


(.get cubeIo (.. (ReadBuilder. cube)
                 (at keywords "a")
                 (at keywords2 "b")))


(.get cubeIo (.. (ReadBuilder. cube)
                 (at keywords "a")))

(.get cubeIo (.. (ReadBuilder. cube)
                 (at keywords "b")))

(.get cubeIo (.. (ReadBuilder. cube)
                 (at keywords "c")))


(.get cubeIo (.. (ReadBuilder. cube)
                 (at keywords "a")))

