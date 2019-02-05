(ns data-cube.core
  (:import [java.util.concurrent ConcurrentHashMap]
           [com.urbanairship.datacube DbHarness$CommitType IdService Dimension Rollup]
           [com.urbanairship.datacube.idservices
            HBaseIdService
            MapIdService
            CachingIdService]
           [com.urbanairship.datacube.dbharnesses
            MapDbHarness]
           [com.urbanairship.datacube.ops
            LongOp
            IntOp]
           [com.urbanairship.datacube.bucketers StringToBytesBucketer HourDayMonthBucketer BigEndianIntBucketer BigEndianLongBucketer BooleanBucketer TagsBucketer ]))


(def id-service (CachingIdService. 5 ^IdService (MapIdService.) "id-service"))

(def backingMap (ConcurrentHashMap.))

(def dbHarness (MapDbHarness. backingMap LongOp/DESERIALIZER DbHarness$CommitType/READ_COMBINE_CAS id-service))

(def hour-day-month-bucketer (HourDayMonthBucketer.))

(def time (Dimension. "time" hour-day-month-bucketer false 8))

(def zipcode (Dimension. "zipcode" (StringToBytesBucketer.) true 5))

(def hourAndZipRollup (Rollup. zipcode time HourDayMonthBucketer/hours))
(def dayAndZipRollup (Rollup. zipcode time HourDayMonthBucketer/days))
(def hourRollup (Rollup. time HourDayMonthBucketer/hours))
(def dayRollup (Rollup. time HourDayMonthBucketer/days))
