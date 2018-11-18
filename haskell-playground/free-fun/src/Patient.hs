{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleContexts #-}

module Patient where
import Control.Monad.Free (Free, liftF, foldFree, hoistFree)
import Data.Time.LocalTime (ZonedTime, zonedTimeToUTC, getCurrentTimeZone, utcToZonedTime, getZonedTime)
import Data.Time.Clock (diffUTCTime, nominalDay, NominalDiffTime, addUTCTime)
import Data.Map.Strict as Map (Map, lookup, (!), fromList, insert, elems)
import Control.Monad.State.Lazy as State (MonadState, State, gets, modify, runState, forM, liftM, runStateT)
import Data.Time.Format (parseTimeM, defaultTimeLocale)
import Data.Maybe (fromJust)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans (lift)

data Patient = Patient {
  patientId :: Int
} deriving Show

data Medication = Medication {
  medicationId :: Int,
  nextRefillDate :: ZonedTime
} deriving Show

data OfficeVisit = OfficeVisit {
  officeVisitId :: Int,
  visitDate :: ZonedTime
} deriving Show

data Configuration = Configuration {
  daysToSchedule :: NominalDiffTime,
  daysToRefill :: NominalDiffTime
} deriving Show

data ClinicalF next
  = GetPatient (Patient -> next)
  | GetOrganizationConfiguration (Configuration -> next)
  | GetMedications Patient ([Medication] -> next)
  | GetOfficeVisits Patient ([OfficeVisit] -> next)
  | GetEffectiveMoment (ZonedTime -> next)
  | RecommendMedications [Medication] next
  | RecommendOfficeVisits [OfficeVisit] next deriving Functor

type Clinical = Free ClinicalF


getPatient :: Clinical Patient
getPatient = liftF $ GetPatient id

getOrganizationConfiguration :: Clinical Configuration
getOrganizationConfiguration = liftF $ GetOrganizationConfiguration id

getMedications :: Patient -> Clinical [Medication]
getMedications patient = liftF $ GetMedications patient id

getOfficeVisits :: Patient -> Clinical [OfficeVisit]
getOfficeVisits patient = liftF $ GetOfficeVisits patient id

recommendMedications :: [Medication] -> Clinical ()
recommendMedications meds = liftF $ RecommendMedications meds ()

recommendOfficeVisits :: [OfficeVisit] -> Clinical ()
recommendOfficeVisits visits = liftF $ RecommendOfficeVisits visits ()

getEffectiveMoment :: Clinical ZonedTime
getEffectiveMoment = liftF $ GetEffectiveMoment id

withinXDaysFromNow :: NominalDiffTime -> ZonedTime -> ZonedTime -> Bool
withinXDaysFromNow days date1 date2 = dayDiff <= days where
  d1 = zonedTimeToUTC date1
  d2 = zonedTimeToUTC date2
  dayDiff = diffUTCTime d2 d1 / nominalDay

withinXDaysAgo :: NominalDiffTime -> ZonedTime -> ZonedTime -> Bool
withinXDaysAgo days date1 date2 = dayDiff >= days where
  d1 = zonedTimeToUTC date1
  d2 = zonedTimeToUTC date2
  dayDiff = diffUTCTime d2 d1 / nominalDay

runningOut :: [Medication] -> Clinical [Medication]
runningOut meds = do
  effectiveMoment <- getEffectiveMoment
  Configuration { daysToRefill = daysToRefill } <- getOrganizationConfiguration
  return $ filter (withinXDaysFromNow daysToRefill effectiveMoment . nextRefillDate) meds

overdueVisits :: [OfficeVisit] -> Clinical [OfficeVisit]
overdueVisits visits = do
  effectiveMoment <- getEffectiveMoment
  Configuration { daysToSchedule = daysToSchedule } <- getOrganizationConfiguration
  return $ filter (not . withinXDaysAgo daysToSchedule effectiveMoment . visitDate) visits

getPatientRecommendations :: Clinical ()
getPatientRecommendations = do
    patient <- getPatient

    medications <- getMedications patient
    refills  <- runningOut medications
    recommendMedications refills

    officeVisits <- getOfficeVisits patient
    appointments <- overdueVisits officeVisits
    recommendOfficeVisits appointments


data TestEntities = TestEntities {
  currentPatient :: Patient,
  organizationConfiguration :: Configuration,
  patientMedication :: Map Int [Medication],
  patientOfficeVisits :: Map Int [OfficeVisit],
  medRecommendations :: [Medication],
  visitRecommendations :: [OfficeVisit],
  effectiveMoment :: ZonedTime
} deriving Show

testInterpreter :: (MonadState TestEntities m) => ClinicalF next -> m next
testInterpreter (GetPatient f) = do
  patient <- gets currentPatient
  return $ f patient
testInterpreter (GetOrganizationConfiguration f) = do
  configuration <- gets organizationConfiguration
  return $ f configuration
testInterpreter (GetMedications patient f) = do
  (Just meds) <- gets (Map.lookup (patientId patient) . patientMedication)
  return $ f meds
testInterpreter (GetOfficeVisits patient f) = do
  (Just visits) <- gets (Map.lookup (patientId patient) . patientOfficeVisits)
  return $ f visits
testInterpreter (GetEffectiveMoment f) = do
  moment <- gets effectiveMoment
  return $ f moment
testInterpreter (RecommendMedications meds next) = do
  recs <- gets medRecommendations
  modify $ \x -> (x {medRecommendations = recs ++ meds})
  return next
testInterpreter (RecommendOfficeVisits visits next) = do
  recs <- gets visitRecommendations
  modify $ \x -> (x {visitRecommendations = recs ++ visits})
  return next


getZonedDate :: String -> ZonedTime
getZonedDate date = fromJust $ parseTimeM True defaultTimeLocale "%Y-%-m-%-d" date

med1 = Medication {
  medicationId = 1,
  nextRefillDate = getZonedDate "2018-06-03"
}

visit1 = OfficeVisit {
  officeVisitId = 3,
  visitDate = getZonedDate "2018-06-03"
}

scenario1 = TestEntities {
  currentPatient = Patient { patientId = 1 },
  organizationConfiguration = Configuration { daysToSchedule = -90, daysToRefill = 5 },
  patientMedication = Map.fromList [(1, [med1])],
  patientOfficeVisits = Map.fromList [(1, [visit1])],
  medRecommendations = [],
  visitRecommendations = [],
  effectiveMoment = getZonedDate "2018-06-03"
}

overriderInterpreter :: (MonadIO m) => (ClinicalF next -> m next) -> ClinicalF next -> m next
overriderInterpreter interpreter (GetPatient f) = do
  liftIO $ putStr "Enter Patient Id: "
  patientId <- liftIO $ getLine
  let patient = Patient { patientId = read patientId :: Int}
  return $ f patient
overriderInterpreter interpreter x =
  interpreter x

q :: (MonadIO m, MonadState TestEntities m) => ClinicalF next -> m next
q = overriderInterpreter testInterpreter

z :: (MonadIO m, MonadState TestEntities m) => m ()
z = foldFree q getPatientRecommendations


main :: IO ()
main = do
  result <- runStateT (foldFree (overriderInterpreter testInterpreter) getPatientRecommendations) scenario1
  putStr $ show result
  return ()
