{-# LANGUAGE DeriveFunctor    #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NamedFieldPuns   #-}
{-# LANGUAGE RankNTypes       #-}

module Patient where
import           Control.Monad.Free       (Free (Free, Pure), foldFree,
                                           hoistFree, liftF, retract)
import           Control.Monad.IO.Class   (MonadIO, liftIO)
import           Control.Monad.Loops      (whileM)
import           Control.Monad.State.Lazy as State (MonadState, State, forM,
                                                    gets, liftM, modify,
                                                    runState, runStateT)
import           Control.Monad.Trans      (lift)
import           Data.Map.Strict          as Map (Map, elems, fromList, insert,
                                                  lookup, (!))
import           Data.Maybe               (fromJust)
import           Data.Time.Clock          (NominalDiffTime, addUTCTime,
                                           diffUTCTime, nominalDay)
import           Data.Time.Format         (defaultTimeLocale, parseTimeM)
import           Data.Time.LocalTime      (ZonedTime, getCurrentTimeZone,
                                           getZonedTime, utcToZonedTime,
                                           zonedTimeToUTC)
import           Text.Pretty.Simple       (pPrint)

import           System.CPUTime           (getCPUTime)

data Patient = Patient {
  patientId :: Int
} deriving Show

instance Prompt Patient where
  prompt = do
    liftIO $ putStr "Enter Patient Id: "
    patientId <- liftIO getLine
    return $ Patient { patientId = read patientId :: Int}

data Medication = Medication {
  medicationId   :: Int,
  nextRefillDate :: ZonedTime
} deriving Show

instance Prompt Medication where
  prompt = do
    liftIO $ putStr "Enter Medication Id: "
    medId <- liftIO getLine
    liftIO $ putStr "Enter Next Refill Date: "
    refillDate <- liftIO getLine
    return $ Medication {
      medicationId = read medId :: Int,
      nextRefillDate = getZonedDate refillDate
    }

data Configuration = Configuration {
  daysToSchedule :: NominalDiffTime,
  daysToRefill   :: NominalDiffTime
} deriving Show

instance Prompt Configuration where
  prompt = do
    liftIO $ putStr "Enter Days To Schedule: "
    toSchedule <- liftIO getLine
    liftIO $ putStr "Enter Days To Refill: "
    toRefill <- liftIO getLine
    return Configuration {
      daysToSchedule = fromInteger (read toSchedule :: Integer),
      daysToRefill = fromInteger (read toRefill :: Integer)
    }

newtype EffectiveMoment = EffectiveMoment ZonedTime deriving Show

instance Prompt EffectiveMoment where
  prompt = do
    liftIO $ putStr "Enter Effective Moment: "
    moment <- liftIO getLine
    return $ EffectiveMoment (getZonedDate moment)

data ClinicalF next
  = GetPatient (Patient -> next)
  | GetOrganizationConfiguration (Configuration -> next)
  | GetMedications Patient ([Medication] -> next)
  | GetEffectiveMoment (EffectiveMoment -> next)
  | RecommendMedications [Medication] next deriving Functor

type Clinical = Free ClinicalF

actionName :: ClinicalF next -> String
actionName (GetPatient _)                   = "GetPatient"
actionName (GetOrganizationConfiguration _) = "GetOrganizationConfiguration"
actionName (GetMedications _ _)             = "GetMedications"
actionName (GetEffectiveMoment _)           = "GetEffectiveMoment"
actionName (RecommendMedications _ _)       = "RecommendMedications"

getPatient :: Clinical Patient
getPatient = liftF $ GetPatient id

getOrganizationConfiguration :: Clinical Configuration
getOrganizationConfiguration = liftF $ GetOrganizationConfiguration id

getMedications :: Patient -> Clinical [Medication]
getMedications patient = liftF $ GetMedications patient id

recommendMedications :: [Medication] -> Clinical ()
recommendMedications meds = liftF $ RecommendMedications meds ()

getEffectiveMoment :: Clinical EffectiveMoment
getEffectiveMoment = liftF $ GetEffectiveMoment id

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
testInterpreter (GetEffectiveMoment f) = do
  moment <- gets effectiveMoment
  return $ f moment
testInterpreter (RecommendMedications meds next) = do
  recs <- gets medRecommendations
  modify $ \x -> (x {medRecommendations = recs ++ meds})
  return next


getZonedDate :: String -> ZonedTime
getZonedDate date = fromJust $ parseTimeM True defaultTimeLocale "%Y-%-m-%-d" date

data TestEntities = TestEntities {
  currentPatient            :: Patient,
  organizationConfiguration :: Configuration,
  patientMedication         :: Map Int [Medication],
  medRecommendations        :: [Medication],
  effectiveMoment           :: EffectiveMoment
} deriving Show

med1 = Medication {
  medicationId = 1,
  nextRefillDate = getZonedDate "2018-06-03"
}

scenario1 = TestEntities {
  currentPatient = Patient { patientId = 1 },
  organizationConfiguration = Configuration { daysToSchedule = -90, daysToRefill = 5 },
  patientMedication = Map.fromList [(1, [med1])],
  medRecommendations = [],
  effectiveMoment = EffectiveMoment $ getZonedDate "2018-06-03"
}

class Prompt a where
  prompt :: IO a

consoleInterpreter :: (MonadIO m) => ClinicalF next -> m next
consoleInterpreter (GetPatient f) = liftIO $ f <$> prompt
consoleInterpreter (GetOrganizationConfiguration f) = liftIO $ f <$> prompt
consoleInterpreter (GetEffectiveMoment f) = liftIO $ f <$> prompt
consoleInterpreter (GetMedications patient f) = liftIO $ f <$> prompt
consoleInterpreter (RecommendMedications meds next) = do
  liftIO $ pPrint meds
  return next

instance Prompt Int where
  prompt = do
    putStr "Give me a number: "
    line <- getLine
    return (read line :: Int)


askIfFinished :: IO Bool
askIfFinished = do
  putStr "Are you finished (y/n): "
  finished <- getLine
  return $ finished == "y"

instance Prompt a => Prompt [a] where
  prompt = do
    p <- prompt
    finished <- askIfFinished
    if finished then
      return [p]
    else do
      ps <- prompt
      return $ p : ps

withinXDaysFromNow :: NominalDiffTime -> ZonedTime -> ZonedTime -> Bool
withinXDaysFromNow days date1 date2 = dayDiff <= days where
  d1 = zonedTimeToUTC date1
  d2 = zonedTimeToUTC date2
  dayDiff = diffUTCTime d2 d1 / nominalDay

runningOut :: [Medication] -> Clinical [Medication]
runningOut meds = do
  EffectiveMoment effectiveMoment <- getEffectiveMoment
  Configuration { daysToRefill } <- getOrganizationConfiguration
  return $ filter (withinXDaysFromNow daysToRefill effectiveMoment
                   . nextRefillDate) meds

getMedRecommendations :: Clinical ()
getMedRecommendations = do
  patient <- getPatient
  medications <- getMedications patient
  refills <- runningOut medications
  recommendMedications refills

chooseInterpreter :: (MonadIO m, MonadState TestEntities m) => (ClinicalF next -> m next) -> (ClinicalF next -> m next) -> ClinicalF next -> m next
chooseInterpreter interpret1 interpret2 action = do
  liftIO $ putStrLn ("About to preform " ++ actionName action)
  liftIO $ putStr "do you want to overide? (y/n) "
  override <- liftIO getLine
  if override == "y" then
    interpret2 action
  else
    interpret1 action

runProgram :: (Monad m) => (forall x. f x -> m x) -> Free f a -> m a
runProgram = foldFree


main :: IO ()
main = do
  result <- runStateT (runProgram (chooseInterpreter testInterpreter consoleInterpreter)
                                  getMedRecommendations)
                       scenario1
  pPrint result
  return ()
