
import React from "react";
import { 
  Presentation,
  Code,
  Headline,
  TwoColumn,
  Point,
  Points,
  preloader,
  BlankSlide,
  Image,
  Text,
  formatCode,
} from "./library";

import CodeSlide from 'spectacle-code-slide';

const images = {
  me: require("./images/me.jpg"),
};

preloader(images);

export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      subtextSize={3}
      subtextCaps={false}
      text="Programs as Values"
      subtext="Fun with Free Monads" />


    <TwoColumn
      title="About Me"
      left={<Image src={images.me} />}
      right={
        <div style={{paddingTop: 80}}>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="Senior Developer - healthfinch" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </div>
      } 
    />

    <Headline
      textAlign="left"
      subtextSize={3}
      subtextCaps={false}
      text="Programs as Values"
      subtext="Fun with Free Monads" />

    <Points title="Plan">
      <Point text="Talk about problems we all face" />
      <Point text="Talk about solutions" />
      <Point text="Show a free monad solution" />
      <Point text="Work from ground up to understanding" />
    </Points>

    <Headline
      color="green"
      text="How do we write business logic?" />

    <Headline
      color="blue"
      text="How do we allow our programs to change behavior?" />

    <Headline
      color="green"
      text="How do we make our code testable?" />

    <Headline
      color="cyan"
      text="How do we implement generic functionality?" />

    <Code
      title="Example"
      lang="java"
      source={`
        public void getMedRecommendations() {
            Patient patient = getPatient();
            List<Medication> medications = getMedications(patient);
            List<Medication> refills = runningOut(medications);
            recommendMedications(refills);
        }
      `} />

    <Code
      title="Example"
      lang="java"
      source={`
        public List<Medication> runningOut(List<Medication> meds) {
            EffectiveMoment effectiveMoment = getEffectiveMoment();
            Configuration configuration = getOrganizationConfiguration();
            Days daysToRefill = configuration.daysToRefill;
            return meds
              .filter(med => withinXDaysFromNow(daysToRefill, 
                                                effectiveMoment, 
                                                med.nextRefillDate));
        }
      `} />

    <Points title="Feature Requests">
      <Point text="Supporting a new platform" />
      <Point text="Need to keep a metric of recommendations" />
      <Point text="Get configuration from cache if possible" />
      <Point text="We want better test coverage" />
      <Point text="Need to test version 2.0 of existing platform" />
    </Points>

    <Points title="Implementation possibilities">
      <Point text="Conditionals" />
      <Point text="Map to common format" />
      <Point text="Dependency Injection" />
    </Points>

    <Code
      title="Interface"
      lang="java"
      source={`
        interface IClinicalService {
            Patient getPatient();
            List<Medication> getMedications(Patient patient);
            EffectiveMoment getEffectiveMoment();
            Configuration getOrganizationConfiguration();
            void recommendMedications(List<Medication> meds);
        }
      `} />

    <Code
      title="Example"
      lang="java"
      source={`
        public void getMedRecommendations() {
            Patient patient = this.clinicalService.getPatient();
            List<Medication> medications = this.clinicalService
                                               .getMedications(patient);
            List<Medication> refills = runningOut(medications);
            this.clinicalService.recommendMedications(refills);
        }
      `} />

    <Points title="Questions">
      <Point text="How do ensure all implementations record metrics?" />
      <Point text="How do we know what instance was used?" />
      <Point text="How do we use more than one instance at a time?" />
      <Point text="Are these thread safe?" />
    </Points>

    <Headline
      color="cyan"
      text="Fun with Free Monads" />

    <Code
      title="Example"
      lang="haskell"
      source={`
        getMedRecommendations :: Clinical ()
        getMedRecommendations = do
          patient <- getPatient
          medications <- getMedications patient
          refills <- runningOut medications
          recommendMedications refills
      `} />

    <Code
      title="Example"
      lang="haskell"
      source={`
        runningOut :: [Medication] -> Clinical [Medication]
        runningOut meds = do
          EffectiveMoment effectiveMoment <- getEffectiveMoment
          Configuration { daysToRefill } <- getOrganizationConfiguration
          return $ filter (withinXDaysFromNow daysToRefill effectiveMoment
                           . nextRefillDate) meds
      `} />


    <Headline
      color="green"
      text="This code does nothing" />

    <Headline
      color="blue"
      text="It describes what needs to be done" />

    <Headline text="Http Example" />


    <Code
      lang="haskell"
      title="Requests as Data"
      source={`
        type Url = String
        type Body = String

        data Http
          = Get Url
          | Post Url Body
      `} />

    <Code
      title="List of Requests"
      lang="haskell"
      source={`
        requests :: [Http]
        requests = [Get "google.com", 
                    Post "twitter.com" "Fun with Free Monads"]
      `} />

    <Code
      title="List of Requests"
      lang="haskell"
      source={`
        fetch :: Http -> IO ()
        fetch (Get url)       = putStrLn $ "Get " ++ url
        fetch (Post url body) = putStrLn $ "Post " ++ url ++ ", " ++ body
      `} />

    <Code
      title="List of Requests"
      lang="haskell"
      source={`
        main :: IO ()
        main =
          for_ requests $ \\request ->
            fetch request
      `} />


    <Points title="What we have">
      <Point text="Program is a value" />
      <Point text="We can interpret our program differently" />
      <Point text="Testing our program is just interpreting" />
    </Points>


    <Points title="Very Limited">
      <Point text="How do we get the return value?" />
      <Point text="What if we want to branch?" />
      <Point text="What if we want to loop?" />
    </Points>

    <Headline
      color="blue"
      text="Things are going to get more complicated" />

    <Code
      lang="haskell"
      title="Recursive Data Type"
      source={`
        data Http
          = Get Url Http
          | Post Url Body Http
          | Done

        instructions :: Http
        instructions =
          Get "google.com" (
            Post "twitter.com" "Fun with Free Monads" Done
          )
      `} />


    <Code
      lang="haskell"
      title="Recursive Data Type"
      source={`
        data Http
          = Get Url Http
          | Post Url Body Http
          | Done

        instructions :: Http
        instructions =
          Get "google.com" $
          Post "twitter.com" "Fun with Free Monads"
          Done

      `} />

    <Code
      lang="haskell"
      title="Interpret Single Level"
      source={`
        fetch :: Http -> IO ()
        fetch (Get url _)       = putStrLn $ "Get " ++ url
        fetch (Post url body _) = putStrLn $ "Post " ++ url ++ ", " ++ body
        fetch Done              = return ()
      `} />


    <Code
      lang="haskell"
      title="Run in order"
      source={`
        run :: (Http -> IO ()) -> Http -> IO ()
        run f h@(Get _ next)    = f h >> run f next
        run f h@(Post _ _ next) = f h >> run f next
        run f Done              = return ()

        main :: IO ()
        main = run fetch instructions
      `} />


    <Code
      lang="haskell"
      title='Abstracting "next"'
      source={`
        data HttpF next
          = Get Url next
          | Post Url Body next
          | Done

        -- examples
        httpInt :: HttpF Int
        httpInt = Get "int.com" 123

        httpString :: HttpF String
        httpString = Get "string.com" "String"
      `} />

    <Code
      lang="haskell"
      title="Gaining Generic Functionality"
      source={`
        instance Functor HttpF where
          fmap f (Get url next)       = Get url $ f next
          fmap f (Post url body next) = Post url body $ f next
          fmap f Done                 = Done
      `} />

    <Code
      lang="haskell"
      title="Tying the knot"
      source={`
        http1 :: HttpF (HttpF next)
        http1 = Get "google.com" Done

        http2 :: HttpF (HttpF (HttpF next))
        http2 = Get "twitter.com" $
                Post "google.com" "Fun with Free Monads"
                Done

      `} />

    <Code
      lang="haskell"
      title="Tying the knot"
      source={`
        data Fix f = Fix (f (Fix f))
      `} />

    <Code
      lang="haskell"
      title="Tying the knot"
      source={`

        type Http = Fix HttpF

        http1 :: Http
        http1 = Fix $ Get "google.com" $ Fix Done

        http2 :: Http
        http2 = Fix $ Get "twitter.com" $
                Fix $ Post "google.com" "Fun with Free Monads" $
                Fix Done
      `} />

    <Code
      lang="haskell"
      title="Automatic recursion"
      source={`
        toString :: HttpF String -> String
        toString (Get url next)  = "Get " ++ url ++ "\\n" ++ next
        toString (Post u b next) = "Post " ++ u ++ ", " ++ b ++ "\\n" ++ next
        toString Done            = ""
      `} />

    <Code
      lang="haskell"
      title="Automatic recursion"
      source={`
        fetch :: HttpF (IO ()) -> IO ()
        fetch (Get url next)  = putStrLn ("Get " ++ url) >> next
        fetch (Post u b next) = putStrLn ("Post " ++ u ++ ", " ++ b) >> next
        fetch Done            = return ()
      `} />

    <Code
      lang="haskell"
      title="Folding over program"
      source={`
        foldFix :: Functor f => (f a -> a) -> Fix f -> a
        foldFix f (Fix t) = f (fmap (foldFix f) t)

        instructions :: Http
        instructions = Fix $ Get "google.com" $
                       Fix $ Post "twitter.com" "Fun with Free Monads" $
                       Fix Done

        main :: IO ()
        main = foldFix fetch instructions
      `} />

    <Code
      lang="haskell"
      title="Eliminating Done"
      source={`
        data HttpF next
          = Get Url next
          | Post Url Body next
      `} />

    <Code
      lang="haskell"
      title="Free Monad"
      source={`
        data Free f r
          = Free (f (Free f r))
          | Pure r
      `} />

    <Code
      lang="haskell"
      title="Free Monad"
      source={`
        instance (Functor f) => Monad (Free f) where
          return = Pure
          (Free x) >>= f = Free (fmap (>>= f) x)
          (Pure r) >>= f = f r
      `} />

    <Code
      lang="haskell"
      title="Transforming Interpreter"
      source={`
        fetch :: HttpF a -> IO a
        fetch (Get url next) = do
          putStrLn ("Get " ++ url)
          return next
        fetch (Post url body next) = do
          putStrLn ("Post " ++ url ++ ", " ++ body)
          return next
      `} />


    <Code
      lang="haskell"
      title="Helper Functions"
      source={`
        liftF :: Functor f => f r -> Free f r
        liftF x = Free (fmap Pure x)

        get :: Url -> Http ()
        get url = liftF $ Get url ()

        post :: Url -> Body -> Http ()
        post url body = liftF $ Post url body ()
      `} /> 

    <Code
      lang="haskell"
      title="Folding Free"
      source={`
        foldFree :: Monad m => (forall x . f x -> m x) -> Free f a -> m a
        foldFree _ (Pure a)  = return a
        foldFree f (Free as) = f as >>= foldFree f

        instructions :: Http ()
        instructions = do
           get "google.com"
           post "twitter.com" "Fun with Free Monads"

        main :: IO ()
        main = foldFree fetch instructions
      `} /> 

    <Code
      lang="haskell"
      title="Callbacks"
      source={`
        data HttpF next
          = Get Url (String -> next)
          | Post Url Body (String -> next)
      `} /> 

    <Code
      lang="haskell"
      title="Putting it all together"
      source={`
        instructions :: Http String
        instructions = do
           x <- get "google.com"
           if x == "twitter" then
             post "twitter.com" "Fun with Free Monads"
           else
             post "facebook.com" "Fun with Free Monads"
      `} />

    <Code
      lang="haskell"
      title="Putting it all together"
      source={`
        fetch :: HttpF a -> IO a
        fetch (Get url f) = do
          putStrLn ("Get " ++ url)
          result <- getLine
          putStrLn $ "Result of get: " ++ result
          return $ f result
        fetch (Post url body f) = do
          putStrLn ("Post " ++ url ++ ", " ++ body)
          result <- getLine
          putStrLn $ "Result of post: " ++ result
          return $ f result
      `} />

    <Code
      lang="haskell"
      title="Original Example"
      source={`
        data ClinicalF a
          = GetPatient (Patient -> a)
          | GetOrganizationConfiguration (Configuration -> a)
          | GetMedications Patient ([Medication] -> a)
          | GetEffectiveMoment (EffectiveMoment -> a)
          | RecommendMedications [Medication] a deriving Functor
      `} />

    <Points title="Feature Requests">
      <Point text="Supporting a new platform" />
      <Point text="Need to keep a metric of recommendations" />
      <Point text="Get configuration from cache if possible" />
      <Point text="We want better test coverage" />
      <Point text="Need to test version 2.0 of existing platform" />
    </Points>


    <Headline text="Over Simplified Sketches" />

    <Code
      lang="haskell"
      title="Supporting a new platform"
      source={`
        customer1Interpreter :: Interpreter a
        customer2Interpreter :: Interpreter a
        customer3Interpreter :: Interpreter a
      `} />

    <Code
      lang="haskell"
      title="Need to keep a metric of recommendations"
      source={`
        metricInterpreter :: Interpreter a -> ClinicalF a -> Result a
        metricInterpreter interpret r@(RecommendMedications meds next) = do
            recordCount "medRecommendations" (length meds)
            interpret r
        metricInterpreter interpreter r = interpreter r
      `} />

    <Code
      lang="haskell"
      title="Get configuration from cache if possible"
      source={`
        cacheInterpreter :: Cache -> Interpreter a -> ClinicalF a -> Result a
        cacheInterpreter cache interpret r@(GetOrganizationConfiguration f) = do
          cached <- get cache configuration
          case cached of
            Just val -> interpret (f val)
            Nothing -> interpret r
        cacheInterpreter cache interpreter r = interpreter r
      `} />

    <Code
      lang="haskell"
      title="We want better test coverage"
      source={`
        testInterpreter :: ClinincalF a -> State TestEntities a

        scenario1 = TestEntities {
          currentPatient = Patient { patientId = 1 },
          organizationConfiguration = Configuration { daysToSchedule = -90, 
                                                      daysToRefill = 5 },
          patientMedication = Map.fromList [(1, [med1])],
          medRecommendations = [],
          effectiveMoment = EffectiveMoment $ getZonedDate "2018-06-03"
        }

        runState (runProgram testInterpreter getMedRecommendations) 
                 scenario1
      `} />

    <Code
      lang="haskell"
      title="Need to test version 2.0 of existing platform"
      source={`
        dualInterpreter :: Interpreter a -> Interpreter a -> ClinincalF a -> Result a
        dualInterpreter interpret1 interpret2 clinical = do
          interpret1 clinical
          interpret2 clinical

        canaryInterpreter = dualInterpreter safeInterpreter2 realInterpreter
      `} />


    <Points title="Other Features">
      <Point text="Combine Multiple Languages" />
      <Point text="Compile one language into another" />
      <Point text="Free applicatives give us automatic concurrency" />
      <Point text="Pause/Replay programs at a given point" />
    </Points>

    <Points title="Taking Further - Extensible Effects">
      <Point text="Used by Idris" />
      <Point text="Used by React with hooks" />
      <Point text="Used by Unison for remote execution" />
    </Points>




    <BlankSlide />

  </Presentation>
