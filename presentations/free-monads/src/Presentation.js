
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
      <Point text="Work backwards to understanding" />
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
      <Point text="Cache all configuration (regardless of implementation)" />
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
      <Point text="When looking at code how do we know what instance was used?" />
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



    <BlankSlide />

  </Presentation>
