// Import React
import React from 'react';


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
  ImageSlide,
  Text,
  QuoteSlide,
} from "./library";


// Require CSS
require('normalize.css');

const images = {
  me: require("./images/me.jpg"),
  falcon: require("./images/falcon.jpg"),
};


preloader(images);
require("./langs")




export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Meander"
      subtext="Declarative Explorations at the Limits of FP" />


    <Headline
      color="green"
      textAlign="left"
      text="FP = Better" />



    <Code
      lang="javascript"
      source={`
        let temp = [];
        for (let i = 0; i < arr.length; i ++) {
          temp.push(i * 2)
        }
      `}
     />

     <Code
      lang="javascript"
      source={`
        let temp = [];
        for (let i = 0; i < arr.length; i ++) {
          if (i % 2 === 0) {
            temp.push(i * 2)
          }
        }
      `}
     />

     <Code
      lang="javascript"
      source={`
        let total = 0;
        for (let i = 0; i < arr.length; i ++) {
          if (i % 2 === 0) {
            total += x*2
          }
        }
      `}
     />


    <Code
      lang="clojure"
      source={`
          (reduce + 0 (filter even? (map (parital * 2) coll)))
      `}
     />


    <Code
      lang="clojure"
      source={`
          (->> coll
               (map (parital * 2))
               (filter even?)
               (reduce + 0))
      `}
     />


    <Points title="What Have We Gained?">
      <Point text="More Readable" />
      <Point text="Declarative Approach" />
      <Point text="About What Not How" />
    </Points>


    <Headline
      color="green"
      textAlign="left"
      text="Is This That Much Better?" />


    <Code
      title="Clean Pipeline"
      lang="clojure"
      source={`
          (->> coll
               (map step1)
               (mapcat step2)
               (partition 2)
               (filter foo?)
               (map (partial extend-bar bar))
               (replace-foo-bar))
      `}
     />



    <Headline
      color="blue"
      textAlign="left"
      text="Still Playing Computer" />


    <Points title="The Plan">
      <Point text="Practical Data Manipulation" />
      <Point text="Advanced Features" />
      <Point text="The Future" />
    </Points>


    <TwoColumn
      left={
        <>
         <Headline color="cyan" size={4} textAlign="center" noSlide text="About Me" />
         <Image height={300} src={images.me} />
       </>
      }
      right={
        <>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="Senior Developer - Adzerk" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </>
      }
    />


    <Points title="Meander">
      <Point text="Clojure Library" />
      <Point text="Actively worked on" />
      <Point text="Takes breaking changes seriously" />
    </Points>


    <ImageSlide
      size={4}
      align="center"
      color="blue"
      caps={false}
      title="Joel (noprompt) (falcon) Holdbrooks"
      src={images.falcon}
    />

    <TwoColumn
      title="Example Problem"
      color="magenta"
      align="left"
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name "Jimmy"
            :address
            {:address1 "123 street ave"
             :address2 "apt 2"
             :city "Townville"
             :state "IN"
             :zip "46203"}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name "Jimmy"
             :address {:line1 "123 street ave"
                       :line2 "apt 2"}
             :city-info {:city "Townville"
                         :state "IN"
                         :zipcode "46203"}}
          `}
        />
      } 
    />




    <Code
      lang="clojure"
      source={`
        (let [address (:address person)]
          {:name (:name person)
           :address {:line1 (:address1 address)
                     :line2 (:address2 address)}
           :city-info {:city (:city address)
                       :state (:state address)
                       :zipcode (:zip address)}})

      `}
     />


    <TwoColumn
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name "Jimmy"
            :address
            {:address1 "123 street ave"
             :address2 "apt 2"
             :city "Townville"
             :state "IN"
             :zip "46203"}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name "Jimmy"
             :address {:line1 "123 street ave"
                       :line2 "apt 2"}
             :city-info {:city "Townville"
                         :state "IN"
                         :zipcode "46203"}}
          `}
        />
      } 
    />


    <TwoColumn
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name ?name
            :address
            {:address1 ?address1
             :address2 ?address2
             :city ?city
             :state ?state
             :zip ?zip}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name "Jimmy"
             :address {:line1 "123 street ave"
                       :line2 "apt 2"}
             :city-info {:city "Townville"
                         :state "IN"
                         :zipcode "46203"}}
          `}
        />
      } 
    />


    <TwoColumn
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name ?name
            :address
            {:address1 ?address1
             :address2 ?address2
             :city ?city
             :state ?state
             :zip ?zip}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name ?name
             :address {:line1 ?address1
                       :line2 ?address2}
             :city-info {:city ?city
                         :state ?state
                         :zipcode ?zip}}
          `}
        />
      } 
    />

    <Code
      lang="clojure"
      source={`
        (match person
          {:name ?name
           :address 
           {:address1 ?address1
            :address2 ?address2
            :city ?city
            :state ?state
            :zip ?zip}}
          
          {:name ?name
           :address {:line1 ?address1
                     :line2 ?address2}
           :city-info {:city ?city
                       :state ?state
                       :zipcode ?zip}})

      `}
     />



    <Headline
      textAlign="left"
      text="Pattern Matching On Steroids" />


    <Headline
      color="blue"
      textAlign="left"
      text="Extended Example" />


      <Points title="Requirements">
        <Point text="All Valid Player/Weapon Combinations" />
        <Point text="name, weapon, class, attack power, and all upgrades" />
        <Point text="No third party weapons allowed" />
      </Points>

      <Code
        lang="clojure"
        textSize={20}
        source={`
          {:players [{:name "Jimmer"
                      :class :warrior}
                     {:name "Sir Will"
                      :class :knight}
                     {:name "Dalgrith"
                      :class :rogue}]
           :weapons [{:name :long-sword
                      :allowed-classes #{:warrior :knight}
                      :standard-upgrade :power}
                     {:name :short-sword
                      :allowed-classes #{:warrior :rogue}
                      :standard-upgrade :speed}
                     {:name :unbeatable
                      :allowed-classes #{:rogue}
                      :standard-upgrade :nuclear}]
           :stats {:short-sword {:attack-power 2
                                 :upgrades []}
                   :long-sword {:attack-power 4
                                :upgrades [:reach]}
                   :unbeatable {:attack-power 9001
                                :upgrades [:indestructible]}}
           :third-party #{:unbeatable}}
        `}
      />

      <Code
        lang="clojure"
        title="players"
        source={`
          [{:name "Jimmer"
            :class :warrior}
           {:name "Sir Will"
            :class :knight}
           {:name "Dalgrith"
            :class :rogue}]
       `}
      />


      <Code
        lang="clojure"
        title="Weapons"
        source={`
          [{:name :long-sword
            :allowed-classes #{:warrior :knight}
            :standard-upgrade :power}
           {:name :short-sword
            :allowed-classes #{:warrior :rogue}
            :standard-upgrade :speed}
           {:name :unbeatable
            :allowed-classes #{:rogue}
            :standard-upgrade :nuclear}]
       `}
      />

      <Code
        lang="clojure"
        title="Stats"
        source={`
          {:short-sword {:attack-power 2
                         :upgrades []}
           :long-sword {:attack-power 4
                        :upgrades [:reach]}
           :unbeatable {:attack-power 9001
                        :upgrades [:indestructible]}}
        `}
      />

      <Code
        lang="clojure"
        title="ThirdParty"
        source={`
          #{:unbeatable}
        `}
      />

      <Code
        textSize={22}
        lang="clojure"
        source={`
          ({:name "Jimmer"
            :weapon :long-sword
            :class :warrior
            :attack-power 4
            :upgrades [:reach :power]}
           {:name "Jimmer"
            :weapon :short-sword
            :class :warrior
            :attack-power 2
            :upgrades [:speed]}
           {:name "Sir Will"
            :weapon :long-sword
            :class :knight
            :attack-power 4
            :upgrades [:reach :power]}
           {:name "Dalgrith"
            :weapon :short-sword
            :class :rogue
            :attack-power 2
            :upgrades [:speed]})
        `}
      />


     <Code
       textSize={16}
       lang="clojure"
       source={`
        (defn weapons-for-class [class weapons]
          (filter (fn [{:keys [allowed-classes]}] 
                    (contains? allowed-classes class)) 
                  weapons))

        (defn gather-weapon-info [class {:keys [weapons stats third-party] :as info}]
          (->> weapons
               (weapons-for-class class)
               (filter #(not (contains? third-party (:name %))))
               (map #(assoc % :stats (stats (:name %))))))

        (defn player-with-weapons [{:keys [weapons stats third-party] :as info} player]
          (map (fn [weapon player]
                 {:name (:name player)
                  :weapon (:name weapon)
                  :class (:class player)
                  :attack-power (get-in weapon [:stats :attack-power])
                  :upgrades (conj (get-in weapon [:stats :upgrades])
                                  (get-in weapon [:standard-upgrade]))})
               (gather-weapon-info (:class player) info)
               (repeat player)))

        (defn players-with-weapons [{:keys [players weapons stats third-party] :as info}]
          (mapcat (partial player-with-weapons info) players))


        (players-with-weapons game-info)


      `}
    />


    <Code
      lang="clojure"
      source={`
        (m/search game-info
          {:players (scan {:name ?name
                           :class ?class})
           :weapons (scan {:name ?weapon
                           :allowed-classes #{?class}
                           :standard-upgrade !upgrades})
           :stats {?weapon {:attack-power ?attack-power
                            :upgrades [!upgrades ...]}}
           :third-party (not #{?weapon})}

          {:name ?name
           :weapon ?weapon
           :class ?class
           :attack-power ?attack-power
           :upgrades !upgrades})

      `}
     />


    <Code
      lang="clojure"
      source={`
        (m/rewrite pokemon
          {:itemTemplates (gather {:pokemonSettings
                                   {:pokemonId !pokemon
                                    :form !form
                                    :rarity (not-nil !rarity)
                                    :stats {:as !stats}}})}

          (gather {:pokemon !pokemon 
                   :form !form
                   :rarity !rarity
                   :stats !stats}))
      `}
     />

    <Code
      lang="clojure"
      source={`
        (m/search (parse-js example)
          ($ (or
              {:type "FunctionDeclaration"
               :id {:name ?name}
               :loc ?loc}

              {:type "VariableDeclarator"
               :id {:name ?name}
               :loc ?loc
               :init {:type (or "FunctionExpression" "ArrowFunctionExpression")}}))
          {:name ?name
           :loc ?loc})
      `}
     />

    <Code
      lang="clojure"
      source={`
        (m/rewrite reddit
          {:data
           {:children 
            (gather {:data 
                     {:title !title
                      :permalink !link
                      :preview {:images
                                [{:source {:url !image}} & _]}}})}}

          [:div {:class :container}
           .
           [:div
            [:p [:a {:href (m/app str "https://reddit.com" !link)} 
                 !title]]
            [:img {:src (m/app unescape !image)}]]
           ...])
      `}
     />


    <Code
      lang="clojure"
      source={`

        (def addition*
          (r/rewrite
            (+ Z ?n)      ?n
            (+ ?n Z)      ?n
            (+ ?n (S ?m)) (+ (S ?n) ?m)))

        (addition* '(+ Z Z)) ;; => Z
        (addition* '(+ (S Z) Z)) ;; => (S Z)
        (addition* '(+ (S Z) (S Z))) ;; => (+ (S (S Z)) Z)

      `}
     />

     <Code
      lang="clojure"
      source={`

        (def addition
          (r/until =
            (r/bottom-up 
              (r/attempt addition*))))

        (addition '(+ Z Z)) ;; => Z
        (addition '(+ (S Z) Z)) ;; => (S Z)
        (addition '(+ (S Z) (S Z))) ;; => (S (S Z))

      `}
     />

    <Code
      lang="clojure"
      source={`

        (def fibonacci
          (r/rewrite
            (+ Z ?n) ?n
            (+ ?n Z) ?n
            (+ ?n (S ?m)) (+ (S ?n) ?m)
           
            (fib Z) Z
            (fib (S Z)) (S Z)
            (fib (S (S ?n))) (+ (fib (S ?n)) (fib ?n))))

      `}
     />





    <BlankSlide />

  </Presentation>
