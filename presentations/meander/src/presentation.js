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
          (->> coll
               (map (partial * 2))
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
      <Point text="Meander and its Philosophy" />
      <Point text="Taking the Approach Further" />
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
            <Point textSize={40} text="Senior Developer - Adzerk" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </>
      }
    />


    <Points title="Meander">
      <Point text="Clojure Library" />
      <Point text="Borrows heavily from Term Rewriting" />
      <Point text="Actively worked on" />
      <Point text="Takes breaking changes seriously" />
      <Point text="Accepted into Clojurists Together" />
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
        source={`
          {:players ...
           :weapons ...
           :stats ...
           :third-party ...}
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

      <Points title="Requirements">
        <Point text="All Valid Player/Weapon Combinations" />
        <Point text="name, weapon, class, attack power, and all upgrades" />
        <Point text="No third party weapons allowed" />
      </Points>


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
          {:players (m/scan {:name ?name
                             :class ?class})
           :weapons (m/scan {:name ?weapon
                             :allowed-classes #{?class}
                             :standard-upgrade !upgrades})
           :stats {?weapon {:attack-power ?attack-power
                            :upgrades [!upgrades ...]}}
           :third-party (m/not #{?weapon})}

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
          {:itemTemplates (m/gather {:pokemonSettings
                                     {:rarity (not-nil !rarity)
                                      :pokemonId !pokemon
                                      :form !form
                                      :stats {:as !stats}}})}

          [{:pokemon !pokemon 
            :form !form
            :rarity !rarity
            :stats !stats} ...])
      `}
     />
    {/* Width of code*/}
    <Code
      lang="clojure"
      source={`
        (m/search (parse-js example)
          (m/$ (m/or
                {:type "FunctionDeclaration"
                 :id {:name ?name}
                 :loc ?loc}

                {:type "VariableDeclarator"
                 :id {:name ?name}
                 :loc ?loc
                 :init {:type (m/or "FunctionExpression" "ArrowFunctionExpression")}}))
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
            (m/gather {:data 
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

    <Points title="A Ton of Operators">
      <Point text="app" />
      <Point text="$" />
      <Point text="and" />
      <Point text="or" />
      <Point text="let" />
      <Point text="pred" />
      <Point text="guard" />
      <Point text="..." />
      <Point text="..1" />
      <Point text="..!n" />
    </Points>

    <Headline
      textAlign="left"
      text="Taking it Further" />

     <Code
      lang="clojure"
      source={`

        (def simplify-addition
          (strat/until =
            (strat/bottom-up 
             (strat/attempt          
              (strat/rewrite
               (+ ?x 0) ?x
               (+ 0 ?x) ?x)))))

        (simplify-addition '(+ (+ 0 (+ 0 3)) 0)) ;; 3
        (simplify-addition '(+ (+ 0 (+ 0 (+ 3 (+ 2 0)))) 0)) ;; (+ 3 2)

      `}
     />

    <Code
      lang="clojure"
      source={`
        (def little-lang
          (r/rewrite
           (if true ?t ?f) ?t
           (if false ?t ?f) ?f
           (= ?x ?x) true
           (= ?x ?y) false
           (ignore ?x ?y) ?y
           (error) ~(throw (ex-info "error" {}))))
      `}
     />


    <Code
      lang="clojure"
      source={`
        (def prog  '(ignore (error) true))

        (strict-eval little-lang prog) ;; "error"

        (lazy-eval little-lang prog) ;; true
      `}
     />



    <Headline
      textAlign="left"
      text="Isn't this going to be slow?" />

    <Points title="Speed">
      <Point text="Meander is often slower than hand written Clojure" />
      <Point text="I have never had it be too slow" />
      <Point text="It will continue to get faster" />
    </Points>


    <Points title="Under the hood">
      <Point text="Parser" />
      <Point text="Expander" />
      <Point text="Matrix Based IR" />
      <Point text="Second Level IR" />
      <Point text="Dead Code Elimination" />
      <Point text="Optimizations" />
      <Point text="Simple Type Inference" />
      <Point text="Code Generation" />
    </Points>

    <Points title="Near(ish) Future">
      <Point text="Better Error Messages" />
      <Point text="Faster Code" />
      <Point text="Less Code Generated" />
      <Point text="More Powerful Recursive Matches" />
      <Point text="Update In Place" />
      <Point text="More User Extensibility" />
    </Points>


    <Headline
      color="blue"
      textAlign="left"
      text="Where this can go" />

    <Headline
      color="red"
      textAlign="left"
      text="Truly data oriented programming" />

    <Code
      lang="clojure"
      source={`
        (fn [] "do things")
        ;; => #function[wander.core5/eval22472/fn--22473]
      `}
    />

    <Headline
      color="blue"
      textAlign="left"
      text="Text of Our Code is Data" />

    <Headline
      color="green"
      textAlign="left"
      text="Execution as data" />

    <Code
      lang="clojure"
      source={`
        {:expr (my-function ?x ?y)}
        => 
        (println ?x ?y)
      `}
    />

    <Code
      lang="clojure"
      source={`
        {:expr (?rule & ?args)
         :result [:weird :data]}
          =>
        (println ?rule ?args)
      `}
    />

    <Code
      lang="clojure"
      source={`
        {:execution-history
         (all-steps-between (m/scan 0) (/ _ 0) !steps)}
        =>
        !steps
      `}
    />

    <Code
      lang="clojure"
      source={`
      (with-execution-rule
        {:rule !rules} => !rules
        (my-test))
      `}
    />



    <Headline
      color="green"
      textAlign="left"
      text="Rules as data" />


    <Code
      lang="clojure"
      source={`
        {:rules
         (m/scan
          {:rhs (m/scan println)
           :as ?rule})}
        =>
        ?rule

      `}
    />

    <Code
      lang="clojure"
      source={`
        {:rules
         (m/scan
          {:lhs (my-deprecated-rule & _)
           :as ?rule})}
        =>
        ?rule

      `}
    />

    <Code
      lang="clojure"
      source={`
        {:rules
         (locate
          {:lhs (my-deprecated-rule ?x ?y ?z)
           & ?rest
           :as ?rule})}
        =>
        (replace ?rule
         {:lhs (new-rule ?y ?x ?z)
          & ?rest})

      `}
    />

    <Code
      lang="clojure"
      source={`
        {:rule-history
         {:rule ?name
          :changes [_ ..10]}}

        => 
        ?name
      `}
    />

    <Headline
      color="green"
      textAlign="left"
      text="Programming by Example" />

    <Code
      lang="clojure"
      source={`
        (infer-rewrite
         {:name "Jimmy"
          :address
          {:address1 "123 street ave"
           :address2 "apt 2"
           :city "Townville"
           :state "IN"
           :zip "46203"}}

         {:name "Jimmy"
          :address {:line1 "123 street ave"
                    :line2 "apt 2"}
          :city-info {:city "Townville"
                      :state "IN"
                      :zipcode "46203"}})
      `}
     />

    <Code
      lang="clojure"
      source={`
        (r/rewrite
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
      color="green"
      textAlign="left"
      text="A More Transparent Future" />







    <BlankSlide />

  </Presentation>
