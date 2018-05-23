
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
} from "./library";

const images = {
  me: require("./images/me.jpg"),
};

preloader(images);

export default () =>
  <Presentation>
    <Headline
      textAlign="left" 
      text="Practical Functional Refactoring" />

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
      text="Practical Functional Refactoring" />
    
    <Points title="Limitations">
      <Point text="Code examples can't be large" />
      <Point text="Can't cover all cases" />
      <Point text="Can't address fighting frameworks" />
    </Points>


    <Headline
      color="green"
      textAlign="left"
      text="Functional is not an end goal" />

    <Headline
      color="blue"
      textAlign="left"
      text="Our aim is to make our code better" />

    <Headline
      color="yellow"
      textAlign="left"
      text="Widely Applicable Refactoring" />

    <Headline
      textAlign="left"
      text="Eliminate Loops" />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        let oldArray = [1,2,3]
        let newArray = [];
        for (var i = 0; i < oldArray.length; i++) {
          if (oldArray[i] % 2 === 0) {
            newArray.push(oldArray[i] * 2);
          }
        }
      `} />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        const oldArray = [1,2,3]
        const newArray = oldArray
          .filter(x => x % 2 === 0)
          .map(x => x * 2)
      `} />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        function sumOfActiveScores(team) {
          let totalScore = 0;
          for (player of team) {
            if (player.active) {
              for (score of player.scores) {
                if (score !== null) {
                  totalScore += score;
                }
              }
            }
          }
          return totalScore;
        }
      `} />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        function sumOfActiveScores(team) {
          return team
            .filter(player => player.active)
            .flatMap(player => player.scores)
            .filter(score => score !== null)
            .reduce((total, score) => total + score, 0);
        }
      `} />

    <Headline
      color="blue"
      textAlign="left"
      text="Remove Reassignment" />

    <Code
      color="blue"
      lang="javascript"
      source={`
        function checkStatus(player) {
          let status = true;
          let superStatus = false;
          if (player.hasThing) {
            if (player.myStuff) {
              superStatus = true;
              status = false
            } else {
              superStatus = false
            }
          }
          if (player.otherThing) { 
            superStatus = true; 
          }
          return status && superStatus;
        }
      `} />

    <Code
      color="blue"
      lang="javascript"
      source={`
        const calcStatus = (player) => {
          if (player.hasThing && player.myStuff) {
            return false;
          } 
          return true;
        }

        const calcSuperStatus = player => {
          if (player.hasThing && player.myStuff) {
            return true;
          }
          if (player.otherThing) {
            return true;
          }
          return false;
        };
      `} />

    <Code
      color="blue"
      lang="javascript"
      source={`
        const calcStatus = (player) => {
          return !(player.hasThing && player.myStuff) {
        }

        const calcSuperStatus = player => {
          return (player.hasThing && player.myStuff) ||
                  player.otherThing'
        };
      `} />

    <Code
      color="blue"
      lang="javascript"
      source={`
        const checkStatus = (player) => {
          return calcStatus(player) && calcSuperStatus(player);
        }
      `} />

    <Headline
      color="yellow"
      textAlign="left"
      text="Extract Side Effects" />

    <Code
      maxWidth={1300}
      title="Extract Side Effects"
      color="yellow"
      lang="javascript"
      source={`
        const createAndSendMessages = (users) => {
          for (user of users) {
            const message = \`Hello \${user.name} thanks for \${user.action}.\`;
            sms.send(info.number, message);
          }
        }
      `} />

    <Code
      maxWidth={1300}
      title="Extract Side Effects"
      color="yellow"
      lang="javascript"
      source={`
        const createMessage = (user) => {
          return \`Hello \${user.name} thanks for \${user.action}.\`;
        }
        const sendMessage = (message) => sms.send(message);

        const createMessages = (users) => {
          return users
            .map(createMessage)
            .map(sendMessage);
        }
      `} />

    <Code
      maxWidth={1300}
      title="Extract Side Effects"
      color="yellow"
      lang="javascript"
      source={`
        const createMessage = (user) => {
          return \`Hello \${user.name} thanks for \${user.action}.\`;
        }
        const sendMessage = (message) => sms.send(message);

        const createAndSendMessages = (users) => {
          return users
            .filter(isActive)
            .map(createMessage)
            .map(sendMessage)
        }
      `} />


    <Code
      maxWidth={1300}
      title="Extract Side Effects"
      color="yellow"
      lang="javascript"
      source={`
        const createAndSendMessages = (users, sendMessage) => {
          return users
            .filter(isActive)
            .map(createMessage)
            .map(sendMessage)
        }

        const testCreateAndSend = (users) => {
          const sentMessages = createAndSendMessages(users, x => x)
          assert isCorrect(sentMessages)
        }
      `} />


    <Headline
      color="green"
      textAlign="left"
      text="Statically Typed Languages" />

    <Headline
      color="red"
      textAlign="left"
      text="Eliminate Null" />

    <Code
      color="red"
      title="Eliminate Null"
      lang="javascript"
      source={`
        public Player {
          Attack special;

          public Attack getSpecialAttack() {
            return this.special;
          }
        }
        ...
        myPlayer.getSpecialAttack().attack();
      `} />

    <Code
      color="red"
      title="Eliminate Null"
      lang="javascript"
      source={`
        public Player {
          Optional<Attack> special = Optional.empty();

          public Optional<Attack> getSpecialAttack() {
            return this.special;
          }
        }
        ...
        myPlayer.getSpecialAttack().ifPresent(Attack::attack);
      `} />

    <Code
      color="red"
      title="Eliminate Null"
      lang="javascript"
      source={`
        myPlayer.getSpecialAttack()
          .orElseGet(myPlayer::getNormalAttack)
          .attack();
      `} />

    <Headline
      color="green"
      textAlign="left"
      text="Immutable Objects" />


    <Code
      color="green"
      title="Immutable Objects"
      lang="javascript"
      source={`
        public class Person
        {
          public String Name { get; }
          public int Age { get; }
          public Person(String name, int age) 
          {
            this.Name = name;
            this.Age = age;
          } 
        }
      `} />

    <Points title="Best Practices?">
      <Point text="Encapsulation is bad" />
      <Point text="Data and behavior are different" />
      <Point text="Avoid Inheritance" />
    </Points>

    <Code
      maxWidth={1300}
      color="yellow"
      title="Partial Application"
      lang="javascript"
      source={`
        public class DataFetcher
        {
          public static getActiveUsers(Connection connection) {}
          public static getUserById(Connection connection, UUID id) {}
         }
      `} />

    <Code
      maxWidth={1300}
      color="yellow"
      title="Partial Application"
      lang="javascript"
      source={`
        public class DataFetcher
        {
          private Connection connection;
          public DataFetcher(Connection connection) 
          {
            this.connection = connection;
          } 

          public getActiveUsers() {}
          public getUserById(UUID id) {}
        }
      `} />

    <Code
      maxWidth={1300}
      color="yellow"
      title="Partial Application"
      lang="javascript"
      source={`
        public class DataFetcher
        {
          private Connection connection;
          public DataFetcher(Connection connection) 
          {
            this.connection = connection;
          } 

          public getActiveUsers() {}
          public getUserById(UUID id) {}
        }
      `} />

    <Headline
      color="red"
      textAlign="left"
      text="Experiments" />

    <Code
      maxWidth={1300}
      color="red"
      title="Experiments"
      lang="javascript"
      source={`
          public interface Set<T> {
              Boolean isEmpty();
              Boolean contains(T t);
              Set<T> insert(T t);
              Set<T> union(Set<T> t);
          }
      `} />

    <Code
      maxWidth={1300}
      color="red"
      lang="javascript"
      textSize={26}
      source={`
        public class EmptySet<T> implements Set<T> {
            public Boolean isEmpty() {
                return true;
            }
            public Boolean contains(T t) {
                return false;
            }
            public Set<T> insert(T t) {
                return new InsertSet<>(this, t);
            }
            public Set<T> union(Set<T> t) {
                return new UnionSet<>(this, t);
            }
        }

      `} />

    <Code
      maxWidth={1300}
      maxHeight={800}
      color="red"
      lang="javascript"
      textSize={22}
      source={`
        public class InsertSet<T> implements Set<T> {
            private final Set<T> other;
            private final T t;
            public InsertSet(Set<T> other, T t) {
                this.other = other;
                this.t = t;
            }
            public Boolean isEmpty() {
                return false;
            }
            public Boolean contains(T t) {
                return t == this.t || other.contains(t);
            }
            public Set<T> insert(T t) {
                return new InsertSet<>(this, t);
            }
            public Set<T> union(Set<T> t) {
                return new UnionSet<>(this, t);
            }
        }
      `} />

    <Code
      maxWidth={1300}
      maxHeight={800}
      color="red"
      lang="javascript"
      textSize={22}
      source={`
          public class UnionSet<T> implements Set<T> {
            private final Set<T> set1;
            private final Set<T> set2;
            public UnionSet(Set<T> set1, Set<T> set2) {
                this.set1 = set1;
                this.set2 = set2;
            }
            public Boolean isEmpty() {
                return set1.isEmpty() && set2.isEmpty();
            }
            public Boolean contains(T t) {
                return set1.contains(t) || set2.contains(t);
            }
            public Set<T> insert(T t) {
                return new InsertSet<>(this, t);
            }
            public Set<T> union(Set<T> t) {
                return new UnionSet<>(this, t);
            }
          }
      `} />
      
    <Code
      color="red"
      lang="javascript"
      source={`
        new EmptySet()
          .insert(7)
          .insert(8)
          .contains(3) // false

        new EmptySet()
          .insert(7)
          .insert(8)
          .contains(7) // true
      `} />

    <Code
      color="red"
      lang="javascript"
      textSize={26}
      source={`
        public class EvenSet implements Set<Integer> {
          public Boolean isEmpty() {
              return false;
          }
          public Boolean contains(Integer i) {
              return i % 2 == 0;
          }
          public Set<Integer> insert(Integer t) {
              return new InsertSet<>(this, t);
          }
          public Set<Integer> union(Set<Integer> t) {
              return new UnionSet<>(this, t);
          }
        }
      `} />

    <Code
      color="red"
      lang="javascript"
      source={`
        new EvenSet()
          .contains(2) // true

        new EvenSet()
          .insert(7)
          .contains(7) // true

        new EvenSet()
          .contains(5123412) // true
      `} />









   <BlankSlide />



    <Code
      color="blue"
      title="Extended Example"
      lang="javascript"
      source={`
        function calculateDamage(player) {
          if (player.bonus) {
            player.damage += 5
          }
          if (player.weak) {
            player.damage -= 1
          }
          if (player.crit) {
            player.damage *= 2
          }
        }
      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        const critStatus = player => {
          return player.crit
            ? update(player, "damage", x => x * 2)
            : player;
        };

        const weakenedStatus = ...
        const attackBonus = ...

        function calculateDamage(player) {
          return critStatus(weakenedStatus(attackBonus(player)));
        }
      `} />

    <Code
      color="blue"
      lang="javascript"
      source={`
        const critStatus = ...
        const weakenedStatus = ...
        const attackBonus = ...

        function calculateDamage(player) {
          return flow(
            attackBonus,
            weakenedStatus,
            critStatus,
          )(player)
        }
      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        const updateIf = (object, property, pred, f) => {
          if (pred(object)) {
            return update(object, property, f)
          } 
          return object;
        }

        const updateDamage = (pred, f) => player => 
          updateIf(player, "damage", pred, f)
      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        const critStatus = updateDamage(p => p.crit, x => x * 2) 
        const applyWeakened = updateDamage(p => p.weak, x => x - 1) 
        const addBonus = updateDamage(p => p.bonus, x => x + 5)

        const calculateDamage =
          flow(
            addBonus,
            applyWeakened,
            applyCrit,
          );

      `} />

    <Headline
      color="green"
      textAlign="left"
      text="Wow that was complicated" />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        function chooseAttack(player) {
          calculateMeleeDamage(player)
          let meleeDamage = player.damage;

          calculateRangedDamage(player)
          let rangedDamage = player.damage;

          if (rangedDamage > meleeDamage) {
            return "ranged"
          } else {
            return "melee"
          }
        }

      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        function chooseAttack(player) {
          let meleePlayer = player.copy();
          calculateMeleeDamage(meleePlayer)
          let meleeDamage = meleePlayer.damage;
          
          let rangedPlayer = player.copy();
          calculateRangedDamage(rangedPlayer)
          let rangedDamage = rangedPlayer.damage;

          if (rangedDamage > meleeDamage) {
            return "ranged"
          } else {
            return "melee"
          }
        }

      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        function chooseAttack(player) {
          const meleeDamage = calculateMeleeDamage(player).damage;
          const rangedDamage = calculateRangedDamage(player).damage;

          if (rangedDamage > meleeDamage) {
            return "ranged"
          } else {
            return "melee"
          }
        }

      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        function calculateDamage(player) {
          for (status of player.statusEffects)
            if (status.name === "attackBonus") {
              player.damage += 5
            }
            if (status.name === "weak") {
              player.damage -= 1
            }
            if (status.name === "crit") {
              player.damage *= 2
            }
           if (status.name === "strengthDrain") {
              player.strength -= 2
            }
          }
        }
      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        function calculateDamage(player) {
          for (status of player.statusEffects)
            if (status.name === "attackBonus") {
              player.damage += 5
            }
            if (status.name === "weak") {
              player.damage -= 1
            }
            if (status.name === "crit") {
              player.damage *= 2
            }
           if (status.name === "strengthDrain") {
              player.strength -= 2
            }
          }
        }
      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        function applyStatusEffects(player) {
          return player.statusEffects.reduce(
            (player, statusEffect) => statusEffect.effect(player))
        }
      `} />

    <Code
      maxWidth={1300}
      color="blue"
      lang="javascript"
      source={`
        function poisonAttack(player) {
          return flow(
            addStatus(weakened),
            causeDamage(hp => hp - rand(1, 6))
          )(player);
        }
      `} />





    <BlankSlide />

  </Presentation>
