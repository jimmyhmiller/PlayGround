
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
      color="red"
      textAlign="left"
      text="Failure to specify values causes conflict" />

    <Points title="Values">
      <Point text="Composability" />
      <Point text="Debuggability" />
      <Point text="Expressiveness" />
    </Points>

    <Headline
      color="yellow"
      textAlign="left"
      text="Fewer lines is not a measure of success" />

    <Headline
      color="green"
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
                  player.otherThing
        };
      `} />

    <Code
      color="blue"
      lang="javascript"
      maxWidth={1300}
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
            sms.send(message);
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
      color="blue"
      textAlign="left"
      text="Make illegal states unrespresentable" />

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
        myPlayer.getSpecialAttack().attack() // Compile error
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
      color="blue"
      textAlign="left"
      text="Eliminate Exceptions" />

    <Code
      color="blue"
      title="Eliminate Exceptions"
      lang="javascript"
      source={`
        public Special {
          public Integer attack() throws InvalidMove {
            return 9001;
          }
        }
        ...
        special.attack()
      `} />

    <Code
      color="blue"
      title="Eliminate Exceptions"
      lang="javascript"
      source={`
        public Special {
          public Try<InvalidMove, Integer> attack() {
            return Try.success(9001);
          }
        }
        ...
        special.attack().failed(regular::attack);
      `} />



    <Headline
      color="yellow"
      size={4}
      textAlign="left"
      text="Going against the grain" />


    <Headline
      color="blue"
      textAlign="left"
      text="Dynamically Typed Languages" />

    <Headline
      color="blue"
      textAlign="left"
      text="Get rid of classes" />

    <Points title="Class Downsides">
      <Point text="Couple data and operations" />
      <Point text="Don't compose" />
      <Point text="Allows hidden state" />
    </Points>

    <Points title="Class Replacements">
      <Point text="Functions" />
      <Point text="Pure, raw data" />
      <Point text="Single state store" />
      <Point text="Higher order functions" />
    </Points>

    <Code
      color="green"
      title="Class Example"
      lang="python"
      source={`
        class Player(object):
            
            def __init__(self, name, strength):
                self.name = name
                self.strength = strength
            
            def get_attack_damage(self):
                return self.strength

        p = Player("Gelabrous", 1)

        print(p.get_attack_damage()) # 1
      `} />


    <Code
      color="green"
      title="Class Example"
      lang="python"
      source={`
        class WeakenedPlayer(Player):
            
            def __init__(self, name, strength):
               super().__init__(name, strength)
            
            def get_attack_damage(self):
                return super().get_attack_damage() - 1

        p = WeakenedPlayer("Gelabrous", 1)

        print(p.get_attack_damage()) # 0
      `} />

    <Code
      color="green"
      title="Class Example"
      lang="python"
      source={`
        class Player(object):
            
            def __init__(self, name, strength, weakened=false):
                ...

            def weaken(self):
                self.weakened = true
            
            def get_attack_damage(self):
                if self.weakened:
                    return strength - 1
                else:
                    return strength

      `} />

    <Code
      color="blue"
      title="Function Example"
      lang="python"
      source={`
        player = {
          "name": "Gelabrous",
          "strength": 1
        }

        def get_attack_damage(player):
            return player["strength"]

        print(get_attack_damage(player)) # 1
      `} />

    <Code
      color="blue"
      title="Function Example"
      lang="python"
      source={`
        player = {
          "name": "Gelabrous",
          "strength": 1,
          "weakened": true
        }

        def get_attack_damage(player):
          if "weakened" in player:
              return strength - 1
          else:
              return strength

        print(get_attack_damage(player)) # 0
      `} />


    <Code
      color="blue"
      title="Function Example"
      lang="python"
      source={`
        def apply_weakening(f):
            def weakened(player):
                if "weakened" in player:
                    return f(player) - 1
                return f(player)
            return weakened

        @apply_weakening
        def get_attack_damage(player):
            return player["strength"]

        print(get_attack_damage(player)) # 0
      `} />


    <Code
      color="blue"
      title="Function Example"
      lang="python"
      source={`
        def status_effect(property, f):
            def apply_effect(g):
                def effect(player):
                    if property in player:
                        return f(g(player))
                    return f(player)
                return effect
            return apply_effect

      `} />

    <Code
      color="blue"
      title="Function Example"
      lang="python"
      source={`
        @status_effect("weakened", lambda x: x - 1)
        @status_effect("raging", lambda x: x * 2)
        def get_attack_damage(player):
            return player["strength"]

        print(get_attack_damage(player)) # 1

      `} />

    <Points color="green" title="Interpret Data First">
      <Point text="Separate what and how" />
      <Point text="Avoid mocks" />
      <Point text="Post-hoc validation" />
    </Points>


    <Code
      color="green"
      lang="javascript"
      source={`
        const loadUsers = (users) => {
          const validUsers = []
          for (user of users) {
            const standardUser = {}
            standardUser.name = user.first_name + user.last_name;
            standardUser.dateAdded = user.date_added;
            if (last5Days(standardUser.dateAdded)) {
              validUsers.push(standardUser);
            } else {
              console.log(\`user was not valid $\{user}\`)
            }

          }
          validUsers.each(sendToDb)
        }
      `} />

    <Code
      color="green"
      lang="javascript"
      source={`
        {action: "logInvalid",
         user: {}}

        {action: "load",
         user: {}}
      `} />


    <Code
      color="green"
      lang="javascript"
      source={`
        const determineAction = (user) => {
          if (isValid(user)) {
            return {action: "load", user: convertToStandard(user)};
          }
          return {action: "logInvalid", user: user};
        }

        const logInvalid = ({user}) => {
          console.log(\`user was not valid $\{user}\`)
        }

        const loadUser = ({user}) => {
          return sendToDb(user)
        }
      `} />


    <Code
      color="green"
      lang="javascript"
      source={`
        const performAction = (performer) => (payload) => {
          return performer[payload.type](payload)
        }

        const loadUsers = (users, actionPerformer) => {
          return users.map(performAction(actionPerformer))
        }
      `} />

    <Code
      color="green"
      lang="javascript"
      source={`
        const actionPerformer = {
          logInvalid: logInvalid,
          load: loadUser
        }

        const testActionPerformer = {
          logInvalid: (x) => ["logged", x],
          load: x => ["loaded", x],
        }

        loadUsers(users, actionPerformer)
        loadUsers(users, testActionPerformer)

      `} />

    <Headline
      size={4}
      text="Boilerplate (and hacks) abound" />

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

    <Code
      title="Immutable Updates"
      lang="javascript"
      source={`
      function updateVeryNestedField(state, action) {
        return {
          ...state,
          first: {
            ...state.first,
            second: {
              ...state.first.second,
              [action.someId]: {
                ...state.first.second[action.someId],
                fourth: action.someValue
              }
            }
          }
        }
      }
    `} />

    <Code
      title="Linked List"
      lang="haskell"
      source={`
        data List a = Nil | Cons a (List a)
      `} />


    <Code
      textSize={14}
      title="Linked List"
      lang="javascript"
      source={`
        interface List<T> {
            public <R> R accept(Visitor<T,R> visitor);

            public static interface Visitor<T,R> {
                public R visitNil();
                public R visitCons(T value, List<T> sublist);
            }
        }

        final class Nil<T> implements List<T> {
            public Nil() { }

            public <R> R accept(Visitor<T,R> visitor) {
                return visitor.visitNil();
            }
        }
        final class Cons<T> implements List<T> {
            public final T value;
            public final List<T> sublist;

            public Cons(T value, List<T> sublist) {
                this.value = value;
                this.sublist = sublist;
            }

            public <R> R accept(Visitor<T,R> visitor) {
                return visitor.visitCons(value, sublist);
            }
        }
      `} />

    <Points title='New "OO" languages are very functional'>
      <Point text="Kotlin" />
      <Point text="Rust" />
      <Point text="Swift" />
    </Points>

    <Code
      title="C# Record types"
      lang="javascript"
      source={`
        sealed abstract class List<T>;
        sealed class Nil<T>() : List<T>;
        sealed class Cons(T t, List<T> list) : List<T>;
      `} />

    <Code
      title="C# Record types"
      lang="javascript"
      source={`
        Expr Simplify(Expr e)
        {
          switch (e) {
            case Mult(Const(0), _): return Const(0);
            case Mult(_, Const(0)): return Const(0);
            case Mult(Const(1), var x): return Simplify(x);
            case Mult(var x, Const(1)): return Simplify(x);
            case Mult(Const(var l), Const(var r)): return Const(l * r);
            case Add(Const(0), var x): return Simplify(x);
            case Add(var x, Const(0)): return Simplify(x);
            case Add(Const(var l), Const(var r)): return Const(l + r);
            case Neg(Const(var k)): return Const(-k);
            default: return e;
          }
        }
      `} />


    <Code
      title="Javascript Pipelines"
      lang="javascript"
      source={`
        function sumOfActiveScores(team) {
          return team
            |> filter(player => player.active)
            |> flatMap(player => player.scores)
            |> filter(score => score !== null)
            |> reduce((total, score) => total + score, 0);
        }
      `} />

    <Code
      title="Javascript Pipelines"
      lang="javascript"
      source={`
        function sumOfActiveScores(team) {
          return team
            |> filter(player => player.active)
            |> flatMap(player => player.scores)
            |> filter(score => score !== null)
            |> sum()
        }
      `} />

    <Code
      title="Javascript Expressions"
      lang="javascript"
      source={`
        let x = do {
          if (foo) { 
            "Yay!"
          } else {
            "No :("
          }
        };
      `} />


    <Headline
      text="Borrowed inspiration" />

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
      title="Immutable Builder"
      lang="javascript"
      source={`
        new User()
            .Name("Jimmy")
            .Hobby("Programming")
            .Build();
      `} />













    <BlankSlide />



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


    <Headline
      color="blue"
      text="Extended Example" />

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
