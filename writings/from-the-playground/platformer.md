# Platformer

Platformer is a very old project (Oct 2014) that I had a ton of fun writing. It isn't code I'd recommend anyone to emulate, but I certainly learned a good amount from it and had a lot of fun. Platformer is exactly that a very simple in browser platformer. It never really turned into a game and was never really meant to.

![Screenshot_2020-02-22 Screenshot](/Users/jimmyhmiller/Documents/Code/PlayGround/writings/from-the-playground/images/platformer.png)

## The Basic Idea

Instead platformer was really about playing with the idea of a clojurescript based Entity-Component-System. The basic idea was to be able to define all the behavior of an entity separately and enable composition of functionality through simple data manipulation. Further, entity can affect other entities by override some part of their data. 

Here is a simple example in the picture above you can see red and blue platforms these are portals. Here is the code responsible for their functionality.

```clojure
(defn portal [x obj floor]
  (assoc obj
    :y (top floor)
    :vy (opposite-velocity (:vy obj))
    :x x))

(def floor3
  {:name :floor3
   :width 100
   :height 30
   :x 200
   :y 10
   :color "#F00"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top (partial portal 400)
            :bottom stand}})

(def floor4
  {:name :floor4
   :width 100
   :height 30
   :x 350
   :y 10
   :color "#00F"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top (partial portal 250)
            :bottom stand}})
```

What we have done is given the floor a function for what should happen when an object interacts with its sides, for example, top uses the portal function. All this function does is changes some data of the entity it is handed. So in this case, we change its velocity and position. This model is incredibly flexible (probably not very performant in the long run.) Here are a couple more examples:

```clojure
(defn bounce [obj floor]
  (assoc obj
    :y (top floor)
    :vy (* .88 (opposite-velocity (:vy obj)))))

(defn fast [obj floor]
  (assoc obj
    :y (fix-y obj floor)
    :vy 9.8
    :standing (= (fix-y obj floor) (top floor))
    :overrides {:walk (walk-with-velocity 250)}))
```

## Good Things

One of the things I loved about this project is how satisfying it was to have something I could play with afterwards. Being able to tweak numbers and code and see the impact was really fun. I was able to make things like anti-gravity with just a simple change. 

For my tiny example performance was pretty good. There was some math I was doing that made things feel a bit choppy, but actual fps stayed at around 59 as the character (a div) moved around the page. I have very little doubt though that adding more moving entities would cause lots of headache for performance.

## Bad Things

My collision detection and "fixing" were terrible. The idea was that an entity would decide what to do when objects collided with it. For example, our player is always falling, but the ground is constantly moving him back to the top. This makes movement feel very awkward, but it also makes has bugs when it comes to platforms. If you jump just right at the edge of a platform, you will be catapaulted in the air.

Not quite bad, but it is interesting to see that this project is implemented an Om, a hot clojurescript framework that died a while back. Om.next was supposed to revitalize it but that never came about. Its spiritual successor, fulcro, has continued developing, but doesn't have much usage.

## Ways to Make This Better

Instead of rendering divs it would have been better to render in canvas. In general this idea was a bit silly. I'd probably want a better approach things like `:standing`, I actually just don't understand what it is doing at all. I also was really inconsistent about the interaction model. Above there is something about overrides. I guess that is me overriding an attribute. I assume I did this because I wanted the ability to restore the old function? In fact, I think on every single frame I am running the collision code and re-associng? Probably should have changed that. Finally, I should probably have done a proper collision detection algorithm with raycasting. As well as consider things that interact at a distance.