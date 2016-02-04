(ns figwheel.connect (:require [devcards.core :include-macros true] [devcard.core] [figwheel.client] [figwheel.client.utils]))
(figwheel.client/start {:build-id "devcards", :devcards true, :websocket-url "ws://localhost:3449/figwheel-ws"})
(devcards.core/start-devcard-ui!)

