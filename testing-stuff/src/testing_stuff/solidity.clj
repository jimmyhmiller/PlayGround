(ns testing-stuff.solidity
  (:require  [promesa.core :as p]
             [cloth.core :as cloth]
             [cloth.contracts :as c]
             [cuerdas.core :as cc]
             [cheshire.core :as json]
             [clojure.core.async :as async]))

@(cloth/balance)

(reset! cloth.core/global-signer (cloth.keys/create-keypair))

(reset! cloth.core/global-signer 
        {:private-key "1c9a33d01ee7bbaa2a3e1d3325021cc6b3e50fbd4a08d438b1320b8df2cba00a"
         :address "0x5d122f0af80d5472d9a68d68f65e8f4513e29fa8"})


(c/defcontract crowdsale "/Users/jimmymiller/Documents/Code/PlayGround/testing-stuff/contracts/Crowdsale.sol")


(c/defcontract moeda-token "/Users/jimmymiller/Documents/Code/PlayGround/testing-stuff/contracts/MoedaToken.sol")

(def contract-address @(deploy-crowdsale!))
(def token @(deploy-moeda-token!))
(def token-amount "1000000000000000000")

@(create! token contract-address token-amount)
@(unlock! token)
@(drain-token! contract-address token "0x5d122f0af80d5472d9a68d68f65e8f4513e29fa8")


(def compiled (c/compile-solidity "/Users/jimmymiller/Documents/Code/PlayGround/testing-stuff/contracts/Crowdsale.sol"))

compiled

(def contract-key :/Users/jimmymiller/Documents/Code/PlayGround/testing-stuff/contracts/Crowdsale.sol:Crowdsale)

(def abi (json/parse-string (get-in compiled [:contracts contract-key :abi]) true))

abi
(filter #(= (:type %) "function") abi)

(keys (ns-publics 'cloth.contracts))

(c/call-contract-fn 'create 'moeda-token [contract-address token-amount])
 
(def drain-ch (:events @(token-drain-ch contract-address)))

(async/put! drain-ch "test1")
(async/go (println (async/<! drain-ch))) 
