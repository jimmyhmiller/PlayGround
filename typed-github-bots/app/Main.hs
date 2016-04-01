{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
module Main where

import Dice
import Lib
import GHC.Generics
import Data.Text.Encoding
import Data.Char
import Codec.Binary.Base64.String
import qualified Data.ByteString.Char8 as Byte
import qualified Data.Text as Text
import Data.List
import Data.Default
import Text.Regex
import Data.Maybe
import qualified GitHub as GH
import qualified GitHub.Data.Id as Id
import qualified Data.Yaml as Y
import qualified GitHub.Endpoints.Repos as Repos
import qualified GitHub.Endpoints.Issues.Comments as Comments
import qualified Data.Vector as V
import Control.Monad.Random
import Control.Monad.Trans.Except
import Control.Monad
import Control.Monad.Trans
import Control.Error.Util
import qualified Control.Arrow as A


data ProposalStatus = Passing | Failing | Tied

data VoteCount = VoteCount {
  yays :: Int,
  nays :: Int
} deriving (Show)


instance Default VoteCount where
    def = VoteCount 0 0

data VoteType = Yay | Nay deriving (Show, Eq)

data GamePlayers = GamePlayers {
  activePlayers :: [Player],
  inactivePlayers :: [Player]
} deriving (Show, Generic)

data Player = Player {
  name :: Text.Text,
  points :: Int
} deriving (Show, Generic, Eq)

data Vote = Vote {
  comment :: GH.IssueComment,
  player :: Player
}

instance Y.FromJSON Player where

instance Y.FromJSON GamePlayers where

infixl 0 |>
(|>) :: a -> (a -> b) -> b
(|>) a f = f a

getFileContents :: Repos.Content -> String
getFileContents (Repos.ContentFile cd) = decode $ Text.unpack $ Repos.contentFileContent cd

propToPred :: Eq b => (a -> b) -> a -> a -> Bool
propToPred prop x y = prop x == prop y

toGamePlayers :: String -> Maybe GamePlayers
toGamePlayers s = Y.decode $ Byte.pack s

distinctBy :: (a -> a -> Bool) -> [a] -> [a]
distinctBy pred xs = xs
  |> groupBy pred
  |> fmap last

isActive :: [Player] -> Text.Text -> Bool
isActive xs s = elem s $ fmap name xs

isVote :: Text.Text -> Bool
isVote s = Text.toLower s == "yay" || Text.toLower s == "nay"

isProposal :: String -> Bool
isProposal title = isJust $ matchRegex (mkRegex "^[0-9]+.*") title

getProposalNumber :: String -> Maybe Int
getProposalNumber title = do
    (number : _) <- matchRegex (mkRegex "^([0-9]+).*") title
    return $ read number :: Maybe Int

getUserName :: GH.SimpleUser -> Text.Text
getUserName u = let login = GH.simpleUserLogin u in Repos.untagName login

findPlayer :: [Player] -> GH.IssueComment -> Player
findPlayer players comment = players
  |> filter (\player -> name player == (comment |> GH.issueCommentUser |> getUserName))
  |> head

getVotes :: [Player] -> [GH.IssueComment] -> [Vote]
getVotes players xs = xs
  |> filter (isVote . GH.issueCommentBody)
  |> distinctBy (propToPred GH.issueCommentUser)
  |> filter (isActive players . getUserName . GH.issueCommentUser)
  |> map (\comment -> Vote comment (findPlayer players comment))

incYay :: VoteCount -> VoteCount
incYay VoteCount { yays = ys, nays = ns } = VoteCount (ys + 1) ns

incNay :: VoteCount -> VoteCount
incNay VoteCount { yays = ys, nays = ns } = VoteCount ys (ns + 1)

toVoteType :: Text.Text -> VoteType
toVoteType "yay" = Yay
toVoteType "nay" = Nay

incVote :: VoteCount -> VoteType -> VoteCount
incVote v Yay = incYay v
incVote v Nay = incNay v

voteCount :: [Vote] -> VoteCount
voteCount = foldl (\v t -> incVote v (toVoteType (GH.issueCommentBody $ comment t))) def

percent :: Int -> Int -> Float
percent x y = 100 * ( a / b )
  where a = fromIntegral x :: Float
        b = fromIntegral y :: Float

percentYays :: VoteCount -> Float
percentYays VoteCount { yays=ys, nays=ns } = percent ys (ys+ns)

isQuorum :: [Vote] -> [Player] -> Bool
isQuorum votes players = percent (length votes) (length players) > 50

proposalStatus :: VoteCount -> ProposalStatus
proposalStatus v
  | yaypercent > 50 = Passing
  | yaypercent < 50 = Failing
  | otherwise = Tied
  where yaypercent = percentYays v

pointsPlus :: [Player] -> [Int] -> [Player]
pointsPlus [] [] = []
pointsPlus (Player name points:ps) (x:xs) = Player name (points + x) : pointsPlus ps xs

updatePlayerPoints :: (MonadRandom m) => [Vote] -> m [Player]
updatePlayerPoints votes = do
  rolls <- replicateM (length votes) (rollDice $ Dice 2 6 1)
  return $ pointsPlus (map player votes) rolls

data CustomError = CommentError Comments.Error | ParseError String
  deriving (Show)

main = do
  result <- runExceptT $ do
    coms <- lift $ Comments.comments "nomicness" "a-whole-new-world" (Id.Id 39)
    file <- lift $ Repos.contentsFor "nomicness" "a-whole-new-world" "players.yaml" Nothing
    comments <- hoistEither $ A.left CommentError coms
    contents <- hoistEither $ A.left CommentError file
    players <- toGamePlayers (getFileContents contents) ?? ParseError "not found"
    let votes = getVotes (activePlayers players) (V.toList comments)
    updatePlayerPoints votes
  print result
