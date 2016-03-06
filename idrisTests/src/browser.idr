import Data.SortedMap
import Effects
import Effect.State
import Effect.Exception

AttrMap : Type
AttrMap = SortedMap String String


record ElementData where
  constructor MkElementData
  tagName : String
  attributes : AttrMap

data NodeType = Text String | Element ElementData

record Node where
  constructor MkNode
  children : List Node
  nodeType : NodeType


text : String -> Node
text str = MkNode [] (Text str)

elem : (name : String) -> (attrs : AttrMap) -> (children : List Node) -> Node
elem name attrs children = MkNode children (Element (MkElementData name attrs))


record Parser where
  constructor MkParser
  pos : Nat
  input : String
  
 
instance Default Parser where
    default = MkParser 0 ""



assert : (pred : a -> Bool) -> (message: String) -> a -> Eff a [STATE Parser, EXCEPTION String]
assert pred message x = if pred x then pure x else raise message

assertEq : (Show a, Eq a) => a -> a -> Eff a [STATE Parser, EXCEPTION String]
assertEq x y = assert (\x => x == y) (show x ++ " does not equal " ++ show y) x



nextChar : Parser -> Char
nextChar (MkParser pos input) = strIndex input (cast pos)

startsWith : Parser -> String -> Bool
startsWith (MkParser pos input) str = str `isPrefixOf`  (substr pos (length input) input)


eof : Parser -> Bool
eof (MkParser pos input) = pos >= length input


getRemaining : Parser -> String
getRemaining (MkParser pos input) = substr pos (length input) input

progressParser : Parser -> Parser
progressParser parser@(MkParser pos input) = record {  pos = (S pos) } parser



consumeChar : Eff (Char) [STATE Parser, EXCEPTION String]
consumeChar = do
  parser <- get
  let input = input parser
  let pos = pos parser
  put $ progressParser parser
  pure $ nextChar parser


consumeWhile : (Char -> Bool) -> Eff (String) [STATE Parser, EXCEPTION String]
consumeWhile pred = do
  parser <- get
  if (not (eof parser)) && (pred $ nextChar parser) then do
    let strChar = cast (nextChar parser)
    put (progressParser parser)
    newStr <- consumeWhile pred
    pure (strChar ++ newStr)
  else (pure "")

consumeWhitespace : Eff String [STATE Parser, EXCEPTION String]
consumeWhitespace = consumeWhile isSpace



parseTagName : Eff String [STATE Parser, EXCEPTION String]
parseTagName = consumeWhile isAlphaNum


parseAttrValue : Eff String [STATE Parser, EXCEPTION String]
parseAttrValue = do
  openQuote <- consumeChar -- " || '
  value <- consumeWhile (\x => x /= openQuote)
  consumeChar -- openQuote
  pure value


parseAttr : Eff (String, String) [STATE Parser, EXCEPTION String]
parseAttr = do
  name <- parseTagName
  assertEq '=' !consumeChar
  value <- parseAttrValue
  pure (name, value)



parseAttributes : Eff AttrMap [STATE Parser, EXCEPTION String]
parseAttributes = do
  consumeWhitespace
  parser <- get
  if nextChar parser == '>' then do
    pure empty
  else do
    (name, value) <- parseAttr
    newMap <- parseAttributes
    pure (insert name value newMap)
  
parseText : Eff Node [STATE Parser, EXCEPTION String]
parseText = do
  value <- consumeWhile (\x => x /= '<')
  pure (text value)




mutual

  parseNode : Eff Node [STATE Parser, EXCEPTION String]
  parseNode = do
    parser <- get
    if nextChar parser == '<' 
    then do
      parseElement
    else parseText

  parseNodes : Eff (List Node) [STATE Parser, EXCEPTION String]
  parseNodes = do
    consumeWhitespace
    parser <- get
    if eof parser || startsWith parser "</" then
      pure []
    else do
      node <- parseNode
      newNodes <- parseNodes
      pure (node :: newNodes)


  --This method proves that this way of doing things may work,
  --but it surely isn't good idris code.
  parseElement : Eff Node [STATE Parser, EXCEPTION String]
  parseElement = do
    assertEq '<' !consumeChar
    tagName <- parseTagName
    attrs <- parseAttributes
    assertEq '>' !consumeChar
    children <- parseNodes
    assertEq '<' !consumeChar
    assertEq '/' !consumeChar
    parseTagName
    assertEq '>' !consumeChar
    pure $ elem tagName attrs children



addParser : String -> Eff () [STATE Parser, EXCEPTION String]
addParser x = do
  put (MkParser 0 x)
  
  
startsWithLess : Eff Char [STATE Parser, EXCEPTION String]
startsWithLess = do
  assertEq '<' !consumeChar


tryMaybe : Char -> Eff (Maybe Char) [STATE Parser, EXCEPTION String]
tryMaybe x = do
  parser <- get
  if (nextChar parser == x) then do
    c <- consumeChar
    pure $ Just c
  else pure Nothing



getState : Eff String [STATE Parser, EXCEPTION String]
getState = do
  parser <- get
  pure (getRemaining parser)



p : Parser
p = MkParser 0 "<div class='test' name='jimmy'>Test</div>"




q : Either String (List Node)
q = run (do
  addParser "<div test='stuff'></div>"
  parseNodes)



showEnd : NodeType -> String
showEnd (Text x) = ""
showEnd (Element (MkElementData tagName attributes)) = "</" ++ tagName ++ ">"

instance Show AttrMap where
    show x = case toList x of
                  [] => ""
                  ((key, value) :: xs) => " " ++ key ++ "=\"" ++ value ++ "\"" ++ show (fromList xs)


instance Show ElementData where
    show (MkElementData tagName attributes) = "<" ++ tagName ++ (show attributes) ++ ">" 


instance Show NodeType where 
    show (Text x) = x
    show (Element x) = show x
    
instance Show Node where
    show (MkNode children nodeType) = show nodeType ++ unlines (map show children) ++ showEnd nodeType


