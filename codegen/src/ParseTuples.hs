{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseTuples where

import Data.Char (toUpper)
import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import Text.Show.Prettyprint (prettyPrint)
import qualified ParseFunctionSig as S

{- spec/tuples.yaml -}

data Tuple = Tuple {
  types :: [S.Parsable]
} deriving (Show, Eq, Generic)

instance FromJSON Tuple
