{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module ShapeSpec (spec) where

import Test.Hspec
import Control.Exception.Safe

import ATen.Managed.Type.Tensor
import Torch.Static.Native

spec :: Spec
spec = do
  it "check shape" $ do
    z <- zeros :: IO (Tensor [2,2,2] Double)
    tensor_print (toDynamic z)
    tensor_dim (toDynamic z) `shouldReturn` 3

