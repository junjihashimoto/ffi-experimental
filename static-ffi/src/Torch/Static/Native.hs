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


module Torch.Static.Native where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Managed.Type.IntArray as ATen
import qualified ATen.Managed.Type.TensorOptions as ATen
import qualified ATen.Managed.Type.Scalar as ATen
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen
import qualified ATen.Managed.Cast
import ATen.Class
import ATen.Cast
import Data.Proxy
import GHC.TypeLits
import Numeric.Dimensions
import Control.Monad (forM_)

data Tensor (n::[Nat]) d  = ToStatic {
  toDynamic :: ForeignPtr ATen.Tensor
}

class MetaTensor a where
  shape :: [Word]
  dtype :: ATen.ScalarType

instance (Dimensions n) => MetaTensor (Tensor n Double) where
  shape = listDims (dims @Nat @n)
  dtype = ATen.kDouble

instance Castable [Word] (ForeignPtr ATen.IntArray) where
  cast xs f = do
    arr <- ATen.newIntArray
    forM_ xs $ (ATen.intArray_push_back_l arr) . fromIntegral
    f arr
  uncast xs f = undefined

instance Castable ATen.ScalarType (ForeignPtr ATen.TensorOptions) where
  cast x f = ATen.newTensorOptions_s x >>= f
  uncast x f = undefined

instance Castable (Tensor n d) (ForeignPtr ATen.Tensor) where
  cast x f = f (toDynamic x)
  uncast x f = f (ToStatic x)

zeros :: forall n d. (Dimensions n, MetaTensor (Tensor n d)) => IO (Tensor n d)
zeros = cast2 ATen.zeros_lo (shape @(Tensor n d))(dtype @(Tensor n d))

ones :: forall n d. (Dimensions n, MetaTensor (Tensor n d)) => IO (Tensor n d)
ones = cast2 ATen.ones_lo (shape @(Tensor n d))(dtype @(Tensor n d))

empty :: forall n d. (Dimensions n, MetaTensor (Tensor n d)) => IO (Tensor n d)
empty = cast2 ATen.empty_lo (shape @(Tensor n d))(dtype @(Tensor n d))

sin :: forall n d. (Dimensions n, MetaTensor (Tensor n d)) => Tensor n d -> IO (Tensor n d)
sin = cast1 ATen.sin_t

cos :: forall n d. (Dimensions n, MetaTensor (Tensor n d)) => Tensor n d -> IO (Tensor n d)
cos = cast1 ATen.cos_t
