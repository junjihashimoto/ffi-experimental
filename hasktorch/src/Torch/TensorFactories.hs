{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE FlexibleContexts #-}

module Torch.TensorFactories where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified Aten.Const as ATen
import qualified Aten.Managed.Native as ATen
import qualified Aten.Managed.Type.TensorOptions as ATen
import qualified Aten.Type as ATen
import Aten.Managed.Cast
import Aten.Class (Castable(..))
import Aten.Cast

import Torch.Tensor
import Torch.TensorOptions

defaultTensorOptions :: TensorOptions
defaultTensorOptions = TensorOptions $ unsafePerformIO $ ATen.newTensorOptions_s ATen.kFloat

type FactoryType = ForeignPtr ATen.IntArray
                    -> ForeignPtr ATen.TensorOptions
                    -> IO (ForeignPtr ATen.Tensor)

mkFactory :: FactoryType -> [Int] -> TensorOptions -> IO Tensor
mkFactory aten_impl shape opts = (cast2 aten_impl) shape opts

mkFactoryUnsafe :: FactoryType -> [Int] -> TensorOptions -> Tensor
mkFactoryUnsafe f shape opts = unsafePerformIO $ mkFactory f shape opts

mkDefaultFactory :: ([Int] -> TensorOptions -> a) -> [Int] -> a
mkDefaultFactory non_default shape = non_default shape defaultTensorOptions

-------------------- Factories --------------------

ones :: [Int] -> TensorOptions -> Tensor
ones = mkFactoryUnsafe ATen.ones_lo

zeros :: [Int] -> TensorOptions -> Tensor
zeros = mkFactoryUnsafe ATen.zeros_lo

rand :: [Int] -> TensorOptions -> IO Tensor
rand = mkFactory ATen.rand_lo

randn :: [Int] -> TensorOptions -> IO Tensor
randn = mkFactory ATen.randn_lo

-------------------- Factories with default type --------------------

ones' :: [Int] -> Tensor
ones' = mkDefaultFactory ones

zeros' :: [Int] -> Tensor
zeros' = mkDefaultFactory zeros

rand' :: [Int] -> IO Tensor
rand' = mkDefaultFactory rand

randn' :: [Int] -> IO Tensor
randn' = mkDefaultFactory randn
