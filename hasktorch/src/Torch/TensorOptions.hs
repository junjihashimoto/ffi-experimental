{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.TensorOptions where

import Foreign.ForeignPtr
import System.IO.Unsafe

import Aten.Class (Castable(..))
import qualified Aten.Type as ATen
import qualified Aten.Const as ATen
import qualified Aten.Managed.Type.TensorOptions as ATen

type ATenTensorOptions = ForeignPtr ATen.TensorOptions

data TensorOptions = TensorOptions ATenTensorOptions

instance Castable TensorOptions ATenTensorOptions where
  cast (TensorOptions aten_opts) f = f aten_opts
  uncast aten_opts f = f $ TensorOptions aten_opts
