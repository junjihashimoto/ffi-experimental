{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.TensorOptions where

import Foreign.ForeignPtr
import System.IO.Unsafe

import Aten.Cast
import Aten.Class (Castable(..))
import qualified Aten.Type as ATen
import qualified Aten.Const as ATen
import qualified Aten.Managed.Type.TensorOptions as ATen

import Torch.DType

type ATenTensorOptions = ForeignPtr ATen.TensorOptions

data TensorOptions = TensorOptions ATenTensorOptions

instance Castable TensorOptions ATenTensorOptions where
  cast (TensorOptions aten_opts) f = f aten_opts
  uncast aten_opts f = f $ TensorOptions aten_opts

defaultOpts :: TensorOptions
defaultOpts = TensorOptions $ unsafePerformIO $ ATen.newTensorOptions_s ATen.kFloat

withGrad :: Bool -> TensorOptions -> TensorOptions
withGrad does_it opts = unsafePerformIO $ (cast2 ATen.tensorOptions_requires_grad_b) opts does_it

withRequiresGrad :: TensorOptions -> TensorOptions
withRequiresGrad = withGrad True

withNoGrad :: TensorOptions -> TensorOptions
withNoGrad = withGrad False

withDType :: DType -> TensorOptions -> TensorOptions
withDType dtype opts = unsafePerformIO $ (cast2 ATen.tensorOptions_dtype_s) opts dtype
