{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Autograd where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified Torch.Managed.Autograd
import qualified Aten.Managed.Type.Tensor as ATen
import qualified Aten.Type as ATen
import Aten.Class
import Aten.Cast

import Torch.Tensor

-- NB: ATen only defines Castable [ForeignPtr ATen.Tensor] (ForeignPtr ATen.TensorList)
instance Castable [Tensor] (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list

grad :: Tensor -> [Tensor] -> [Tensor]
grad y inputs = unsafePerformIO $ (cast2 Torch.Managed.Autograd.grad) y inputs

requiresGrad :: Tensor -> Bool
requiresGrad t = unsafePerformIO $ (cast1 ATen.tensor_requires_grad) t

independent :: Tensor -> Tensor
independent t | not (requiresGrad t) = t
              | otherwise = unsafePerformIO $ (cast1 Torch.Managed.Autograd.makeIndependent) t
