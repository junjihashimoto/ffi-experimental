
module Torch.Functions where

import System.IO.Unsafe

import qualified Aten.Managed.Native as ATen
import Aten.Cast

import Torch.Tensor

sumAll :: Tensor -> Tensor
sumAll t = unsafePerformIO $ (cast1 ATen.sum_t) t

