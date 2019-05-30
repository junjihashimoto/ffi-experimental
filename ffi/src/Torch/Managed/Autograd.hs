
module Torch.Managed.Autograd where

import Foreign.ForeignPtr

import qualified Torch.Unmanaged.Autograd as Unmanaged
import qualified Aten.Unmanaged.Type.Tensor
import qualified Aten.Unmanaged.Type.TensorList
import Aten.Type
import Aten.Class
import Aten.Cast


grad :: ForeignPtr Tensor -> ForeignPtr TensorList -> IO (ForeignPtr TensorList)
grad = cast2 Unmanaged.grad
