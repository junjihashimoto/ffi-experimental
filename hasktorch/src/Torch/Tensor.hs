{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Tensor where

import Foreign.ForeignPtr
import System.IO.Unsafe

import Aten.Cast
import Aten.Class (Castable(..))
import qualified Aten.Managed.Type.Tensor as ATen
import qualified Aten.Type as ATen


type ATenTensor = ForeignPtr ATen.Tensor

data Tensor = Tensor ATenTensor

instance Castable Tensor ATenTensor where
    cast (Tensor aten_tensor) f = f aten_tensor
    uncast aten_tensor f = f $ Tensor aten_tensor

instance Show Tensor where
    show t = "<Tensor>"

debugPrint :: Tensor -> IO ()
debugPrint = cast1 ATen.tensor_print

numel :: Tensor -> Int
numel t = unsafePerformIO $ cast1 ATen.tensor_numel $ t

select :: Tensor -> Int -> Int -> Tensor
select t dim idx = unsafePerformIO $ (cast3 ATen.tensor_select_ll) t dim idx


asDouble :: Tensor -> Double
asDouble t = unsafePerformIO $ cast1 ATen.tensor_item_double $ t

asInt :: Tensor -> Int
asInt t = unsafePerformIO $ cast1 ATen.tensor_item_int64_t $ t


class TensorIndex a where
  (@@) :: Tensor -> a -> Tensor

instance {-# OVERLAPPABLE #-} Integral a => TensorIndex a where
  t @@ idx = select t 0 $ fromIntegral (toInteger idx)

instance (TensorIndex a, TensorIndex b) => TensorIndex (a,b) where
  t @@ (a, b) = (t @@ a) @@ b

instance (TensorIndex a, TensorIndex b, TensorIndex c) => TensorIndex (a,b,c) where
  t @@ (a, b, c) = (t @@ (a, b)) @@ c

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d) => TensorIndex (a,b,c,d) where
  t @@ (a, b, c, d) = (t @@ (a, b, c)) @@ c
