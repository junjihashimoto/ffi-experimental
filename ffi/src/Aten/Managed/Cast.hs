{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Aten.Managed.Cast where

import Foreign.ForeignPtr
import Control.Monad

import Aten.Class
import Aten.Cast
import Aten.Type
import Aten.Managed.Type.IntArray

instance Castable [Int] (ForeignPtr IntArray) where
  cast xs f = do
    arr <- newIntArray
    forM_ xs $ (intArray_push_back_l arr) . fromIntegral
    f arr
  uncast xs f = undefined
