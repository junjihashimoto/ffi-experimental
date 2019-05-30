{-# LANGUAGE RecordWildCards #-}

module Main where

import Torch.Tensor
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions

data LinearParams = LParams { weight :: Tensor, bias :: Tensor }
  deriving (Show)


get_params :: LinearParams -> [Tensor]
get_params LParams{..} = [weight, bias]

random_params :: Int -> Int -> IO LinearParams
random_params in_features out_features = do
    w <- randn [in_features, out_features] opts
    b <- randn [out_features] opts
    return $ LParams w b
  where
    opts = withRequiresGrad defaultOpts

-- TODO: sigmoid!
linear :: LinearParams -> Tensor -> Tensor
linear LParams{..} input = (matmul input weight) + bias

mkLayerSizes :: Int -> [Int] -> [(Int, Int)]
mkLayerSizes input_size features =
    tail $ scanl shift (0, input_size) features
  where
    shift (a, b) c = (b, c)

batch_size = 32

main :: IO ()
main = do
  params <- mapM (uncurry random_params) $ mkLayerSizes 2 [10, 10]
  let flat_params = map get_params params
  let layers = map linear params
  let model = foldl (.) id $ map (relu .) layers

  input <- rand' [batch_size, 2] >>= return . (gt 0.5)
  putStrLn $ show input
  let output = model input
  putStrLn "Done!"
