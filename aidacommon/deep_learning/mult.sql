CREATE OR REPLACE FUNCTION mult()
  RETURNS TABLE(module text)
AS $$
  import numpy as np
  import time as time

  start_t = time.time()
  A = np.random.rand(100000,10)
  B = np.random.rand(10,4)

  for x in range(400000):
    C = A.dot(B)

  end_t = time.time()
  duration = end_t - start_t
  return str(duration);
$$ LANGUAGE plpython3u;
SELECT * FROM mult();
