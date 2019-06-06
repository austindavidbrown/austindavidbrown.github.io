def lcg(N):
  modulus = 2**(31) - 1
  a = 48271
  c = 0
  seed = 123
  arr = []
  for i in range(0, N):
    seed = (a * seed + c) % modulus
    arr.append(seed/modulus)
  return arr


lcg(10)