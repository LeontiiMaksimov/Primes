import math, time
import gmpy2

def print_result(exp, *, tb = time.time()):
    print(exp, 'at', round(time.time() - tb, 3), 'secs', flush = True, end = ',   \n')
    
def mersennes_v1():
    def is_prime(n):
        if n <= 2:
            return n == 2
        if n & 1 == 0:
            return False
        for j in range(3, math.floor(math.sqrt(n + 0.1)) + 1, 2):
            if n % j == 0:
                return False
        return True

    def lucas_lehmer_test(p):
        # Lucas Lehmer Test https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test
        
        mask = gmpy2.mpz((1 << p) - 1)

        def mer_rem(x, bits):
            # Below is same as:   return x % mask
            while True:
                r = gmpy2.mpz(0)
                while x != 0:
                    r += x & mask
                    x >>= bits
                if r == mask:
                    r = gmpy2.mpz(0)
                if r < mask:
                    return r
                x = r

        s = 4
        s, p = gmpy2.mpz(s), gmpy2.mpz(p)
        for k in range(2, p):
            s = mer_rem(s * s - 2, p)
        return s == 0
        
    for p in range(3, 1 << 30):
        if is_prime(p) and lucas_lehmer_test(p):
            print_result(p)

mersennes_v1()