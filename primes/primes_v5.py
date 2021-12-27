import math, time, multiprocessing
import gmpy2

def print_result(exp, *, tb = time.time()):
    print(exp, 'at', round(time.time() - tb, 3), 'secs', flush = True, end = ',   \n')

def lucas_lehmer_test(p):
    # Lucas Lehmer Test https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test

    def num(n):
        return gmpy2.mpz(n)
    
    mask = num((1 << p) - 1)
    
    def mer_rem(x, bits):
        # Below is same as:   return x % mask
        while True:
            r = num(0)
            while x != 0:
                r += x & mask
                x >>= bits
            if r == mask:
                r = num(0)
            if r < mask:
                return r
            x = r

    s, p, two = num(4), num(p), num(2)
    for k in range(2, p):
        s = mer_rem(s * s - two, p)
    return p, s == 0
    
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

    print('Num Cores Used:', multiprocessing.cpu_count())

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for p, _ in filter(lambda e: e[1], pool.imap(lucas_lehmer_test, filter(is_prime, range(3, 1 << 30)))):
            print_result(p)

if __name__ == '__main__':
    mersennes_v1()