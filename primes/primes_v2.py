import math, time
import numba

def print_result(exp, *, tb = time.time()):
    print(exp, 'at', round(time.time() - tb, 3), 'secs', flush = True, end = ',   \n')
    
@numba.njit
def mersennes_v0():
    def is_prime(n):
        if n <= 2:
            return n == 2
        if n & 1 == 0:
            return False
        for j in range(3, math.floor(math.sqrt(n + 0.1)) + 1, 2):
            if n % j == 0:
                return False
        return True

    for i in range(2, 1000):
        if is_prime(i) and is_prime((1 << i) - 1):
            with numba.objmode():
                print_result(i)

mersennes_v0()
input()