import math, time

def print_result(exp, *, tb = time.time()):
    print(exp, 'at', round(time.time() - tb, 3), 'secs', flush = True, end = ',   \n')
    
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

    for i in range(2, 12):
        if is_prime(i) and is_prime((1 << i) - 1):
            print_result(i)

mersennes_v0()
print("hit enter to close")
input()