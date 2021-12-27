import math, time, multiprocessing, multiprocessing.pool, multiprocessing.shared_memory, sys, os, faulthandler, traceback, secrets
faulthandler.enable()
import numpy as np, gmpy2, colorama
colorama.init()

def is_mersenne(args, *, data = {'tf': 0}):
    try:
        def lucas_lehmer_test(exp):
            # Lucas Lehmer Test https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test

            def num(n):
                return gmpy2.mpz(n)
            
            mask = num((1 << exp) - 1)
            
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

            tb = time.time()
            
            s, exp, two = num(4), num(exp), num(2)
            for k in range(2, exp):
                s = mer_rem(s * s - two, exp)

            #print('ll', '-+'[s==0], '_', int(exp), '_', round(time.time() - tb, 3), sep = '', end = ' ', flush = True)
            return s == 0

        def trial_div_test(exp):
            import numpy as np
            if 'ptaba' not in data:
                data['shp'] = multiprocessing.shared_memory.SharedMemory(args['all_primes_shname'])
                data['ptaba'] = np.ndarray((args['all_primes_size'] // 4,), dtype = np.uint32, buffer = data['shp'].buf)
            if exp <= 32:
                return True, {'bits': 0}
            bits = max(16, min(-19 + math.floor(math.log(exp) / math.log(1.25)), math.ceil(args['prime_bits'])))
            if ('ptabb', bits) not in data:
                cnt = min(np.searchsorted(data['ptaba'], 1 << bits), len(data['ptaba']))
                data[('ptabb', bits)] = np.ndarray((cnt,), dtype = np.uint32, buffer = data['shp'].buf)
            sptab = data[('ptabb', bits)]
            tb, bl, probably_prime = time.time(), 1 << 13, True
            spows = np.empty((bl,), dtype = np.uint64)
            for i in range(0, sptab.size, bl):
                ptab = sptab[i : i + bl]
                pows = spows if spows.size == ptab.size else spows[:ptab.size]
                pows[...] = 2
                for b in bin(exp)[3:]:
                    pows *= pows
                    if b == '1':
                        pows <<= 1
                    pows %= ptab
                if np.count_nonzero(pows == 1) > 0:
                    probably_prime = False
                    break
            #print('td', '-+'[probably_prime], '_', int(exp), '_', round(time.time() - tb, 3), sep = '', end = ' ', flush = True)
            return probably_prime, {'bits': bits}

        p, stats = args['p'], {'tt': time.time()}
        r, tinfo = trial_div_test(p)
        stats['tt'], stats['tr'], stats['tte'], stats['tinfo'] = time.time() - stats['tt'], r, time.time(), tinfo
        if not r:
            return p, r, stats
        stats['lt'] = time.time()
        r = lucas_lehmer_test(p)
        stats['lt'], stats['lte'] = time.time() - stats['lt'], time.time()
        return p, r, stats
    except:
        return None, True, '', traceback.format_exc()
    
def mersennes_v2():
    prime_bits = 32
    num_cores = multiprocessing.cpu_count()
    console_width = os.get_terminal_size().columns

    def gen_primes(stop, *, dtype = 'int64'):
        # https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
        import math, numpy as np
        if stop < 2:
            return np.zeros((0,), dtype = dtype)
        primes = np.ones((stop >> 1,), dtype = np.uint8)
        primes[0] = False
        for p in range(3, math.floor(math.sqrt(stop)) + 1, 2):
            if primes[p >> 1]:
                primes[(p * p) >> 1 :: p] = False
        return np.concatenate((np.array([2], dtype = dtype), (np.flatnonzero(primes).astype(dtype) << 1) + 1))
        
    def RoundFix(x, n):
        s = str(round(x, n))
        return s + '0' * (n - len(s) + ('.' + s).rfind('.'))
        
    prime_bits = max(24, min(prime_bits, math.log(math.sqrt((1 << 63) - 1)) / math.log(2)))
    print(f'Generating {round(prime_bits, 2)}-bits primes...', end = ' ', flush = True)
    tb = time.time()
    all_primes = gen_primes(math.floor(2 ** prime_bits), dtype = 'uint32')
    print(f'{round(time.time() - tb)} secs.', flush = True)
    all_primes_size = len(all_primes.data) * all_primes.dtype.itemsize
    all_primes_shname = secrets.token_hex(16).upper()
    shp = multiprocessing.shared_memory.SharedMemory(
        name = all_primes_shname, create = True, size = all_primes_size)
    shp.buf[:all_primes_size] = all_primes.tobytes()
    del all_primes
    all_primes = np.ndarray((all_primes_size // 4,), dtype = np.uint32, buffer = shp.buf)

    print('Using', num_cores, 'cores.', flush = True)
    print('\n\n')
    offx, tstart, tlast, ptimes = 0, time.time(), time.time(), []
    tstats = {'tt': 0.0, 'tc': 0, 'tts': [], 'tta': 0.0, 'lt': 0.0, 'lc': 0, 'lts': [], 'lta': 0.0, 'st': [(0, 0, time.time())], 'tbits': []}
    with multiprocessing.Pool(num_cores) as pool:
        for e in pool.imap(is_mersenne, ({
            'p': int(e), 'all_primes_size': all_primes_size, 'all_primes_shname': all_primes_shname, 'prime_bits': prime_bits,
        } for e in all_primes)):
            if e[0] is None:
                pool.terminate()
                print('!!!Exception!!!\n', e[3], sep = '', flush = True)
                break
            stats = e[2]
            def fill(p):
                tstats[f'{p}t'] += stats[f'{p}t']
                tstats[f'{p}ts'] += [(stats[f'{p}t'], stats[f'{p}te'])]
                while len(tstats[f'{p}ts']) > 20 and tstats[f'{p}ts'][-1][1] - tstats[f'{p}ts'][0][1] > 120:
                    tstats[f'{p}ts'] = tstats[f'{p}ts'][1:]
                tstats[f'{p}c'] += 1
                tstats[f'{p}ta'] = sum(e[0] for e in tstats[f'{p}ts']) / len(tstats[f'{p}ts'])
                if p == 't':
                    tstats['st'] += [(stats['tt'] + stats.get('lt', 0), stats.get('lt', tstats['st'][-1][1]), stats.get('lte', stats['tte']))]
                    while len(tstats['st']) > 50 and tstats['st'][-1][2] - tstats['st'][0][2] > 300:
                        tstats['st'] = tstats['st'][1:]
                    tstats['sta'] = sum(e[1] for e in tstats['st']) / max(0.001, sum(e[0] for e in tstats['st']))
                    tstats['tbits'] = (tstats['tbits'] + [stats['tinfo']['bits']])[-20:]
                    tstats['tbitsa'] = sum(tstats['tbits']) / len(tstats['tbits'])
            fill('t')
            if 'lt' in stats:
                fill('l')
            if not e[1]:
                s0 = f'{str(e[0]).rjust(6)}| trial:{RoundFix(stats["tt"], 3)}/lucas:' + (f'{RoundFix(stats["lt"], 3)}' if 'lt' in stats else '-' * len(RoundFix(tstats['lta'], 3))) + f' secs (avg t:{RoundFix(tstats["tta"], 3)}/l:{RoundFix(tstats["lta"], 3)})    '
                s1 = f'{"".rjust(6)}| cnt t:{tstats["tc"]}({tstats["tc"] - tstats["lc"]})/l:{tstats["lc"]}, tboost:{RoundFix(tstats["sta"], 3)}x tbits:{RoundFix(tstats["tbitsa"], 2)}     '
                print('\033[2A' + s0[:console_width - 4] + '\n' + s1[:console_width - 4] + '\n', end = '', flush = True)
            else:
                s = str(e[0]).rjust(6) + ' at ' + str(round(time.time() - tstart)).rjust(6) + ' secs, '
                if offx + len(s) <= console_width - 4:
                    print('\033[3A\033[' + str(offx) + 'C' + s + '\n\n\n', end = '', flush = True)
                else:
                    print('\033[2A' + (' ' * (console_width - 4)) + '\n\033[1A' + s + '\n\n\n', end = '', flush = True)
                    offx = 0
                offx += len(s)
            tlast = time.time()
                
if __name__ == '__main__':
    mersennes_v2()