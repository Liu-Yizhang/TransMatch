import os
import sys
import time

cmd = 'python ./main.py'


def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_total = int(gpu_status[2].split('/')[1].split('M')[0].strip())*2
    gpu_memory3 = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_memory4 = int(gpu_status[6].split('/')[0].split('M')[0].strip())
    gpu_memory_used = gpu_memory3 + gpu_memory4
    gpu_left = gpu_total - gpu_memory_used
    return gpu_left


def narrow_setup(interval=2):
    gpu_left = gpu_info()
    i = 0
    while gpu_left < 47000:  # set waiting condition
        gpu_left = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_left_str = 'gpu memory left:%d MiB |' % gpu_left
        sys.stdout.write('\r' + gpu_left_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()
