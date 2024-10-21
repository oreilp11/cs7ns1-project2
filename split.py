import os
import random

def main():
    path = 'train-imgs'
    dest = 'test-imgs'
    split = 0.2

    n = len(os.listdir(path))
    split_size = int(split*n)

    os.makedirs(dest, exist_ok=True)
    for _ in range(split_size):
        k = random.randint(0,n-1)
        file = os.listdir(path)[k]
        os.rename(os.path.join(path,file),os.path.join(dest,file))
        n = len(os.listdir(path))
        print(n, k, file)    
        

if __name__ == '__main__':
    main()