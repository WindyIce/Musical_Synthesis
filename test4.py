
import sys
import math

def solve(n,a):
    if n==1:
        if a[0]==0: return None
        return [-a[1]/float(a[0])]

    elif n==2:
        d=a[0]
        b=a[1]
        c=a[2]
        d=float(d)
        b=float(b)
        c=float(c)
        judge=b*b-4*d*c
        if judge<0: return None
        elif judge==0: return [-b/(d*2)]
        else:
            x1=(-b+math.sqrt(judge))/(2*d)
            x2=(-b-math.sqrt(judge))/(2*d)
            x1=round(x1,2)
            x2=round(x2,2)
            return [x1,x2]

    else: return None

if __name__ == "__main__":
    n= int(sys.stdin.readline().strip())
    ans = 0
    for i in range(n+1):
        # 读取每一行
        line = sys.stdin.readline().strip()
        a=  list(map(int, line.split()))

        ans=solve(n,a)
        if ans is None:print('No')
        else:
            for item in ans:
                print(item)

   # print(solve(2,[1,2,1]))
