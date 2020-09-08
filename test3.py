#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。



import sys

def solve(n,all_arrays):
    for i in range(n):
        for k in range(n-i-1):
            t=k+i+1
            if all_arrays[i]==all_arrays[t]:
                return True

    return False



if __name__ == "__main__":
    T = int(sys.stdin.readline().strip())
    ans = 0
    for i in range(T):
        # 读取每一行
        n = int(sys.stdin.readline().strip())

        all_arrays=[]

        for j in range(n):
            line = sys.stdin.readline().strip()
            values = list(map(int, line.split()))
            values.sort()
            all_arrays.append(values)

        if solve(n,all_arrays):
            print('YES')
        else:
            print('NO')



    #print(solve(2,[[1,2,3,4,5,6],[7,5,7,2,5,8]]))