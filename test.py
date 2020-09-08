#coding=utf-8
import sys

def max_length(n,a):
    max=0
    now_length=0
    front_length=1
    back_length=1
    state_down=True
    for i in range(n-1):
        if state_down:
            if a[i + 1] < a[i]:
                front_length += 1
            elif a[i+1]==a[i]:
                continue
            else:
                if front_length>1:
                    state_down=False
                    back_length=1

        else:
            if a[i+1]>a[i]:
                back_length+=1
            elif a[i+1]==a[i]:
                now_length=min(front_length,back_length)*2
                if now_length>max: max=now_length
                front_length = 1
                back_length = 1
                state_down=True
            else:
                now_length = min(front_length, back_length) * 2
                if now_length>max:max=now_length
                front_length = 2
                back_length = 1
                state_down = True
            now_length = min(front_length, back_length) * 2




    if now_length>max:max=now_length


    return int(max)





if __name__ == "__main__":
    T= int(sys.stdin.readline().strip())
    ans = 0
    for i in range(T):
        # 读取每一行
        n = int(sys.stdin.readline().strip())
        line =  sys.stdin.readline().strip()
        a=list(map(int, line.split()))
        ans=max_length(int(n),a)
        print(ans)



    #rint(max_length(14,[87,70,17,12,14,86,61,51,12,90,69,89,4,65]))

