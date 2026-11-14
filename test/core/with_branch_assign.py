# Assignments in if/else branches inside with/try/loops update the outer variable.
import threading
lock = threading.RLock()
def w_if(x: int) -> int:
    with lock:
        if x > 1:
            cmd = 7
        else:
            cmd = 0
    return cmd

def w_elif(x: int) -> int:
    with lock:
        if x > 5:
            cmd = 2
        elif x > 1:
            cmd = 1
        else:
            cmd = 0
    return cmd

def w_float(x: int) -> float:
    with lock:
        if x > 1:
            v = 1.5
        else:
            v = -2.25
    return v

def w_nested(x: int) -> int:
    with lock:
        with lock:
            if x > 1:
                cmd = 7
            else:
                cmd = 3
    return cmd

def t_finally(x: int) -> int:
    try:
        if x > 1:
            cmd = 7
        else:
            cmd = 0
    finally:
        pass
    return cmd

def t_except(x: int) -> int:
    try:
        if x > 1:
            cmd = 7
        else:
            cmd = 0
    except ValueError:
        cmd = -1
    return cmd

def w_for(n: int) -> int:
    with lock:
        cmd = 0
        for i in range(n):
            if i % 2 == 0:
                cmd = cmd + i
            else:
                cmd = cmd - 1
    return cmd

def w_while(n: int) -> int:
    with lock:
        cmd = 0
        i = 0
        while i < n:
            if i > 2:
                cmd = 9
            else:
                cmd = i
            i += 1
    return cmd

def w_reassign(x: int) -> int:
    with lock:
        cmd = 1
        if x > 1:
            cmd = 7
    return cmd

def plain(x: int) -> int:
    if x > 1:
        cmd = 7
    else:
        cmd = 0
    return cmd

print(w_if(0), w_elif(0), w_float(0), w_nested(0), t_finally(0), t_except(0), w_for(0), w_while(0), w_reassign(0), plain(0))  #: 0 0 -2.25 3 0 0 0 0 1 0
print(w_if(3), w_elif(3), w_float(3), w_nested(3), t_finally(3), t_except(3), w_for(3), w_while(3), w_reassign(3), plain(3))  #: 7 1 1.5 7 7 7 1 2 7 7
print(w_if(9), w_elif(9), w_float(9), w_nested(9), t_finally(9), t_except(9), w_for(9), w_while(9), w_reassign(9), plain(9))  #: 7 2 1.5 7 7 7 16 9 7 7
