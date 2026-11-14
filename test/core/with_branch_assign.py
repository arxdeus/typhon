import threading
lock = threading.Lock()
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

for x in (0, 3, 9):
    print(w_if(x), w_elif(x), w_float(x), w_nested(x), t_finally(x), t_except(x), w_for(x), w_while(x), w_reassign(x), plain(x))
