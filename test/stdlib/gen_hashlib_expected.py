# Generates expected literals for test/stdlib/hashlib_test.py
import hashlib
print("EXPECTED = {")
for n in ["md5","sha1","sha224","sha256","sha384","sha512"]:
    for L in [0,1,3,55,56,63,64,65,111,112,127,128,129,1000003]:
        s = "".join([chr(97 + (i*7)%26) for i in range(L)])
        print(f'    ("{n}", {L}): "{hashlib.new(n, s.encode()).hexdigest()}",')
print("}")
