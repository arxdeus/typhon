# Ported from CPython 3.14 graphlib (PSF license)

class CycleError(ValueError):
    pass

class TopologicalSorter[T]:
    _nodes: Dict[T, int]          # node -> index
    _names: List[T]
    _npred: List[int]
    _succ: List[List[int]]
    _ready: List[int]
    _ready_head: int
    _prepared: bool
    _nfinished: int
    _nvisited: int
    _done: List[bool]
    _passed: List[bool]

    def __init__(self):
        self._nodes = {}
        self._names = []
        self._npred = []
        self._succ = []
        self._ready = []
        self._ready_head = 0
        self._prepared = False
        self._nfinished = 0
        self._nvisited = 0
        self._done = []
        self._passed = []

    def _get(self, n: T) -> int:
        if n in self._nodes:
            return self._nodes[n]
        i = len(self._names)
        self._nodes[n] = i
        self._names.append(n)
        self._npred.append(0)
        self._succ.append([])
        self._done.append(False)
        self._passed.append(False)
        return i

    def add(self, node: T, *predecessors):
        if self._prepared:
            raise ValueError("Nodes cannot be added after a call to prepare()")
        i = self._get(node)
        for p in predecessors:
            j = self._get(p)
            self._npred[i] += 1
            self._succ[j].append(i)

    def find_cycle(self) -> Optional[List[T]]:
        n = len(self._names)
        # predecessor lists
        preds = [List[int]() for _ in range(n)]
        for j in range(n):
            for i in self._succ[j]:
                preds[i].append(j)
        state = [0] * n
        for s in range(n):
            if state[s] != 0:
                continue
            stack = [s]
            idx = [0]
            state[s] = 1
            path = [s]
            while len(stack) > 0:
                v = stack[-1]
                k = idx[-1]
                if k < len(preds[v]):
                    idx[-1] = k + 1
                    w = preds[v][k]
                    if state[w] == 0:
                        state[w] = 1
                        stack.append(w)
                        idx.append(0)
                        path.append(w)
                    elif state[w] == 1:
                        pos = 0
                        while path[pos] != w:
                            pos += 1
                        res = List[T]()
                        for q in range(pos, len(path)):
                            res.append(self._names[path[q]])
                        res.append(self._names[w])
                        return res
                else:
                    state[v] = 2
                    stack.pop()
                    idx.pop()
                    path.pop()
        return None

    def prepare(self):
        if self._prepared:
            raise ValueError("cannot prepare() more than once")
        self._prepared = True
        cycle = self.find_cycle()
        if cycle is not None:
            raise CycleError("nodes are in a cycle")
        for i in range(len(self._names)):
            if self._npred[i] == 0:
                self._ready.append(i)

    def get_ready(self) -> List[T]:
        if not self._prepared:
            raise ValueError("prepare() must be called first")
        res = List[T]()
        while self._ready_head < len(self._ready):
            i = self._ready[self._ready_head]
            self._ready_head += 1
            self._passed[i] = True
            self._nvisited += 1
            res.append(self._names[i])
        return res

    def is_active(self) -> bool:
        if not self._prepared:
            raise ValueError("prepare() must be called first")
        return self._nfinished < self._nvisited or self._ready_head < len(self._ready)

    def __bool__(self) -> bool:
        return self.is_active()

    def done(self, *nodes):
        if not self._prepared:
            raise ValueError("prepare() must be called first")
        for n in nodes:
            if n not in self._nodes:
                raise ValueError(f"node {n!r} was not added using add()")
            i = self._nodes[n]
            if self._done[i]:
                raise ValueError(f"node {n!r} was already marked done")
            if not self._passed[i]:
                raise ValueError(f"node {n!r} was not passed out (still not ready)")
            self._done[i] = True
            self._nfinished += 1
            for s in self._succ[i]:
                self._npred[s] -= 1
                if self._npred[s] == 0:
                    self._ready.append(s)

    def static_order(self):
        self.prepare()
        while self.is_active():
            node_group = self.get_ready()
            for node in node_group:
                yield node
            for node in node_group:
                self.done(node)
