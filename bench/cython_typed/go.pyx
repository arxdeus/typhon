# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Typed Cython port of bench/codon/go.py (no argv).
# Entry point: module-level code runs versus_cpu() at import (see run.py).
# Optional env GO_GAMES / GO_ROUNDS override GAMES / outer loop count for
# quick correctness checks only. Python `random` is kept for identical
# sequences. Squares/Board/EmptySet/UCTNode are cdef classes with typed
# attributes; neighbour lists, hash set and UCT child list stay Python
# containers (dynamic, object-heavy benchmark).
import random
import os
from time import time
from libc.math cimport sqrt, log

DEF SIZE = 9
DEF NSQ = 81
GAMES = int(os.environ.get("GO_GAMES", 200))
ROUNDS = int(os.environ.get("GO_ROUNDS", 100))
cdef double KOMI = 7.5
DEF EMPTY = 0
DEF WHITE = 1
DEF BLACK = 2
DEF PASS = -1
DEF MAXMOVES = SIZE * SIZE * 3
cdef long TIMESTAMP = 0
cdef long MOVES = 0


cdef inline int to_pos(int x, int y):
    return y * SIZE + x


cdef class Square:
    cdef Board board
    cdef public int pos
    cdef long timestamp, removestamp
    cdef long long zobrist_strings[3]
    cdef list neighbours
    cdef public int color
    cdef Square reference
    cdef int ledges, temp_ledges
    cdef bint used

    def __init__(self, Board board, int pos):
        self.board = board
        self.pos = pos
        self.timestamp = TIMESTAMP
        self.removestamp = TIMESTAMP
        cdef int i
        for i in range(3):
            self.zobrist_strings[i] = random.randrange(9223372036854775807)

    cdef void set_neighbours(self):
        cdef int x = self.pos % SIZE, y = self.pos // SIZE
        cdef int newx, newy, dx, dy, k
        cdef int dxs[4]
        cdef int dys[4]
        dxs[:] = [-1, 1, 0, 0]
        dys[:] = [0, 0, -1, 1]
        self.neighbours = []
        for k in range(4):
            newx = x + dxs[k]
            newy = y + dys[k]
            if 0 <= newx < SIZE and 0 <= newy < SIZE:
                self.neighbours.append(self.board.squares[to_pos(newx, newy)])

    cdef void move(self, int color):
        global TIMESTAMP, MOVES
        cdef Square neighbour, neighbour_ref
        cdef int neighcolor
        TIMESTAMP += 1
        MOVES += 1
        self.board.zobrist.update(self, color)
        self.color = color
        self.reference = self
        self.ledges = 0
        self.used = True
        for neighbour in self.neighbours:
            neighcolor = neighbour.color
            if neighcolor == EMPTY:
                self.ledges += 1
            else:
                neighbour_ref = neighbour.find(True)
                if neighcolor == color:
                    if neighbour_ref.reference.pos != self.pos:
                        self.ledges += neighbour_ref.ledges
                        neighbour_ref.reference = self
                    self.ledges -= 1
                else:
                    neighbour_ref.ledges -= 1
                    if neighbour_ref.ledges == 0:
                        neighbour.remove(neighbour_ref, True)
        self.board.zobrist.add()

    cdef void remove(self, Square reference, bint update):
        cdef Square neighbour, neighbour_ref
        self.board.zobrist.update(self, EMPTY)
        self.removestamp = TIMESTAMP
        if update:
            self.color = EMPTY
            self.board.emptyset.add(self.pos)
        for neighbour in self.neighbours:
            if neighbour.color != EMPTY and neighbour.removestamp != TIMESTAMP:
                neighbour_ref = neighbour.find(update)
                if neighbour_ref.pos == reference.pos:
                    neighbour.remove(reference, update)
                else:
                    if update:
                        neighbour_ref.ledges += 1

    cdef Square find(self, bint update):
        cdef Square reference = self.reference
        if reference.pos != self.pos:
            reference = reference.find(update)
            if update:
                self.reference = reference
        return reference


cdef class EmptySet:
    cdef Board board
    cdef int empties[NSQ]
    cdef int nempties
    cdef int empty_pos[NSQ]

    def __init__(self, Board board):
        self.board = board
        cdef int i
        for i in range(NSQ):
            self.empties[i] = i
            self.empty_pos[i] = i
        self.nempties = NSQ

    cdef int random_choice(self):
        cdef int choices = self.nempties
        cdef int i, pos
        while choices:
            i = <int>(random.random() * choices)
            pos = self.empties[i]
            if self.board.useful(pos):
                return pos
            choices -= 1
            self.set(i, self.empties[choices])
            self.set(choices, pos)
        return PASS

    cdef inline void add(self, int pos):
        self.empty_pos[pos] = self.nempties
        self.empties[self.nempties] = pos
        self.nempties += 1

    cdef inline void remove(self, int pos):
        self.set(self.empty_pos[pos], self.empties[self.nempties - 1])
        self.nempties -= 1

    cdef inline void set(self, int i, int pos):
        self.empties[i] = pos
        self.empty_pos[pos] = i


cdef class ZobristHash:
    cdef set hash_set
    cdef long long hash

    def __init__(self, Board board):
        cdef Square square
        self.hash_set = set()
        self.hash = 0
        for square in board.squares:
            self.hash ^= square.zobrist_strings[EMPTY]
        self.hash_set.add(self.hash)

    cdef inline void update(self, Square square, int color):
        self.hash ^= square.zobrist_strings[square.color]
        self.hash ^= square.zobrist_strings[color]

    cdef inline void add(self):
        self.hash_set.add(self.hash)

    cdef inline bint dupe(self):
        return self.hash in self.hash_set


cdef class Board:
    cdef list squares
    cdef EmptySet emptyset
    cdef ZobristHash zobrist
    cdef public int color
    cdef bint finished
    cdef int lastmove
    cdef list history
    cdef int white_dead, black_dead

    def __init__(self):
        cdef Square square
        self.squares = [Square(self, pos) for pos in range(NSQ)]
        for square in self.squares:
            square.set_neighbours()
        self.reset()

    cdef void reset(self):
        cdef Square square
        for square in self.squares:
            square.color = EMPTY
            square.used = False
        self.emptyset = EmptySet(self)
        self.zobrist = ZobristHash(self)
        self.color = BLACK
        self.finished = False
        self.lastmove = -2
        self.history = []
        self.white_dead = 0
        self.black_dead = 0

    cdef void move(self, int pos):
        cdef Square square
        if pos != PASS:
            square = self.squares[pos]
            square.move(self.color)
            self.emptyset.remove(square.pos)
        elif self.lastmove == PASS:
            self.finished = True
        if self.color == BLACK:
            self.color = WHITE
        else:
            self.color = BLACK
        self.lastmove = pos
        self.history.append(pos)

    cdef inline int random_move(self):
        return self.emptyset.random_choice()

    cdef bint useful_fast(self, Square square):
        cdef Square neighbour
        if not square.used:
            for neighbour in square.neighbours:
                if neighbour.color == EMPTY:
                    return True
        return False

    cdef bint useful(self, int pos):
        global TIMESTAMP
        TIMESTAMP += 1
        cdef Square square = self.squares[pos]
        if self.useful_fast(square):
            return True
        cdef long long old_hash = self.zobrist.hash
        self.zobrist.update(square, self.color)
        cdef int empties = 0, opps = 0, weak_opps = 0, neighs = 0, weak_neighs = 0
        cdef int neighcolor
        cdef Square neighbour, neighbour_ref
        for neighbour in square.neighbours:
            neighcolor = neighbour.color
            if neighcolor == EMPTY:
                empties += 1
                continue
            neighbour_ref = neighbour.find(False)
            if neighbour_ref.timestamp != TIMESTAMP:
                if neighcolor == self.color:
                    neighs += 1
                else:
                    opps += 1
                neighbour_ref.timestamp = TIMESTAMP
                neighbour_ref.temp_ledges = neighbour_ref.ledges
            neighbour_ref.temp_ledges -= 1
            if neighbour_ref.temp_ledges == 0:
                if neighcolor == self.color:
                    weak_neighs += 1
                else:
                    weak_opps += 1
                    neighbour_ref.remove(neighbour_ref, False)
        cdef bint dupe = self.zobrist.dupe()
        self.zobrist.hash = old_hash
        cdef int strong_neighs = neighs - weak_neighs
        cdef int strong_opps = opps - weak_opps
        return not dupe and \
            (empties or weak_opps or (strong_neighs and (strong_opps or weak_neighs)))

    cdef list useful_moves(self):
        cdef list out = []
        cdef int k, pos
        for k in range(self.emptyset.nempties):
            pos = self.emptyset.empties[k]
            if self.useful(pos):
                out.append(pos)
        return out

    cdef void replay(self, list history):
        cdef int pos
        for pos in history:
            self.move(pos)

    cdef double score(self, int color):
        cdef double count
        cdef Square square, neighbour
        cdef int surround
        if color == WHITE:
            count = KOMI + self.black_dead
        else:
            count = self.white_dead
        for square in self.squares:
            if square.color == color:
                count += 1
            elif square.color == EMPTY:
                surround = 0
                for neighbour in square.neighbours:
                    if neighbour.color == color:
                        surround += 1
                if surround == len(square.neighbours):
                    count += 1
        return count


cdef class UCTNode:
    cdef UCTNode bestchild
    cdef int pos
    cdef long wins, losses
    cdef list pos_child
    cdef UCTNode parent
    cdef list unexplored

    def __init__(self):
        self.bestchild = None
        self.pos = -1
        self.wins = 0
        self.losses = 0
        self.pos_child = [None] * NSQ
        self.parent = None

    cdef void play(self, Board board):
        cdef int color = board.color
        cdef UCTNode node = self, child
        cdef list path = [node]
        cdef int pos
        while True:
            pos = node.select(board)
            if pos == PASS:
                break
            board.move(pos)
            child = node.pos_child[pos]
            if child is None:
                child = UCTNode()
                node.pos_child[pos] = child
                child.unexplored = board.useful_moves()
                child.pos = pos
                child.parent = node
                path.append(child)
                break
            path.append(child)
            node = child
        self.random_playout(board)
        self.update_path(board, color, path)

    cdef int select(self, Board board):
        cdef Py_ssize_t i, n
        cdef int pos
        if self.unexplored:
            n = len(self.unexplored)
            i = random.randrange(n)
            pos = self.unexplored[i]
            self.unexplored[i] = self.unexplored[n - 1]
            self.unexplored.pop()
            return pos
        elif self.bestchild is not None:
            return self.bestchild.pos
        else:
            return PASS

    cdef void random_playout(self, Board board):
        cdef int x
        for x in range(MAXMOVES):
            if board.finished:
                break
            board.move(board.random_move())

    cdef void update_path(self, Board board, int color, list path):
        cdef bint wins = board.score(BLACK) >= board.score(WHITE)
        cdef UCTNode node
        for node in path:
            if color == BLACK:
                color = WHITE
            else:
                color = BLACK
            if wins == (color == BLACK):
                node.wins += 1
            else:
                node.losses += 1
            if node.parent is not None:
                node.parent.bestchild = node.parent.best_child()

    cdef double score(self):
        cdef double winrate = self.wins / <double>(self.wins + self.losses)
        cdef long parentvisits = self.parent.wins + self.parent.losses
        if not parentvisits:
            return winrate
        cdef long nodevisits = self.wins + self.losses
        return winrate + sqrt(log(<double>parentvisits) / (5 * nodevisits))

    cdef UCTNode best_child(self):
        cdef double maxscore = -1
        cdef UCTNode maxchild = None, child
        for child in self.pos_child:
            if child is not None and child.score() > maxscore:
                maxchild = child
                maxscore = child.score()
        return maxchild

    cdef UCTNode best_visited(self):
        cdef long maxvisits = -1
        cdef UCTNode maxchild = None, child
        for child in self.pos_child:
            if child is not None and (child.wins + child.losses) > maxvisits:
                maxvisits = child.wins + child.losses
                maxchild = child
        return maxchild


cdef int computer_move(Board board, int games):
    cdef int pos = board.random_move()
    if pos == PASS:
        return PASS
    cdef UCTNode tree = UCTNode(), node
    tree.unexplored = board.useful_moves()
    cdef Board nboard = Board()
    cdef int game
    for game in range(games):
        node = tree
        nboard.reset()
        nboard.replay(board.history)
        node.play(nboard)
    return tree.best_visited().pos


def versus_cpu():
    cdef int i, games = GAMES, mv
    debug = os.environ.get("GO_DEBUG")
    for i in range(ROUNDS):
        random.seed(i)
        board = Board()
        mv = computer_move(board, games)
        if debug:
            print("move", i, mv)


t0 = time()
versus_cpu()
t1 = time()
print(t1 - t0)
