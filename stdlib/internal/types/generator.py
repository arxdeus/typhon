
@extend
class Generator:
    def _yield_final(val):
        """Compiler-generated."""
        pass

    def _yield_in_no_suspend(T: type) -> T:
        """Compiler-generated."""
        pass

    @__internal__
    def __promise__(self) -> Ptr[T]:
        pass

    def _advance(self) -> bool:
        """Resume once unless exhausted; return whether a value was yielded."""
        if self.__done__():
            return False
        self.__resume__()
        return not self.__done__()

    def _get(self: Generator[T]) -> T:
        """Read the current value after a successful `_advance()`, without resuming."""
        if isinstance(T, None):
            pass
        else:
            return self.__promise__()[0]

    def __next__(self: Generator[T], /) -> T:
        if not self._advance():
            raise StopIteration()
        return self._get()

    def __iter__(self, /) -> Generator[T]:
        return self

    @pure
    @derives
    @llvm
    def __raw__(self) -> Ptr[u8]:
        ret ptr %self

    @pure
    @derives
    @llvm
    def __new__(ptr: cobj) -> Generator[T]:
        ret ptr %ptr

    @overload
    def __new__() -> Generator[T]:
        compile_error("invalid generator")

    @pure
    @llvm
    def __done__(self) -> bool:
        declare i1 @llvm.coro.done(ptr nocapture readonly)
        %0 = call i1 @llvm.coro.done(ptr %self)
        %1 = zext i1 %0 to i8
        ret i8 %1

    @nocapture
    @llvm
    def __resume__(self) -> None:
        declare void @llvm.coro.resume(ptr)
        call void @llvm.coro.resume(ptr %self)
        ret {} {}

    def __repr__(self, /) -> str:
        return Ptr._ptr_to_str(self.__raw__(), "generator")

    def send(self, what: T, /) -> T:
        if self.__done__():
            raise StopIteration()
        if not isinstance(T, None):
            self.__promise__()[0] = what
        return self.__next__()

    @__internal__
    def __exc_slot__(self) -> Ptr[cobj]:
        pass

    def throw(self: Generator[T], exc, /) -> T:
        """Raise `exc` at the suspended `yield` and run to the next yield.

        Returns the next yielded value; raises StopIteration if the generator
        finishes, or propagates the exception if not handled inside."""
        if isinstance(exc, type):
            return self.throw(exc())
        e: BaseException = exc
        if self.__done__():
            raise exc
        self.__exc_slot__()[0] = e.__raw__()
        self.__resume__()
        if self.__done__():
            raise StopIteration()
        return self._get()

    def close(self) -> None:
        """Raise GeneratorExit at the suspended `yield`; error if it yields again."""
        if self.__done__():
            return
        try:
            self.throw(GeneratorExit())
        except GeneratorExit:
            return
        except StopIteration:
            return
        raise RuntimeError("generator ignored GeneratorExit")

    @nocapture
    @llvm
    def _destroy(self) -> None:
        declare void @llvm.coro.destroy(ptr)
        call void @llvm.coro.destroy(ptr %self)
        ret {} {}

generator = Generator
