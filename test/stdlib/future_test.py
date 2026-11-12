from __future__ import annotations
from __future__ import division, absolute_import, print_function, generators
from __future__ import nested_scopes, with_statement, unicode_literals, generator_stop
import __future__

@test
def test_future():
    assert 7 / 2 == 3.5
    assert __future__.annotations.compiler_flag == 0x1000000
    assert __future__.division.getMandatoryRelease()[0] == 3
    assert len(__future__.all_feature_names) == 10
    def f(x: int) -> int:
        return x + 1
    assert f(1) == 2
test_future()
