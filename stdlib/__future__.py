# Feature flags from CPython's __future__ module. In Typhon these are no-ops:
# every listed feature is already the default behavior.

class _Feature:
    optional: Tuple[int, int, int, str, int]
    mandatory: Tuple[int, int, int, str, int]
    compiler_flag: int

    def __init__(self, optional: Tuple[int, int, int, str, int],
                 mandatory: Tuple[int, int, int, str, int], compiler_flag: int):
        self.optional = optional
        self.mandatory = mandatory
        self.compiler_flag = compiler_flag

    def getOptionalRelease(self):
        return self.optional

    def getMandatoryRelease(self):
        return self.mandatory

    def __repr__(self):
        return f"_Feature({self.optional}, {self.mandatory}, {self.compiler_flag})"

all_feature_names = ["nested_scopes", "generators", "division", "absolute_import",
                     "with_statement", "print_function", "unicode_literals",
                     "barry_as_FLUFL", "generator_stop", "annotations"]

nested_scopes = _Feature((2, 1, 0, "beta", 1), (2, 2, 0, "alpha", 0), 0x0010)
generators = _Feature((2, 2, 0, "alpha", 1), (2, 3, 0, "final", 0), 0)
division = _Feature((2, 2, 0, "alpha", 2), (3, 0, 0, "alpha", 0), 0x20000)
absolute_import = _Feature((2, 5, 0, "alpha", 1), (3, 0, 0, "alpha", 0), 0x40000)
with_statement = _Feature((2, 5, 0, "alpha", 1), (2, 6, 0, "alpha", 0), 0x80000)
print_function = _Feature((2, 6, 0, "alpha", 2), (3, 0, 0, "alpha", 0), 0x100000)
unicode_literals = _Feature((2, 6, 0, "alpha", 2), (3, 0, 0, "alpha", 0), 0x200000)
barry_as_FLUFL = _Feature((3, 1, 0, "alpha", 2), (4, 0, 0, "alpha", 0), 0x400000)
generator_stop = _Feature((3, 5, 0, "beta", 1), (3, 7, 0, "alpha", 0), 0x800000)
annotations = _Feature((3, 7, 0, "beta", 1), (3, 7, 0, "alpha", 0), 0x1000000)
