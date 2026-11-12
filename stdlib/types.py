# Subset of `types`. Runtime type objects (FunctionType, ModuleType, ...) do
# not exist in this statically compiled dialect.
EllipsisType = ellipsis
GeneratorType = Generator


def SimpleNamespace(**kwargs):
    # Returns a read-only named tuple: attribute access works (`ns.a`), but
    # attributes cannot be added or reassigned, and repr is `(a: 1, b: 'x')`.
    return kwargs
