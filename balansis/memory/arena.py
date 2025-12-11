class AbsoluteArena:
    def __init__(self):
        self._cache = {}

    def alloc(self, magnitude: float, direction: int):
        key = (float(magnitude), int(direction))
        val = self._cache.get(key)
        if val is None:
            from balansis.core.absolute import AbsoluteValue
            val = AbsoluteValue(magnitude=key[0], direction=key[1])
            self._cache[key] = val
        return val

    def size(self) -> int:
        return len(self._cache)
