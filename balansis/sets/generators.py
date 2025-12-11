from balansis.core.absolute import AbsoluteValue

def harmonic_generator(sign: int = 1):
    n = 1
    while True:
        yield AbsoluteValue(magnitude=1.0 / float(n), direction=int(sign))
        n += 1

def grandis_generator():
    d = 1
    while True:
        yield AbsoluteValue(magnitude=1.0, direction=d)
        d = -d
