def check(f):
    assert f([])==[]
    data=[(6,3),(2,1),(2,3),(10,10)]
    assert [tuple(x) for x in f(data)]==[(1,6),(10,10)]
    assert data==[(6,3),(2,1),(2,3),(10,10)]
    assert [tuple(x) for x in f([(1,1),(1,1)])]==[(1,1)]
