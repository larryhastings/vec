import os.path
from math import pi
import sys
import types

vec_base = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.insert(0, vec_base)
from vec import Vector2, vector2_zero

# white box testing
v1 = Vector2(1, 2)
x, y, r, theta, r_squared, _polar = v1._get_raw_slots()
assert x and y and (not _polar)
assert r == theta == r_squared == None
v1.r_squared
x, y, r, theta, r_squared, _polar = v1._get_raw_slots()
assert x and y and r_squared and (not _polar), f"x {x} y {y} r {r} r_squared {r_squared} _polar {_polar}"
assert r == theta == None

v1 = Vector2(1, 2)
x, y, r, theta, r_squared, _polar = v1._get_raw_slots()
assert x and y and (not _polar)
assert r == theta == r_squared == None
v1.r
x, y, r, theta, r_squared, _polar = v1._get_raw_slots()
assert x and y and r and r_squared and (not _polar), f"x {x} y {y} r {r} r_squared {r_squared} _polar {_polar}"
assert theta == None

vr = Vector2(r=1, theta=pi)
assert Vector2(vr) is vr

x, y, r, theta, r_squared, _polar = vr._get_raw_slots()
assert r and theta and _polar
assert x == y == r_squared == None

three = Vector2(3, 1)
assert Vector2(three) is three

r1 = Vector2(r=1, theta=5)
assert Vector2(v1) is v1

for s in """
    Vector2()
    Vector2(None)
    Vector2(None, None)
    Vector2(0, 0)
    Vector2((0, 0))
    Vector2([0, 0])
    Vector2(types.SimpleNamespace(x=0, y=0))
    Vector2(r=0, theta=0)
    Vector2(r=0)
    (three - three)
    (three + -three)
    three * 0
    Vector2(r=0, theta=5)
    (r1 - r1)
    # (r1 + -r1) # doesn't work, we accumulate too much error
    (r1 * 0)
    Vector2(vector2_zero)
    """.strip().split("\n"):
    s = s.strip()
    if s.startswith("#"):
        continue
    value = eval(s)
    assert value is vector2_zero, f"{s} should evaluate to vector2_zero but instead gives us {value}!"

for d in (
    {"x": 3},
    {"y": 3},
    {"r": 3},
    {"theta": 3},

    {"x":3, "r": 3},
    {"x":3, "theta": 3},
    {"y":3, "r": 3},
    {"y":3, "theta": 3},

    {"x": 0},
    {"y": 0},
    # Vector2(r=0) is explicitly allowed
    {"theta": 0},
    ):
    try:
        Vector2(**d)
        sys.exit(f"should have thrown a value error for Vector2(**{d}) but did not!")
    except ValueError:
        pass

def test(text, v):
    print(text + ":")
    print("  v", v)
    print("  v is vector2_zero", v is vector2_zero)
    print("  v.x", v.x, "v.y", v.y)
    print("  v.r", v.r, "v.theta", v.theta)
    print("  v.r_squared", v.r_squared)
    print("  v[0] v[1]", v[0], v[1])
    print("  [*v]", [*v])
    print()

test("Vector2()", Vector2())
test("(1, 0)", Vector2(1, 0))
test(".from_polar(r=1, theta=pi/4)", Vector2.from_polar(r=1, theta=pi/4))
test("Vector2(1, 0).rotated(pi/2)", Vector2(1, 0).rotated(pi/2))
test("Vector2(r=1, theta=0).rotated(pi/2)", Vector2(r=1, theta=0).rotated(pi/2))
