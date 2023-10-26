#!/usr/bin/env python3

#
# vec
# Copyright 2019-2023 by Larry Hastings
# See LICENSE for license information.
#

def preload_local_vec():
    """
    Pre-load the local "vec" module, to preclude finding
    an already-installed one on the path.
    """
    import pathlib
    import sys
    argv_0 = pathlib.Path(sys.argv[0])
    vec_dir = argv_0.resolve().parent
    while True:
        vec_init = vec_dir / "vec" / "__init__.py"
        if vec_init.is_file():
            break
        vec_dir = vec_dir.parent

    # this almost certainly *is* a git checkout
    # ... but that's not required, so don't assert it.
    # assert (vec_dir / ".git" / "config").is_file()

    if vec_dir not in sys.path:
        sys.path.insert(1, str(vec_dir))

    import vec
    assert vec.__file__.startswith(str(vec_dir))
    return vec_dir

vec_dir = preload_local_vec()


import inspect
from math import atan, pi, sqrt, tau
import os.path
import pathlib
import sys
import types
import unittest
import vec

from vec import Vector2, vector2_zero, vector2_1_0, vector2_0_1, vector2_1_1


class FalseSentinel:
    def __repr__(self): # pragma: nocover
        return "<FalseSentinel>"

    def __bool__(self): # pragma: nocover
        return False


places = 14

class VecTests(unittest.TestCase):

    def assertAlmostEqualVector(self, v1, v2):
        self.assertTrue(v1.almost_equal(v2, places=places))

    # if everything is broken, fix this first!
    def test_eq(self):
        # note: assertNotEqual uses !=, which uses __ne__
        # so if you want to test a falsity use assertFalse

        vc = Vector2(3, 7)
        self.assertEqual(vc, vc)
        self.assertEqual(vc, Vector2(3, 7))

        self.assertFalse(vc == 1)
        # note: __eq__ does NOT cast to Vector2 for you
        self.assertFalse(vc == (3, 7))

        vp = Vector2(r=3, theta=7)
        self.assertEqual(vp, vp)
        self.assertEqual(vp, Vector2(r=3, theta=7))
        self.assertFalse(vp == 1)
        self.assertFalse(vp == (3, 7))

        self.assertFalse(vc == vp)

        vc1 = Vector2(1, 0)
        vp1 = Vector2(r=1, theta=0)
        self.assertEqual(vc1, vp1)

        vc1 = Vector2(1, 0)
        vp1 = Vector2(r=1, theta=0)
        self.assertEqual(vp1, vc1)

    # ... maybe fix this second.
    def test_ne(self):
        # note: to test that != fails,
        # you have to assertFalse on the actual expression

        vc = Vector2(3, 7)
        self.assertFalse(vc != vc)
        self.assertFalse(vc != Vector2(3, 7))

        self.assertNotEqual(vc, 1)
        # note: __eq__ does NOT cast to Vector2 for you
        self.assertNotEqual(vc, (3, 7))

        vp = Vector2(r=3, theta=7)
        self.assertFalse(vp != vp)
        self.assertFalse(vp != Vector2(r=3, theta=7))
        self.assertNotEqual(vp, 1)
        self.assertNotEqual(vp, (3, 7))

        self.assertNotEqual(vc, vp)

        vc1 = Vector2(1, 0)
        vp1 = Vector2(r=1, theta=0)
        self.assertFalse(vc1 != vp1)

        vc1 = Vector2(1, 0)
        vp1 = Vector2(r=1, theta=0)
        self.assertFalse(vp1 != vc1)

    def test_white_box_testing(self):
        sentinel = FalseSentinel()

        def get_raw_slots(v):
            v2 = super(Vector2, v)
            get = v2.__getattribute__
            results = []
            for attr in "x y r theta r_squared _cartesian _polar".split():
                # print("   >> ", attr, end=" ")
                try:
                    value = get(attr)
                    # print(value)
                except AttributeError:
                    value = sentinel
                    # print(sentinel)
                results.append(value)
            # print(f"{v} -> {results}")
            return results

        def assertIsSentinel(o): self.assertIs(o, sentinel)

        def new_vectors():
            # (vc, vp)
            # vector-cartesian and vector-polar
            return (Vector2(1, 2), Vector2(r=1, theta=pi/3))

        vc, vp = new_vectors()
        self.assertIs(Vector2(vc), vc)
        self.assertIs(Vector2(vp), vp)

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vc)
        self.assertTrue(x and y)
        assertIsSentinel(r)
        assertIsSentinel(theta)
        assertIsSentinel(r_squared)
        self.assertEqual(_cartesian, 2)
        self.assertFalse(_polar, 0)

        # force vc to compute r_squared
        _ = vc.r_squared

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vc)
        self.assertTrue(x and y)
        assertIsSentinel(r)
        assertIsSentinel(theta)
        self.assertTrue(r_squared)
        self.assertEqual(_cartesian, 2)
        self.assertFalse(_polar, 0)

        # force vc to compute r
        _ = vc.r

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vc)
        self.assertTrue(x and y)
        self.assertTrue(r)
        assertIsSentinel(theta)
        self.assertTrue(r_squared)
        self.assertEqual(_cartesian, 2)
        self.assertEqual(_polar, 1)

        # force vc to compute theta
        _ = vc.theta

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vc)
        self.assertTrue(x and y)
        self.assertTrue(r)
        self.assertTrue(theta)
        self.assertTrue(r_squared)
        self.assertEqual(_cartesian, 2)
        self.assertEqual(_polar, 2)

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vp)
        assertIsSentinel(x)
        assertIsSentinel(y)
        self.assertEqual(r, 1)
        self.assertEqual(theta, pi/3)
        assertIsSentinel(r_squared)
        self.assertEqual(_cartesian, 0)
        self.assertEqual(_polar, 2)

        # force vp to compute r_squared
        _ = vp.r_squared

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vp)
        assertIsSentinel(x)
        assertIsSentinel(y)
        self.assertEqual(r, 1)
        self.assertEqual(theta, pi/3)
        self.assertTrue(r_squared)
        self.assertEqual(_cartesian, 0)
        self.assertEqual(_polar, 2)

        # force vp to compute y
        _ = vp.y

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vp)
        assertIsSentinel(x)
        self.assertTrue(y)
        self.assertEqual(r, 1)
        self.assertEqual(theta, pi/3)
        self.assertTrue(r_squared)
        self.assertEqual(_cartesian, 1)
        self.assertEqual(_polar, 2)

        # force vp to compute x
        _ = vp.x

        x, y, r, theta, r_squared, _cartesian, _polar = get_raw_slots(vp)
        self.assertTrue(x)
        self.assertTrue(y)
        self.assertEqual(r, 1)
        self.assertEqual(theta, pi/3)
        self.assertTrue(r_squared)
        self.assertEqual(_cartesian, 2)
        self.assertEqual(_polar, 2)

    def test_identity_tests(self):
        three = Vector2(3, 1)
        self.assertIs(Vector2(three), three)

        r1 = Vector2(r=1, theta=5)
        self.assertIs(Vector2(r1), r1)

    def test_zero_vector(self):

        self.assertIs(Vector2(), vector2_zero)
        self.assertIs(Vector2(0, 0), vector2_zero)
        self.assertIs(Vector2((0, 0)), vector2_zero)
        self.assertIs(Vector2([0, 0]), vector2_zero)
        self.assertIs(Vector2(types.SimpleNamespace(x=0, y=0)), vector2_zero)
        self.assertIs(Vector2(r=0, theta=None), vector2_zero)

        three = Vector2(3, 1)
        self.assertIs((three - three), vector2_zero)
        self.assertIs((three + -three), vector2_zero)
        self.assertIs(three * 0, vector2_zero)
        self.assertIs(Vector2(r=0, theta=None), vector2_zero)

        r1 = Vector2(r=1, theta=5)
        # (r1 + -r1) # doesn't work, we accumulate too much error
        self.assertIs((r1 - r1), vector2_zero)
        self.assertIs((r1 * 0), vector2_zero)

        self.assertIs(Vector2(vector2_zero), vector2_zero)

    def test_vector2_1_0(self):
        self.assertIs(Vector2(1, 0), vector2_1_0)
        self.assertIs(Vector2((1, 0)), vector2_1_0)
        self.assertIs(Vector2([1, 0]), vector2_1_0)
        self.assertIs(Vector2(types.SimpleNamespace(x=1, y=0)), vector2_1_0)
        self.assertIs(Vector2(r=1, theta=0), vector2_1_0)

        self.assertIs(Vector2(vector2_1_0), vector2_1_0)
        self.assertIs((vector2_1_0 * 3) - (vector2_1_0 * 2), vector2_1_0)

    def test_vector2_0_1(self):
        self.assertIs(Vector2(0, 1), vector2_0_1)
        self.assertIs(Vector2((0, 1)), vector2_0_1)
        self.assertIs(Vector2([0, 1]), vector2_0_1)
        self.assertIs(Vector2(types.SimpleNamespace(x=0, y=1)), vector2_0_1)
        self.assertIs(Vector2(r=1, theta=pi / 2), vector2_0_1)

        self.assertIs(Vector2(vector2_0_1), vector2_0_1)
        self.assertIs((vector2_0_1 * 3) - (vector2_0_1 * 2), vector2_0_1)

    def test_metaclass_call(self):
        # This test triggers every raise and every return
        # in Vector2Metaclass.__call__,
        # in order.

        def raises_value(**kwargs):
            with self.assertRaises(ValueError):
                Vector2(**kwargs)

        def raises_type(**kwargs):
            with self.assertRaises(TypeError):
                Vector2(**kwargs)

        def raises_key(**kwargs):
            with self.assertRaises(KeyError):
                Vector2(**kwargs)

        equal = self.assertEqual

        #
        # x is a Vector2
        #

        # nothing else should be set
        v1_1 = Vector2(1, 1)
        raises_value(x=v1_1, y=5)
        raises_value(x=v1_1, r=5)
        raises_value(x=v1_1, theta=5)
        raises_value(x=v1_1, r_squared=5)

        # passes
        equal(v1_1, Vector2(v1_1))

        #
        # x is an object with x and y attributes
        #

        # nothing else should be set
        namespace = types.SimpleNamespace(x=3, y=4)
        raises_value(x=namespace, y=5)
        raises_value(x=namespace, r=5)
        raises_value(x=namespace, theta=5)
        raises_value(x=namespace, r_squared=5)

        # passes
        equal(Vector2(namespace), Vector2(3, 4))

        #
        # x has a '__getitem__'
        d = {'x': 33, 'y': 44}
        equal(Vector2(d), Vector2(33, 44))
        raises_key(x={1: 'b', 2: 'd'})
        raises_value(x={'x': 33, 'y': 45, 'z': 22})
        raises_value(x=d, y=45)
        raises_value(x=d, r=22)
        raises_value(x=d, theta=pi)
        raises_value(x=d, r_squared=16)


        #
        # x has an '__iter__'
        #

        # must not be a set
        raises_type(x={1, 2})

        # nothing else should be set
        t4_5 = (4, 5)
        raises_value(x=t4_5, y=5)
        raises_value(x=t4_5, r=5)
        raises_value(x=t4_5, theta=5)
        raises_value(x=t4_5, r_squared=5)
        l6_7 = [6, 7]
        raises_value(x=l6_7, y=5)
        raises_value(x=l6_7, r=5)
        raises_value(x=l6_7, theta=5)
        raises_value(x=l6_7, r_squared=5)

        i = iter([6, 7])
        equal(Vector2(i), Vector2(6, 7))
        i = iter([6, 7])
        raises_value(x=i, y=5)
        i = iter([6, 7])
        raises_value(x=i, r=5)
        i = iter([6, 7])
        raises_value(x=i, theta=5)
        i = iter([6, 7])
        raises_value(x=i, r_squared=5)
        i = iter([6, ])
        raises_value(x=i)
        i = iter([6, 7, 8])
        raises_value(x=i)

        # must contain exactly two items
        raises_value(x=tuple())
        raises_value(x=[])
        raises_value(x=(1,))
        raises_value(x=[1,])
        raises_value(x=(1, 2, 3))
        raises_value(x=[1, 2, 3])

        equal(Vector2(t4_5), Vector2(4, 5))
        equal(Vector2(l6_7), Vector2(6, 7))

        # everything is unset
        self.assertIs(Vector2(), vector2_zero)

        # only one of x or y or r or theta
        raises_value(x=1)
        raises_value(y=1)
        raises_value(r=1)
        raises_value(theta=1)

        # at this point x must be int or float
        raises_type(x=3+1j, y=4)
        raises_type(x=types.SimpleNamespace(x=3+1j, y=4))
        raises_type(x=[3+1j, 4])
        raises_type(x=(3+1j, 4))

        # at this point y must be int or float
        raises_type(x=4, y=3+1j)
        raises_type(x=types.SimpleNamespace(x=4, y=3+1j))
        raises_type(x=[4, 3+1j])
        raises_type(x=(4, 3+1j))

        # at this point r must be int or float
        raises_type(r=3+1j, theta=4)

        # if theta is None, r must be 0
        raises_type(theta=None, r=3)
        # theta is not None, must be int or float
        raises_type(theta=3+1j, r=3)
        # theta is not None, r must != 0
        raises_type(theta=1, r=0)

        # normalize theta
        equal(Vector2(theta=6*tau, r=1), Vector2(1, 0))

        # normalize r
        equal(Vector2(theta=pi, r=-1), Vector2(1, 0))

        # can't specify r < 0 without theta
        raises_value(x=1, y=1, r=-3)

        # r_squared must be int or float
        raises_type(x=1, y=1, r_squared=2+1j)

        # x and y are both 0
        raises_value(x=0, y=0, r=3)
        raises_type(x=0, y=0, theta=1)
        raises_value(x=0, y=0, r_squared=1)
        self.assertIs(Vector2(x=0, y=0), vector2_zero)

        # x and y are not both 0
        raises_value(x=0, y=1, r=0)
        raises_value(x=1, y=0, r=0)
        raises_type(x=0, y=1, theta=None)
        raises_type(x=1, y=0, theta=None)
        raises_value(x=0, y=1, r_squared=0)
        raises_value(x=1, y=0, r_squared=0)

        # r and theta both set, r is 0
        raises_value(theta=None, r=0, x=5)
        raises_value(theta=None, r=0, y=5)
        raises_value(theta=None, r=0, r_squared=5)
        self.assertIs(Vector2(theta=None, r=0), vector2_zero)
        self.assertIs(Vector2.from_polar(0, None), vector2_zero)
        self.assertIs(Vector2(theta=None, r=0, x=0), vector2_zero)
        self.assertIs(Vector2(theta=None, r=0, y=0), vector2_zero)
        self.assertIs(Vector2(theta=None, r=0, r_squared=0), vector2_zero)

        # r and theta both set, r is not 0
        raises_value(theta=pi, r=1, r_squared=0)


    def test_from_polar(self):
        self.assertIs(Vector2.from_polar(0, None), vector2_zero)

        self.assertEqual(Vector2.from_polar(1, 0), Vector2(r=1, theta=0))
        self.assertEqual(Vector2.from_polar(1, pi/3), Vector2(r=1, theta=pi/3))
        self.assertEqual(Vector2.from_polar(1, pi/2), Vector2(r=1, theta=pi/2))
        self.assertEqual(Vector2.from_polar(1, pi), Vector2(r=1, theta=pi))

        self.assertEqual(Vector2.from_polar(2, 0), Vector2(r=2, theta=0))
        self.assertEqual(Vector2.from_polar(2, pi/3), Vector2(r=2, theta=pi/3))
        self.assertEqual(Vector2.from_polar(2, pi/2), Vector2(r=2, theta=pi/2))
        self.assertEqual(Vector2.from_polar(2, pi), Vector2(r=2, theta=pi))

        with self.assertRaises(TypeError):
            Vector2.from_polar(3)
        with self.assertRaises(TypeError):
            Vector2.from_polar(3, None)
        with self.assertRaises(TypeError):
            Vector2.from_polar(0, 0)
        with self.assertRaises(TypeError):
            Vector2.from_polar(1+2j, 5)
        with self.assertRaises(TypeError):
            Vector2.from_polar(1, 5+2j)

        v1 = Vector2(1, 3)
        self.assertIs(Vector2.from_polar(v1), v1)
        with self.assertRaises(ValueError):
            Vector2.from_polar(v1, theta=5)

        namespace = types.SimpleNamespace(r=1, theta=pi)
        self.assertEqual(Vector2.from_polar(namespace), Vector2(r=1, theta=pi))
        with self.assertRaises(ValueError):
            Vector2.from_polar(namespace, theta=2)

        d = {'r':1, 'theta':pi}
        self.assertEqual(Vector2.from_polar(d), Vector2(r=1, theta=pi))
        with self.assertRaises(ValueError):
            Vector2.from_polar(d, theta=2)
        with self.assertRaises(KeyError):
            Vector2.from_polar({1:2, 3:4})
        d['funk'] = 'town'
        with self.assertRaises(ValueError):
            Vector2.from_polar(d)

        iterable = [1, pi]
        self.assertEqual(Vector2.from_polar(iterable), Vector2(r=1, theta=pi))
        with self.assertRaises(ValueError):
            Vector2.from_polar(iterable, theta=2)
        with self.assertRaises(ValueError):
            Vector2.from_polar([1])
        with self.assertRaises(ValueError):
            Vector2.from_polar([1, pi, 5])

        iterable = (1, pi)
        self.assertEqual(Vector2.from_polar(iterable), Vector2(r=1, theta=pi))
        with self.assertRaises(ValueError):
            Vector2.from_polar(iterable, theta=2)
        with self.assertRaises(ValueError):
            Vector2.from_polar((1,))
        with self.assertRaises(ValueError):
            Vector2.from_polar((1, pi, 5))

        s = set((1, pi))
        with self.assertRaises(TypeError):
            Vector2.from_polar(s)

        i = iter((1, pi))
        self.assertEqual(Vector2.from_polar(i), Vector2(r=1, theta=pi))
        i = iter((1, pi))
        with self.assertRaises(ValueError):
            Vector2.from_polar(i, theta=2)
        i = iter((1, pi, 55))
        with self.assertRaises(ValueError):
            Vector2.from_polar(i)


        self.assertEqual(Vector2.from_polar({'r': 33, 'theta': 55}), Vector2(r=33, theta=55))



    def test_vector_math(self):
        def base_test(v, x, y, r_squared, theta):
            r = sqrt(r_squared)

            self.assertAlmostEqual(v.x, x, places=places)
            self.assertAlmostEqual(v.y, y, places=places)
            self.assertAlmostEqual(v.r, r, places=places)
            self.assertAlmostEqual(v.r_squared, r_squared, places=places)
            self.assertAlmostEqual(v.theta, theta, places=places)

        def test(x, y, r_squared, theta):
            v = Vector2(x, y)
            base_test(v, x, y, r_squared, theta)

            v2 = Vector2(r=sqrt(r_squared), theta=theta)
            base_test(v2, x, y, r_squared, theta)

        test(1, 0, 1, 0)
        test(0, 1, 1, pi/2)
        test(-1, 0, 1, -pi)
        test(0, -1, 1, -pi/2)

        test(1, 1,  2, pi/4)
        test(2, 2,  8, pi/4)
        test(3, 3, 18, pi/4)

        test(2, 3, 13, atan(3/2))

        v = Vector2(r=1, theta=0)
        base_test(v, 1, 0, 1, 0)
        v2 = Vector2(v.x, v.y).rotated(pi/2)
        base_test(v2, 0, 1, 1, pi/2)

        v = Vector2(r=1, theta=-2 * tau)
        base_test(v, 1, 0, 1, 0)


    def test_normalize_functions(self):
        self.assertEqual(vec.normalize_angle(pi), -pi)
        self.assertEqual(vec.normalize_angle(-pi), -pi)
        self.assertEqual(vec.normalize_angle(-2 * pi), 0)
        self.assertEqual(vec.normalize_angle(2 * pi), 0)

        self.assertEqual(vec.normalize_polar(3, pi/2), (3, pi/2))
        self.assertEqual(vec.normalize_polar(-3, pi/2), (3, -pi/2))

        self.assertEqual(vec.normalize_polar(0, None), (0, None))

        with self.assertRaises(TypeError):
            vec.normalize_polar(3, None)

        with self.assertRaises(TypeError):
            vec.normalize_polar(0, pi)

    def test_repr(self):
        def test(s, **kwargs):
            v = Vector2(**kwargs)
            self.assertEqual(s, repr(v))

        test("Vector2(1, 5)", x=(1, 5))
        test("Vector2(1, 5)", x=1, y=5)
        test("Vector2(1, 5, r=3)", x=1, y=5, r=3)
        test("Vector2(1, 5, theta=3)", x=1, y=5, theta=3)
        test("Vector2(1, 5, r_squared=3)", x=1, y=5, r_squared=3)
        test("Vector2(1, 5, r=1, theta=2, r_squared=3)", x=1, y=5, r=1, theta=2, r_squared=3)

        test("Vector2(r=1, theta=0.1)", r=1, theta=0.1)
        test("Vector2(x=3, r=1, theta=0.1)", x=3, r=1, theta=0.1)
        test("Vector2(y=3, r=1, theta=0.1)", y=3, r=1, theta=0.1)

        vector2_zero_repr = "Vector2(0, 0, r=0, theta=None, r_squared=0)"
        self.assertEqual(vector2_zero_repr, repr(vector2_zero))
        test(vector2_zero_repr, x=0, y=0)
        test(vector2_zero_repr, r=0, theta=None)

    def test_setattr(self):
        v = Vector2(1, 5)
        with self.assertRaises(TypeError):
            v.x = 3
        with self.assertRaises(TypeError):
            v.y = 3
        with self.assertRaises(TypeError):
            v.r = 2
        with self.assertRaises(TypeError):
            v.theta = 0.5
        with self.assertRaises(TypeError):
            v.r_squared = 2

    def test_getattr(self):
        # test the calculated values
        self.assertAlmostEqual(Vector2(r=sqrt(2), theta=pi/4).x, 1, places=places)
        self.assertAlmostEqual(Vector2(r=sqrt(2), theta=pi/4).y, 1, places=places)
        self.assertAlmostEqual(Vector2(x=1, y=1).r, sqrt(2), places=places)
        self.assertAlmostEqual(Vector2(x=1, y=1).theta, pi/4, places=places)
        self.assertAlmostEqual(Vector2(r=sqrt(2), theta=pi/4).r_squared, 2, places=places)
        self.assertAlmostEqual(Vector2(x=1, y=1).r_squared, 2, places=places)

        with self.assertRaises(AttributeError):
            Vector2(x=1, y=1).blort

    def test_add(self):
        def test_add(a1, a2, result):
            self.assertEqual(a1 + a2, result)
            self.assertEqual(a2 + a1, result)

        def test_add_is(a1, a2, result):
            self.assertIs(a1 + a2, result)
            self.assertIs(a2 + a1, result)

        v1_1 = Vector2(1, 1)
        test_add_is(v1_1, vector2_zero, v1_1)

        v3_4 = Vector2(3, 4)
        test_add(v1_1, Vector2(2, 3), v3_4)
        test_add(v1_1,        (2, 3), v3_4)
        test_add(v1_1,        [2, 3], v3_4)

        rp1 = Vector2(r=1, theta=1)

        test_add(rp1, rp1, rp1 * 2)
        test_add(rp1*2, -rp1, rp1)
        test_add_is(rp1, -rp1, vector2_zero)
        test_add_is(rp1, Vector2(r=-1, theta=1), vector2_zero)

    def test_sub(self):
        def test_sub(s1, s2, result):
            self.assertEqual(s1 - s2, result)
            self.assertEqual(s2 - s1, -result)

        def test_sub_is(s1, s2, result):
            self.assertIs(s1 - s2, result)

        v1_1 = Vector2(1, 1)
        test_sub_is(v1_1, vector2_zero, v1_1)

        v1_2 = Vector2(1, 2)
        test_sub(Vector2(2, 3), v1_1, v1_2)
        test_sub(       (2, 3), v1_1, v1_2)
        test_sub(       [2, 3], v1_1, v1_2)

        rp1 = Vector2(r=1, theta=1)
        rp1a = Vector2(r=1, theta=1)

        test_sub(rp1*2, rp1, rp1)
        test_sub(rp1, -rp1, rp1 * 2)
        test_sub_is(rp1, rp1, vector2_zero)
        test_sub_is(rp1, rp1a, vector2_zero)
        test_sub_is(rp1a, rp1, vector2_zero)

    def test_mul(self):
        def test_mul(m1, m2, result):
            self.assertEqual(m1 * m2, result)
            self.assertEqual(m1.scaled(m2), result)
            self.assertEqual(m2 * m1, result)

        def test_mul_is(m1, m2, result):
            self.assertIs(m1 * m2, result)
            self.assertIs(m1.scaled(m2), result)
            self.assertIs(m2 * m1, result)

        v3_5 = Vector2(3, 5)
        test_mul(   v3_5, -2, Vector2(-6, -10))
        test_mul(   v3_5, -1, -v3_5)
        test_mul_is(v3_5,  0, vector2_zero)
        test_mul_is(v3_5,  1, v3_5)
        test_mul(   v3_5,  2, Vector2(6, 10))
        test_mul(   v3_5,  3, Vector2(9, 15))


        test_mul(Vector2(1, 1, r=sqrt(2)), 2, Vector2(2, 2, r=2))
        test_mul(Vector2(1, 1, theta=pi/4), 2, Vector2(2, 2, theta=pi/4))
        test_mul(Vector2(1, 1, r_squared=2), 2, Vector2(2, 2, r_squared=4))

        v=Vector2(1, 1, theta=pi / 4)
        multiplicand = -2
        desired=Vector2(-2, -2, theta=(5 * pi) / 4)
        test_mul(v, multiplicand, desired)
        computed = v * multiplicand
        self.assertEqual(v.theta - pi, desired.theta)

        vp = Vector2(r=1.5, theta=1)
        test_mul(   vp, -1, -vp)
        test_mul(   vp, -2, Vector2(r=3, theta=1-pi))
        test_mul_is(vp,  0, vector2_zero)
        test_mul_is(vp,  1, vp)
        test_mul(   vp,  2, Vector2(r=3, theta=1))
        test_mul(   vp,  3, Vector2(r=4.5, theta=1))
        test_mul(   vp,  3.5, Vector2(r=5.25, theta=1))

        with self.assertRaises(ValueError):
            v3_5 * 2+1j
        with self.assertRaises(ValueError):
            v3_5 * [1, 2, 3]
        with self.assertRaises(ValueError):
            v3_5 * v3_5
        with self.assertRaises(ValueError):
            v3_5 * vp

        with self.assertRaises(ValueError):
            vp * 2+1j
        with self.assertRaises(ValueError):
            vp * [1, 2, 3]
        with self.assertRaises(ValueError):
            vp * v3_5
        with self.assertRaises(ValueError):
            vp * vp


    def test_truediv(self):
        def test_truediv(dividend, divisor, result):
            self.assertEqual(dividend / divisor, result)
            if divisor:
                self.assertEqual(dividend.scaled(1 / divisor), result)

        v3_5 = Vector2(3, 5)
        test_truediv(v3_5,-1, -v3_5)
        test_truediv(v3_5, 0, vector2_zero)
        test_truediv(v3_5, 1, v3_5)
        test_truediv(v3_5, 2, Vector2(1.5, 2.5))

        vp = Vector2(r=1.5, theta=1)
        test_truediv(vp,-1, -vp)
        test_truediv(vp, 0, vector2_zero)
        test_truediv(vp, 1, vp)
        test_truediv(vp, 2, Vector2(r=0.75, theta=1))
        test_truediv(vp, -2, Vector2(r=0.75, theta=1-pi))

        v=Vector2(2, 2, theta=1)
        divisor = -2
        desired=Vector2(-1, -1, theta=1-pi)
        test_truediv(v, divisor, desired)
        computed = v / divisor
        self.assertEqual(v.theta - pi, desired.theta)

        with self.assertRaises(ValueError):
            v3_5 / 2+1j
        with self.assertRaises(ValueError):
            v3_5 / [1, 2, 3]
        with self.assertRaises(ValueError):
            v3_5 / v3_5
        with self.assertRaises(ValueError):
            v3_5 / vp

        with self.assertRaises(ValueError):
            vp / 2+1j
        with self.assertRaises(ValueError):
            vp / [1, 2, 3]
        with self.assertRaises(ValueError):
            vp / v3_5
        with self.assertRaises(ValueError):
            vp / vp


    def test_pos(self):
        v = Vector2(1, 5)
        self.assertEqual(v, +v)
        v = Vector2(r=3.3, theta=1.2)
        self.assertEqual(v, +v)

    def test_neg(self):
        v = Vector2(1, 5)
        self.assertEqual(-v, Vector2(-1, -5))
        v = Vector2(r=3.3, theta=1)
        self.assertEqual(-v, Vector2(r=3.3, theta=1-pi))
        self.assertEqual(-vector2_zero, vector2_zero)


    def test_hash(self):
        v1_0 = Vector2(1, 0)
        v1_1 = Vector2(1, 1)
        self.assertNotEqual(hash(v1_0), hash(v1_1))
        self.assertEqual(hash(v1_0), hash(Vector2(1, 0)))
        self.assertEqual(hash(v1_1), hash(Vector2(1, 1)))

        self.assertEqual(hash(v1_0), hash(Vector2(r=1, theta=0)))


    def test_bool(self):
        self.assertFalse(vector2_zero)
        self.assertFalse(Vector2(0, 0))
        self.assertFalse(Vector2(r=0, theta=None))

        self.assertTrue(Vector2(1, 0))
        self.assertTrue(Vector2(0, 1))
        self.assertTrue(Vector2(r=1, theta=0))
        self.assertTrue(Vector2(r=-1, theta=1))


    def test_len(self):
        self.assertEqual(len(Vector2(1, 5)), 2)
        self.assertEqual(len(Vector2(r=1, theta=1)), 2)


    def test_iter(self):
        self.assertEqual(list(Vector2(1, 5)), [1, 5])
        self.assertEqual(tuple(Vector2(1, 5)), (1, 5))
        self.assertEqual(tuple(Vector2(r=0, theta=None)), (0, 0))


    def test_getitem(self):
        with self.assertRaises(IndexError):
            Vector2(1, 5)[-1]
        self.assertEqual(Vector2(1, 5)[0], 1)
        self.assertEqual(Vector2(1, 5)[1], 5)
        with self.assertRaises(IndexError):
            Vector2(1, 5)[2]

        with self.assertRaises(IndexError):
            Vector2(r=1, theta=0)[-1]
        self.assertEqual(Vector2(r=1, theta=0)[0], 1)
        self.assertEqual(Vector2(r=1, theta=0)[1], 0)
        with self.assertRaises(IndexError):
            Vector2(r=1, theta=0)[2]


    def test_setitem(self):
        with self.assertRaises(TypeError):
            Vector2(1, 5)[2] = 33
        with self.assertRaises(TypeError):
            Vector2(r=1, theta=5)[2] = 33

    def test_polar(self):
        self.assertEqual(vector2_zero.polar(), (0, None))
        self.assertEqual(Vector2(1, 0).polar(), (1, 0))
        self.assertEqual(Vector2(-1, 0).polar(), (1, -pi))

    def test_almost_equal(self):
        v1 = Vector2(0.99991111, 0)
        v2 = Vector2(0.99992222, 0)
        self.assertTrue(v1.almost_equal(v2, 1))
        self.assertTrue(v1.almost_equal(v2, 2))
        self.assertTrue(v1.almost_equal(v2, 3))
        self.assertTrue(v1.almost_equal(v2, 4))
        self.assertFalse(v1.almost_equal(v2, 5))
        self.assertFalse(v1.almost_equal(v2, 6))

        v1 = Vector2(0, 0.99991111)
        v2 = Vector2(0, 0.99992222)
        self.assertTrue(v1.almost_equal(v2, 1))
        self.assertTrue(v1.almost_equal(v2, 2))
        self.assertTrue(v1.almost_equal(v2, 3))
        self.assertTrue(v1.almost_equal(v2, 4))
        self.assertFalse(v1.almost_equal(v2, 5))
        self.assertFalse(v1.almost_equal(v2, 6))

        v1 = Vector2(0.99991111, 0.99992222)
        v2 = [0.99992222, 0.99991111]
        self.assertTrue(v1.almost_equal(v2, 1))
        self.assertTrue(v1.almost_equal(v2, 2))
        self.assertTrue(v1.almost_equal(v2, 3))
        self.assertTrue(v1.almost_equal(v2, 4))
        self.assertFalse(v1.almost_equal(v2, 5))
        self.assertFalse(v1.almost_equal(v2, 6))


    def test_scaled_to_length(self):
        vp_4_4 = Vector2(r=4, theta=4)
        self.assertEqual(vp_4_4.scaled_to_length(4), vp_4_4)
        self.assertEqual(vp_4_4.scaled_to_length(2), Vector2(r=2, theta=4))
        self.assertIs(vp_4_4.scaled_to_length(0), vector2_zero)

        v = Vector2(1, 0)
        self.assertIs(v.scaled_to_length(0), vector2_zero)
        self.assertIs(v.scaled_to_length(1), v)
        self.assertEqual(v.scaled_to_length(-1), Vector2(-1, 0))
        self.assertEqual(v.scaled_to_length(4), Vector2(4, 0))
        self.assertEqual(v.scaled_to_length(-4), Vector2(-4, 0))

        v = Vector2(0, 1)
        self.assertIs(v.scaled_to_length(0), vector2_zero)
        self.assertIs(v.scaled_to_length(1), v)
        self.assertEqual(v.scaled_to_length(-1), Vector2(0, -1))
        self.assertEqual(v.scaled_to_length(4), Vector2(0, 4))
        self.assertEqual(v.scaled_to_length(-4), Vector2(0, -4))

        # round-trip through polar!
        v = Vector2(4, 4).scaled_to_length(2 * sqrt(2))
        v2 = Vector2(2, 2)
        self.assertAlmostEqualVector(v, v2)
        self.assertIs(v.scaled_to_length(0), vector2_zero)

        with self.assertRaises(ValueError):
            vector2_zero.scaled_to_length(1)
        with self.assertRaises(ValueError):
            vector2_zero.scaled_to_length(4)
        # this one is allowed!
        self.assertIs(vector2_zero.scaled_to_length(0), vector2_zero)


    def test_normalized(self):
        v = Vector2(r=1, theta=3)
        self.assertIs(v.normalized(), v)
        v2 = Vector2(r=44, theta=3)
        self.assertEqual(v2.normalized(), v)

        v = Vector2(1, 0)
        self.assertIs(v.normalized(), v)
        v2 = Vector2(2, 0)
        self.assertEqual(v2.normalized(), v)

        v = Vector2(0, 1)
        self.assertIs(v.normalized(), v)
        v2 = Vector2(0, 2)
        self.assertEqual(v2.normalized(), v)

        v = Vector2(1, 1)
        v2 = v.normalized()
        v2n = Vector2(sqrt(2)/2, sqrt(2)/2)
        self.assertAlmostEqualVector(v2, v2n)

        with self.assertRaises(ValueError):
            vector2_zero.normalized()

    def test_rotated(self):
        # ninety degrees stuff
        v = Vector2(1, 0)
        self.assertIs(v.rotated(0), v)
        self.assertEqual(v.rotated( vec.pi_over_two), Vector2( 0,  1))
        self.assertEqual(v.rotated( pi),              Vector2(-1,  0))
        self.assertEqual(v.rotated(-vec.pi_over_two), Vector2( 0, -1))

        v = Vector2(3, 2)
        self.assertIs(v.rotated(0), v)
        self.assertEqual(v.rotated( vec.pi_over_two), Vector2(-2,  3))
        self.assertEqual(v.rotated(pi),               Vector2(-3, -2))
        self.assertEqual(v.rotated(-vec.pi_over_two), Vector2( 2, -3))

        def test_rotated_polar(v, theta):
            v2 = v.rotated(theta)
            self.assertEqual(v.r, v2.r)
            self.assertAlmostEqual(vec.normalize_angle(v.theta + theta), v2.theta, places=places)

        v = Vector2(r=3, theta=0.1)
        self.assertIs(v.rotated(0), v)
        test_rotated_polar(v, vec.pi_over_two)
        test_rotated_polar(v, pi)
        test_rotated_polar(v, vec.negative_pi_over_two)
        test_rotated_polar(v, -1)
        test_rotated_polar(v, 1)
        test_rotated_polar(v, 2)
        test_rotated_polar(v, 3)
        test_rotated_polar(v, 4)
        test_rotated_polar(v, 5)

        v = Vector2(1, 0)
        v2 = v.rotated(pi/4)
        v2r = Vector2(sqrt(2)/2, sqrt(2)/2)
        self.assertAlmostEqualVector(v2, v2r)

        self.assertIs(vector2_zero.rotated(0), vector2_zero)
        with self.assertRaises(ValueError):
            vector2_zero.rotated(3)
        with self.assertRaises(ValueError):
            vector2_zero.rotated(-3)

    def test_dot(self):
        v1 = Vector2(1, 2)
        v2 = Vector2(3, 5)

        self.assertEqual(v1.dot(vector2_zero), 0)
        self.assertEqual(v2.dot(vector2_zero), 0)
        self.assertEqual(vector2_zero.dot(v1), 0)
        self.assertEqual(vector2_zero.dot(v2), 0)

        self.assertEqual(v1.dot(v2), (1*3) + (2*5))
        self.assertEqual(v1.dot([v2.x, v2.y]), (1*3) + (2*5))
        self.assertEqual(v2.dot(v2), (3*3) + (5*5))
        self.assertEqual(v2.dot([v2.x, v2.y]), (3*3) + (5*5))

    def test_cross(self):
        v1 = Vector2(1, 2)
        v2 = Vector2(3, 5)

        self.assertEqual(v1.cross(vector2_zero), 0)
        self.assertEqual(v2.cross(vector2_zero), 0)

        self.assertEqual(v1.cross(v1), 0)
        self.assertEqual(v2.cross(v2), 0)

        self.assertEqual(v1.cross(v2), (1*5) - (2*3))
        self.assertEqual(v1.cross([v2.x, v2.y]), (1*5) - (2*3))

    def test_lerp(self):
        v1 = Vector2(1, 2)
        v2 = Vector2(3, 6)

        self.assertIs(v1.lerp(v1, 0), v1)
        self.assertIs(v1.lerp(v1, 1.5e13), v1)
        self.assertIs(v1.lerp(v1, -80_005), v1)

        self.assertIs(v1.slerp(v2, 1), v2)

        def test_lerp(v1, v2, halfway):
            self.assertIs(v1.lerp(v2, 0), v1)
            self.assertEqual(v1.lerp(v2, 0.5), halfway)
            self.assertEqual(v1.lerp(v2, 1), Vector2(v2))

        halfway = Vector2((v1.x + v2.x) / 2, (v1.y + v2.y) / 2)
        test_lerp(v1, v2, halfway)
        test_lerp(v1, [v2.x, v2.y], halfway)

        with self.assertRaises(ValueError):
            v1.lerp(v2, 2+12j)

    def test_slerp(self):
        v1 = Vector2(0, 2)
        halfway = Vector2(5.224158034564077, 8.881068658758931)
        v2 = Vector2(10, 15)

        self.assertIs(v1.slerp(v1, 0), v1)
        self.assertIs(v1.slerp(v1, 1.5e13), v1)
        self.assertIs(v1.slerp(v1, -80_005), v1)

        self.assertIs(v1.slerp(v2, 1), v2)

        def test_slerp(v1, v2, halfway):
            self.assertIs(v1.slerp(v2, 0), v1)
            self.assertAlmostEqualVector(v1.slerp(v2, 0.5), halfway)
            self.assertEqual(v1.slerp(v2, 1), Vector2(v2))

        test_slerp(v1, v2, halfway)
        test_slerp(v1, [10, 15], halfway)

        # these vectors force slerp to fallback to lerp
        v1 = Vector2(1, 2)
        v2 = Vector2(3, 6)
        test_slerp(v1, v2, v1.lerp(v2, 0.5))

        with self.assertRaises(ValueError):
            v1.slerp(v2, 2+12j)


    def test_nlerp(self):
        v1 = Vector2(0, 2)
        halfway = Vector2(5.077237786604788, 8.631304237228141)
        v2 = Vector2(10, 15)

        self.assertIs(v1.nlerp(v1, 0), v1)
        self.assertIs(v1.nlerp(v1, 1.5e13), v1)
        self.assertIs(v1.nlerp(v1, -80_005), v1)

        self.assertIs(v1.nlerp(v2, 1), v2)

        def test_nlerp(v1, v2, halfway):
            self.assertIs(v1.nlerp(v2, 0), v1)
            self.assertAlmostEqualVector(v1.nlerp(v2, 0.5), halfway)
            self.assertEqual(v1.nlerp(v2, 1), Vector2(v2))

        test_nlerp(v1, v2, halfway)
        test_nlerp(v1, [10, 15], halfway)

        with self.assertRaises(ValueError):
            v1.nlerp(v2, 2+12j)


    def test_permit_coordinate_type(self):
        original = vec.permitted_coordinate_types

        with self.assertRaises(TypeError):
            Vector2(2+1j, 3+2j)

        vec.permit_coordinate_type(complex)

        v1_1 = Vector2(1, 1)
        v = Vector2(2+1j, 3+2j)
        v2 = Vector2(3+1j, 4+2j)

        self.assertEqual(v + v1_1, v2)

        vec.permitted_coordinate_types = original

        with self.assertRaises(TypeError):
            Vector2(2+1j, 3+2j)


unittest.main()
