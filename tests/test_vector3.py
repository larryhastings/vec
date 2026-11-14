#!/usr/bin/env python3
"""Vector3 behavior and numerical regressions, runnable on Python 3.6+."""
from decimal import Decimal
from itertools import permutations
from math import cos, pi, sin, sqrt, tau
from pathlib import Path
import operator
import random
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vec
from vec import (Vector2, Vector3, vector3_zero, vector3_1_0_0,
                 vector3_0_1_0, vector3_0_0_1)


class Vector3Tests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=12):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)

    def test_construction(self):
        expected = Vector3(2, 3, 4)
        for value in ((2, 3, 4), [2, 3, 4], iter((2, 3, 4)),
                      {'z': 4, 'y': 3, 'x': 2},
                      SimpleNamespace(x=2, y=3, z=4)):
            self.assertEqual(Vector3(value), expected)
        self.assertEqual(Vector3(x=2, y=3, z=4), expected)
        self.assertIs(Vector3(expected), expected)
        self.assertIs(Vector3(), vector3_zero)
        self.assertIs(Vector3(0, -0.0, 0), vector3_zero)
        self.assertIs(Vector3([0, 0, 0]), vector3_zero)
        self.assertIs(vec.vector3_0_0_0, vector3_zero)
        for coords, v in (((1, 0, 0), vector3_1_0_0),
                          ((0, 1, 0), vector3_0_1_0),
                          ((0, 0, 1), vector3_0_0_1)):
            self.assertIs(Vector3(coords), v)

    def test_invalid_construction(self):
        for args in ((1,), (1, 2), ((),), ((1, 2),), ((1, 2, 3, 4),),
                     ({'x': 1, 'y': 2},), ({'x': 1, 'y': 2, 'z': 3, 'w': 4},)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    Vector3(*args)
        with self.assertRaises(ValueError):
            Vector3(y=2, z=3)
        with self.assertRaises(KeyError):
            Vector3({'x': 1, 'y': 2, 'w': 3})
        for args in ((None,), ({1, 2, 3},), ('123',), (b'123',),
                     (bytearray(b'123'),), (None, 2, 3), (1, None, 3),
                     (1, 2, None), (1, 2, '3'), (Vector3(2, 3, 4), 2, 3)):
            with self.subTest(args=args):
                with self.assertRaises(TypeError):
                    Vector3(*args)
        with self.assertRaises(ValueError):
            Vector3(Vector2(2, 3))

    def test_permitted_coordinates(self):
        original = vec.permitted_coordinate_types
        try:
            vec.permit_coordinate_type(Decimal)
            a = Vector3(Decimal('2.5'), Decimal('3.5'), Decimal('4.5'))
            self.assertEqual(tuple(a * 2), (Decimal(5), Decimal(7), Decimal(9)))
        finally:
            vec.permitted_coordinate_types = original

    def test_slots_are_lazy_and_cached(self):
        for order in permutations(('r', 'r_squared', '_hash')):
            v = Vector3(2., 3., 6.)
            for field in ('r', 'r_squared', '_hash'):
                with self.assertRaises(AttributeError):
                    object.__getattribute__(v, field)
            for field in order:
                first = getattr(v, field)
                self.assertIs(getattr(v, field), first)
            self.assertEqual(v.r, 7)
            self.assertEqual(v.r_squared, 49)
            self.assertEqual(hash(v), hash((2., 3., 6.)))
        with self.assertRaises(AttributeError):
            v.missing
        with self.assertRaises(AttributeError):
            v.__dict__

    def test_immutable(self):
        for v in (Vector3(2, 3, 4), vector3_zero, vector3_1_0_0):
            for field in Vector3.__slots__ + ('new_attribute',):
                with self.subTest(v=v, field=field):
                    with self.assertRaises(TypeError):
                        setattr(v, field, 99)
                    with self.assertRaises(TypeError):
                        delattr(v, field)
            with self.assertRaises(TypeError):
                v[0] = 99

    def test_predefined_cache_unchanged(self):
        constants = (vector3_zero, vector3_1_0_0, vector3_0_1_0, vector3_0_0_1)
        before = [tuple(object.__getattribute__(v, f) for f in Vector3.__slots__)
                  for v in constants]
        for axis in constants[1:]:
            self.assertIs((axis * 5).normalized(), axis)
            self.assertIs((axis * -5).scaled_to_length(-1.0), axis)
        for v, original in zip(constants, before):
            for field, value in zip(Vector3.__slots__, original):
                self.assertIs(object.__getattribute__(v, field), value)

    def test_repr_and_protocols(self):
        v = Vector3(2, 3.5, -4)
        self.assertEqual(repr(v), 'Vector3(2, 3.5, -4)')
        self.assertEqual(len(v), 3)
        self.assertEqual(list(v), [2, 3.5, -4])
        self.assertEqual([v[i] for i in range(3)], list(v))
        self.assertFalse(vector3_zero)
        self.assertTrue(v)
        for i in (-1, 3, 100):
            with self.assertRaises(IndexError):
                v[i]

    def test_equality_hashing(self):
        a, b = Vector3(2, 3, 6), Vector3(2., 3., 6.)
        values = {a: 'present'}
        self.assertEqual(a, b)
        self.assertFalse(a != b)
        for v in (a, b):
            v.r
            v.r_squared
            hash(v)
        self.assertEqual(values[b], 'present')
        self.assertEqual(len({a, b}), 1)
        for other in ((2, 3, 6), Vector2(2, 3), Vector3(2, 3, 7), None):
            self.assertFalse(a == other)
            self.assertTrue(a != other)
        self.assertEqual(a, a)
        with self.assertRaises(TypeError):
            a < b

    def test_arithmetic(self):
        a, b = Vector3(2, 3, 4), Vector3(5, 6, 7)
        self.assertEqual(a + b, Vector3(7, 9, 11))
        self.assertEqual(a + (5, 6, 7), Vector3(7, 9, 11))
        self.assertEqual((5, 6, 7) + a, Vector3(7, 9, 11))
        self.assertEqual(a - b, Vector3(-3, -3, -3))
        self.assertEqual(a - (5, 6, 7), Vector3(-3, -3, -3))
        self.assertEqual((5, 6, 7) - a, Vector3(3, 3, 3))
        self.assertIs(a + vector3_zero, a)
        self.assertIs(vector3_zero + a, a)
        self.assertIs(a - vector3_zero, a)
        self.assertIs(a - a, vector3_zero)
        self.assertIs(+a, a)
        self.assertEqual(-a, Vector3(-2, -3, -4))
        self.assertIs(-vector3_zero, vector3_zero)
        self.assertEqual(a * -2, Vector3(-4, -6, -8))
        self.assertEqual(2 * a, Vector3(4, 6, 8))
        self.assertIs(a * 1, a)
        self.assertIs(a * 0, vector3_zero)
        self.assertEqual(a / 2, Vector3(1, 1.5, 2))
        self.assertEqual(a / -2, Vector3(-1, -1.5, -2))
        self.assertIs(a / 1, a)
        self.assertEqual(a.scaled(2), a * 2)
        for invalid in (None, '2', a, complex(2, 1)):
            with self.assertRaises(ValueError):
                a * invalid
            with self.assertRaises(ValueError):
                a / invalid
        for dividend in (a, vector3_zero):
            for divisor in (0, 0., -0.):
                with self.assertRaises(ZeroDivisionError):
                    dividend / divisor

    def test_almost_equal(self):
        a = Vector3(2, 3, 4)
        self.assertTrue(a.almost_equal(a, 8))
        self.assertTrue(a.almost_equal((2, 3, 4.0001), 3))
        self.assertFalse(a.almost_equal((2, 3, 4.0001), 5))
        self.assertFalse(a.almost_equal((2.1, 3, 4), 3))
        self.assertFalse(a.almost_equal((2, 3.1, 4), 3))

    def test_lengths_and_normalization(self):
        a = Vector3(2, 3, 6)
        self.assertEqual(a.r_squared, 49)
        self.assertEqual(a.r, 7)
        self.assertIs(a.scaled_to_length(7), a)
        self.assertIs(a.scaled_to_length(0), vector3_zero)
        self.assertIs(vector3_zero.scaled_to_length(0), vector3_zero)
        self.assertVectorClose(a.normalized(), (2/7, 3/7, 6/7))
        self.assertEqual(a.normalized().r, 1)
        self.assertVectorClose(a.scaled_to_length(-14), (-4, -6, -12))
        for v in (vector3_1_0_0, vector3_0_1_0, vector3_0_0_1):
            self.assertIs(v.normalized(), v)
            self.assertIs((-v).scaled_to_length(-1), v)
        with self.assertRaises(ValueError):
            vector3_zero.normalized()
        with self.assertRaises(ValueError):
            vector3_zero.scaled_to_length(5)
        with self.assertRaises(TypeError):
            a.scaled_to_length('5')

    def test_dot_and_cross(self):
        x, y, z = vector3_1_0_0, vector3_0_1_0, vector3_0_0_1
        self.assertIs(x.cross(y), z)
        self.assertIs(y.cross(z), x)
        self.assertIs(z.cross(x), y)
        self.assertEqual(y.cross(x), -z)
        self.assertEqual(x.cross((0, 1, 0)), z)
        self.assertEqual(x.dot((0, 1, 0)), 0)
        a, b = Vector3(2, 3, 4), Vector3(5, -6, 7)
        self.assertEqual(a.dot(b), 20)
        self.assertEqual(a.cross(b), Vector3(45, 6, -27))
        self.assertIs(a.cross(a), vector3_zero)
        self.assertIs(a.cross(vector3_zero), vector3_zero)
        self.assertEqual(a.cross(b).dot(a), 0)
        self.assertEqual(a.cross(b).dot(b), 0)

    def test_rotated_principal_axes(self):
        x, y, z = vector3_1_0_0, vector3_0_1_0, vector3_0_0_1
        self.assertIs(x.rotated(z, pi/2), y)
        self.assertIs(y.rotated(x, pi/2), z)
        self.assertIs(z.rotated(y, pi/2), x)
        self.assertEqual(x.rotated(z, -pi/2), -y)
        self.assertEqual(x.rotated(z, pi), -x)
        self.assertIs(x.rotated(z, tau), x)
        self.assertIs(x.rotated(z, 0), x)
        self.assertEqual(x.rotated((0, 0, -7), pi/2), -y)
        self.assertEqual(Vector3(2, 3, 4).rotated(z, pi/2), Vector3(-3, 2, 4))
        self.assertIs(vector3_zero.rotated(z, 0), vector3_zero)
        with self.assertRaises(ValueError):
            vector3_zero.rotated(z, 1)
        with self.assertRaises(ValueError):
            x.rotated(vector3_zero, 0)
        with self.assertRaises(TypeError):
            x.rotated(z, '1')

    def test_rotated_arbitrary_axis(self):
        v, axis = Vector3(2, 3, 6), Vector3(1, -2, 3)
        result = v.rotated(axis, 0.7)
        self.assertAlmostEqual(result.r, v.r, places=12)
        self.assertAlmostEqual(result.dot(axis), v.dot(axis), places=12)
        self.assertVectorClose(result.rotated(axis, -0.7), v)
        self.assertVectorClose(v.rotated(-axis, 0.7), v.rotated(axis, -0.7))
        # A third-turn around the diagonal cyclically permutes coordinates.
        self.assertVectorClose(Vector3(1, 0, 0).rotated((1, 1, 1), tau/3), (0, 1, 0))

    def test_interpolation_endpoints_and_types(self):
        a, b = Vector3(2, 3, 6), Vector3(7, 8, 9)
        for name in ('lerp', 'nlerp', 'slerp'):
            for start, end in ((a, b), (a, -a), (a, vector3_zero),
                               (vector3_zero, a), (vector3_zero, vector3_zero)):
                method = getattr(start, name)
                self.assertIs(method(end, 0), start)
                self.assertIs(method(end, 1), end)
            self.assertIs(getattr(a, name)(a, 0.4), a)
            with self.assertRaises(TypeError):
                getattr(a, name)(b, '0.5')
            self.assertVectorClose(getattr(a, name)(tuple(b), .5),
                                   getattr(a, name)(b, .5))

    def test_lerp_and_nlerp(self):
        a, b = Vector3(2, 0, 0), Vector3(0, 4, 0)
        self.assertEqual(a.lerp(b, .5), Vector3(1, 2, 0))
        self.assertEqual(a.lerp(b, -.5), Vector3(3, -2, 0))
        self.assertEqual(a.lerp(b, 1.5), Vector3(-1, 6, 0))
        n = a.nlerp(b, .5)
        self.assertAlmostEqual(n.r, 3)
        self.assertVectorClose(n, (3/sqrt(5), 6/sqrt(5), 0))
        self.assertVectorClose(a.nlerp(vector3_zero, .5), (1, 0, 0))
        self.assertVectorClose(vector3_zero.nlerp(a, .5), (1, 0, 0))
        with self.assertRaises(ValueError):
            a.nlerp(-a, .5)

    def test_slerp_short_arc_and_length(self):
        a, b = Vector3(2, 0, 0), Vector3(0, 4, 0)
        for ratio in (-.5, .25, .5, .75, 1.5):
            radius = 2 + 2*ratio
            expected = (radius*cos(pi*ratio/2), radius*sin(pi*ratio/2), 0)
            self.assertVectorClose(a.slerp(b, ratio), expected)
            self.assertVectorClose(b.slerp(a, 1-ratio), expected)
        # In 3D this defines a plane tilted away from all principal planes.
        a, b = Vector3(1, 0, 1), Vector3(0, 2, 0)
        result = a.slerp(b, .5)
        self.assertAlmostEqual(result.r, (sqrt(2)+2)/2, places=12)
        self.assertAlmostEqual(result.dot(a.cross(b)), 0, places=12)
        self.assertVectorClose(a.slerp(b, .5, axis=(1, 0, 0)), result)

    def test_slerp_opposite_axes(self):
        for axis, expected in (((0, 0, 1), (0, 3, 0)),
                               ((0, 0, -4), (0, -3, 0)),
                               ((0, 2, 0), (0, 0, -3))):
            a, b = Vector3(2, 0, 0), Vector3(-4, 0, 0)
            self.assertVectorClose(a.slerp(b, .5, axis=axis), expected)
        a, b = Vector3(2, 3, 6), Vector3(-4, -6, -12)
        axis = a.cross(Vector3(1, 2, 3))
        result = a.slerp(b, .5, axis=axis)
        self.assertAlmostEqual(result.r, 10.5, places=12)
        self.assertAlmostEqual(result.dot(a), 0, places=12)
        self.assertAlmostEqual(result.dot(axis), 0, places=12)
        self.assertVectorClose(a.slerp(b, .5, axis=-axis), -result)
        with self.assertRaises(ValueError):
            a.slerp(b, .5)
        with self.assertRaises(ValueError):
            a.slerp(b, .5, axis=a)
        with self.assertRaises(ValueError):
            a.slerp(b, 0, axis=vector3_zero)
        with self.assertRaises(ValueError):
            a.slerp(vector3_zero, .5)
        with self.assertRaises(ValueError):
            vector3_zero.slerp(a, .5)

    def test_slerp_parallel_and_nearly_opposite(self):
        self.assertVectorClose(Vector3(1, 2, 3).slerp(Vector3(2, 4, 6), .5),
                               (1.5, 3, 4.5))
        for sign in (-1, 1):
            a, b = Vector3(1, 0, 0), Vector3(-1, sign*1e-8, 0)
            result = a.slerp(b, .5)
            self.assertVectorClose(result, (5e-9, sign, 0))
            self.assertAlmostEqual(result.r, 1, places=12)
        result = Vector3(1, 0, 0).slerp(Vector3(1, 1e-8, 0), .5)
        self.assertVectorClose(result, (1, 5e-9, 0))

    def test_deterministic_geometry_properties(self):
        rng = random.Random(314159)
        for _ in range(50):
            a = Vector3(*(rng.uniform(-5, 5) for _ in range(3)))
            b = Vector3(*(rng.uniform(-5, 5) for _ in range(3)))
            angle = rng.uniform(-pi, pi)
            self.assertAlmostEqual(a.dot(b), b.dot(a), places=12)
            self.assertVectorClose(a.cross(b), -b.cross(a))
            self.assertAlmostEqual(a.cross(b).dot(a), 0, places=11)
            rotated = a.rotated(b, angle)
            self.assertAlmostEqual(rotated.r, a.r, places=11)
            self.assertVectorClose(rotated.rotated(b, -angle), a, places=11)
            result = a.slerp(b, .3)
            self.assertAlmostEqual(result.r, .7*a.r + .3*b.r, places=11)
            self.assertVectorClose(result, b.slerp(a, .7), places=11)


if __name__ == '__main__':
    unittest.main()
