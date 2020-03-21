#!/usr/bin/env python3
#
# vec
# Copyright 2019-2020 by Larry Hastings
#
# A collection of classes for vectors and rects
# designed to be convenient for use in video games.
#
# Features:
#   * Vectors are immutable.
#   * Vectors support both cartesian and polar coordinates,
#     and lazily calculate them as needed.
#   * There is a singleton zero vector (vector2_zero).
#   * APIs accept a wide range of values where possible.
#     For example, these all work:
#          Vector2(1, 2) + [3, 4]
#          Vector2(1, 2) + (3, 4)
#          Vector2(1, 2) + types.SimpleNamespace(x=3, y=4)
#
# Future plans:
#   * Rect2
#   * Vector3
#   * rewrite as C extension for speedy speeds
#
# TODO
#   * figure out _polar when creating new vectors automatically e.g. v*3, v.normalized()

"""
A collection of classes for vectors and rects
designed to be convenient for use in video games.
"""

from math import sin, cos, atan2, sqrt, pi, tau
import sys
import collections.abc

__version__ = "0.5"


_optimize = sys.flags.optimize



def normalize_angle(theta):
    """
    we keep theta in the range
        -pi <= theta < pi
    (symmetric with two's compliment)
    """
    while theta >= pi:
        theta -= tau
    while theta < -pi:
        theta += tau
    return theta


def normalize_polar(r, theta):
    if r == 0:
        return r, theta
    if r < 0:
        r = -r
        theta = theta - pi
    return r, normalize_angle(theta)


#
# All these return vector2_zero:
#   Vector2()
#   Vector2(None, None)
#   Vector2(None)
#   Vector2(0, 0)
#   Vector2((0, 0))
#   Vector2([0, 0])
#   Vector2(types.SimpleNamespace(x=0, y=0))
#
# These return non-zero vectors:
#   Vector2(1, 2)
#   Vector2(any_vector2_object_that_isnt_0_0)
#   Vector2((1, 2))
#   Vector2([1, 2])
#   Vector2(types.SimpleNamespace(x=1, y=2))
#   Vector2(r=1, theta=math.pi/2)
#

valid_types_for_x = (int, float, collections.abc.Iterable)
valid_types_for_parameter = (int, float)

class Vector2Metaclass(type):

    def raise_value_error(self, s):
        raise ValueError(f"'{self.__class__.__name__}': {s}")

    def __call__(self, x=None, y=None, *, r=None, theta=None, r_squared=None, _polar=None):

        # All input validation is done here.
        #
        # You must specify either
        #    None for all arguments,
        # or
        #    both x and y,
        # or
        #    both r and theta.
        # If you specify one of those two pairs, then you
        # may specify any additional arguments you like.
        #
        # If you specify x but not y, x may optionally
        # be an object that represents both x and y.
        # (As in, you specify one argument which represents
        # a vector all by itself.)
        # The following objects are permitted:
        #   * an iterable with two items
        #   * an object exposing attributes named "x" and "y"
        # (For the purposes of input validation, this counts
        # as specifying "both x and y".)
        #
        # When optimization is not enabled
        # (aka when python is run without the "-O" flag),
        # Vector2 will do additional input validation
        # on its parameters:
        # it will confirm that any additional parameters
        # match their computed values, and throw an exception
        # if they don't.
        #
        # When optimization is not enabled,
        # Vector2 will dumbly keep any additional parameters
        # passed in and not do any validation on them.
        # (This is in keeping with the "Consenting adults" rule.)

        # print(f"Vector2Metaclass(self={self}, x={x}, y={y}, r={r}, theta={theta}, r_squared={r_squared})")

        # step 1: handle x if it's a special object
        if x is not None:
            if not isinstance(x, (int, float)):
                if (y is not None) or (r is not None) or (theta is not None) or (r_squared is not None):
                    self.raise_value_error("when x is an object, you must not specify y, r, or theta")
                if isinstance(x, Vector2):
                    return x
                elif (hasattr(x, 'x') and hasattr(x, 'y')):
                    y = x.y
                    x = x.x
                else:
                    if not hasattr(x, "__iter__"):
                        self.raise_value_error("x must be int, float, Vector2, object with .x and .y, or iterable")
                    i = iter(x)
                    x = next(i)
                    y = next(i)
                    try:
                        next(i)
                        self.raise_value_error("if x is an iterable, it must contain exactly 2 items")
                    except StopIteration:
                        pass
        elif theta is not None:
            theta = normalize_angle(theta)
        elif (y is None) and (r is None):
            return vector2_zero

        if _polar is None:
            _polar = (x is None) or (y is None)

        if not _polar:
            if not (isinstance(x, valid_types_for_parameter) and isinstance(y, valid_types_for_parameter)):
                self.raise_value_error("x and y must be either int or float")
            if not (x or y):
                assert r in (0, 0.0, None)
                assert r_squared in (0, 0.0, None)
                return vector2_zero
            value = super().__call__(x=x, y=y)
            if r_squared is not None:
                assert r_squared == value.r_squared
                object.__setattr__(self, 'r_squared', r_squared)
            if r is not None:
                assert r == value.r
                object.__setattr__(self, 'r', r)
            if theta is not None:
                assert theta == value.theta
                object.__setattr__(self, 'theta', theta)
            return value

        # _polar is True
        if (r == 0):
            if (x not in (None, 0)) or (y not in (None, 0)):
                self.raise_value_error("you must specify either (x, y) or (r, theta)")
            return vector2_zero
        if (r is None) or (theta is None):
            self.raise_value_error("you must specify either (x, y) or (r, theta)")

        value = super().__call__(r=r, theta=theta)
        if r_squared is not None:
            assert r_squared == value.r_squared
            object.__setattr__(self, 'r_squared', r_squared)
        if x is not None:
            assert x == value.x
            object.__setattr__(self, 'x', x)
        if y is not None:
            assert y == value.y
            object.__setattr__(self, 'y', y)
        return value



class Vector2(metaclass=Vector2Metaclass):

    __slots__ = ['x', 'y', 'r', 'theta', 'r_squared', '_polar']

    def __init__(self, x=None, y=None, *, r=None, theta=None, r_squared=None, _polar=None):
        # we don't have to handle vector2_zero here.
        # (or any input validation whatsoever.)
        # it's all handled in the metaclass.

        if x is not None:
            object.__setattr__(self, 'x', x)
        if y is not None:
            object.__setattr__(self, 'y', y)
        if r is not None:
            object.__setattr__(self, 'r', r)
        if theta is not None:
            object.__setattr__(self, 'theta', theta)
        if r_squared is not None:
            object.__setattr__(self, 'r_squared', r_squared)
        if _polar is None:
            _polar = (x is None) or (y is None)
        object.__setattr__(self, '_polar', _polar)
        assert self._validate()

    @classmethod
    def from_polar(cls, r, theta):
        return cls(r=r, theta=theta)


    def _get_raw_slots(self):
        try:
            x = object.__getattribute__(self,  'x')
        except AttributeError:
            x = None

        try:
            y = object.__getattribute__(self,  'y')
        except AttributeError:
            y = None

        try:
            r = object.__getattribute__(self,  'r')
        except AttributeError:
            r = None

        try:
            theta = object.__getattribute__(self,  'theta')
        except AttributeError:
            theta = None

        try:
            r_squared = object.__getattribute__(self,  'r_squared')
        except AttributeError:
            r_squared = None

        try:
            _polar = object.__getattribute__(self,  '_polar')
        except AttributeError:
            _polar = None

        return x, y, r, theta, r_squared, _polar

    def _validate(self):
        x, y, r, theta, r_squared, _polar = self._get_raw_slots()
        if _polar:
            assert (r is not None) and (theta is not None)
        else:
            assert (x is not None) and (y is not None)
        if r is not None:
            assert r >= 0
        if theta is not None:
            assert -pi <= theta < pi
        return True


    def __repr__(self):
        assert self._validate()
        if not self._polar:
            return f"<{self.__class__.__name__} ({self.x}, {self.y})>"
        return f"<{self.__class__.__name__} r={self.r} theta={self.theta})>"


    def __setattr__(self, name, value):
        assert self._validate()
        raise TypeError(f"'{self.__class__.__name__}' object is immutable")

    ##
    ## Note complexity here!
    ##
    ## Doing this lazy computation means that we can't use normal
    ## methods for getting / setting / has-attr-ing attributes.
    ##
    ## hasattr(Vector2(), "x") will call __getattr__ to get it.
    ## so we can't use hasattr to test whether or not an attribute
    ## has been cached yet.  the only way that works (that I know of)
    ## is object.__getattribute__ and catch AttributeError.
    ##
    ##

    def __getattr__(self, name):
        assert self._validate()
        if name == 'x':
            x = cos(self.theta) * self.r
            object.__setattr__(self, 'x', x)
            return x
        if name == 'y':
            y = sin(self.theta) * self.r
            object.__setattr__(self, 'y', y)
            return y
        if name == 'r_squared':
            try:
                r = object.__getattribute__(self,  'r')
                r_squared = r * r
            except AttributeError:
                r_squared = (self.x * self.x) + (self.y * self.y)
            object.__setattr__(self, 'r_squared', r_squared)
            return r_squared
        if name == 'r':
            r = sqrt(self.r_squared)
            object.__setattr__(self, 'r', r)
            return r
        if name == 'theta':
            if self.x == self.y == 0:
                raise ValueError("theta is undefined for a null vector")
            theta = atan2(self.y, self.x)
            object.__setattr__(self, 'theta', theta)
            return theta
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute " + name)

    ##
    ## iterator protocol
    ##

    def __len__(self):
        assert self._validate()
        return 2

    def __iter__(self):
        assert self._validate()
        yield self.x
        yield self.y

    ##
    ## sequence protocol
    ##

    def __getitem__(self, index):
        assert self._validate()
        index = index.__index__()
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError(f"'{self.__class__.__name__}' index out of range")

    def __setitem__(self, index, value):
        assert self._validate()
        raise TypeError(f"'{self.__class__.__name__}' object does not support item assignment")

    ##
    ## binary operators
    ##

    def __add__(self, other):
        assert self._validate()
        if other is vector2_zero:
            return self
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        else:
            assert other._validate()
        # if both vectors are currently in polar and they have the same theta,
        # we can just add the magnitudes.  otherwise, punt back to cartesian.
        #
        # there is a way to add vectors without leaving the polar domain...
        # but it requires roughly the same number of triginometry operations
        # (cos, arccos, etc), so punting back to cartesian is way easier and
        # not particularly slower.
        if self._polar and other._polar and (self.theta == other.theta):
            return self.__class__(r=self.r + other.r, theta=self.theta)
        return self.__class__(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        assert self._validate()
        if other is vector2_zero:
            return self
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        else:
            assert other._validate()
        # see comment in __add__
        if self._polar and other._polar and (self.theta == other.theta):
            r = self.r - other.r
            if r == 0:
                return vector2_zero
            theta=self.theta
            if r < 0:
                theta = normalize_angle(theta - pi)
                r = -r
            return self.__class__(r=r, theta=theta)
        return self.__class__(self.x - other.x, self.y - other.y)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        assert self._validate()
        if not isinstance(other, (int, float)):
            raise ValueError(f"'{self.__class__.__name__}' can only be multiplied by scalars (int, float), use .dot or .cross")
        if other == 1:
            return self
        if other == 0:
            return vector2_zero
        # do the cheapest math we can to produce the new object
        if not self._polar:
            result = self.__class__(self.x * other, self.y * other)
            # theta hasn't changed!  (it might be negated, that's it.)
            # if we calculated it already, preserve it.
            try:
                theta = object.__getattribute__(self, 'theta')
                if other < 0:
                    theta = normalize_angle(theta - pi)
                object.__setattr__(self, 'theta', theta)
            except AttributeError:
                pass
            try:
                r = object.__getattribute__(self, 'r')
                object.__setattr__(self, 'r', r * abs(other))
            except AttributeError:
                pass
            return result
        return self.__class__(r=self.r * other, theta=self.theta)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        assert self._validate()
        if not isinstance(other, (int, float)):
            raise ValueError(f"'{self.__class__.__name__}' can only be divided by scalars (int, float)")
        if other == 1:
            return self
        if other == 0:
            return vector2_zero
        # do the cheapest math we can to produce the new object
        if not self._polar:
            x = object.__getattribute__(self, 'x')
            y = object.__getattribute__(self, 'y')
            result = self.__class__(x / other, y / other)
            # theta hasn't changed!  (it might be negated, that's it.)
            # if we calculated it already, preserve it.
            try:
                theta = object.__getattribute__(self, 'theta')
                if other < 0:
                    theta = normalize_angle(theta - pi)
                object.__setattr__(self, 'theta', theta)
            except AttributeError:
                pass
            try:
                r = object.__getattribute__(self, 'r')
                object.__setattr__(self, 'r', r / abs(other))
            except AttributeError:
                pass
            return result
        return self.__class__(r=self.r / other, theta=self.theta)

    def __eq__(self, other):
        assert self._validate()
        if self is other:
            return True
        if not isinstance(other, Vector2):
            return False
        assert other._validate()
        # if both are polar, do the comparison in polar.
        # otherwise do the comparison in cartesian.
        # (if only one of the two vectors is cartesian, force the other to be cartesian too.)
        if self._polar and other._polar:
            return (self.theta == other.theta) and (self.r == other.r)
        return (self.x == other.x) and (self.y == other.y)

    def __ne__(self, other):
        return not self.__eq__(other)

    ##
    ## unary operators
    ##

    def __pos__(self):
        assert self._validate()
        return self

    def __neg__(self):
        # preserve everything we can about the original object
        assert self._validate()

        try:
            x = -object.__getattribute__(self, 'x')
        except AttributeError:
            x = None

        try:
            y = -object.__getattribute__(self, 'y')
        except AttributeError:
            y = None

        try:
            r = object.__getattribute__(self, 'r')
        except AttributeError:
            r = None

        try:
            theta = normalize_angle(object.__getattribute__(self, 'theta') + pi)
        except AttributeError:
            theta = None

        try:
            r_squared = object.__getattribute__(self, 'r_squared')
        except AttributeError:
            r_squared = None

        result = self.__class__(x=x, y=y, r=r, theta=theta, _polar=self._polar)
        if r_squared is not None:
            object.__setattr__(self, 'r_squared', r_squared)
        return result

    def __bool__(self):
        assert self._validate()
        return self is not vector_zero

    def __hash__(self):
        assert self._validate()
        return hash((self.x, self.y))

    ##
    ## methods
    ##

    def scaled(self, multiplier):
        return self * multiplier

    def scaled_to_length(self, r):
        assert self._validate()
        if self.r == r:
            return self
        return self.__class__(r=r, theta=self.theta)

    def normalized(self):
        assert self._validate()
        if self.r == 1:
            return self
        return self.__class__(r=1, theta=self.theta)

    def rotated(self, theta):
        assert self._validate()
        theta = normalize_angle(theta)
        if theta == 0:
            return self

        try:
            x = object.__getattribute__(self, 'x')
        except AttributeError:
            x = None

        try:
            y = object.__getattribute__(self, 'y')
        except AttributeError:
            y = None

        try:
            r = object.__getattribute__(self, 'r')
        except AttributeError:
            r = None

        try:
            computed_theta = normalize_angle(object.__getattribute__(self, 'theta') + theta)
        except AttributeError:
            computed_theta = None

        try:
            r_squared = object.__getattribute__(self, 'r_squared')
        except AttributeError:
            r_squared = None

        # if self currently has cartesian coordinates,
        # then do the calculation in cartesian
        # (most people use cartesian way more than polar)
        if not self._polar:
            sin_theta = sin(theta)
            cos_theta = cos(theta)
            return self.__class__(
                x=((self.x * cos_theta) - (self.y * sin_theta)),
                y=((self.x * sin_theta) + (self.y * cos_theta)),
                r=r,
                theta=computed_theta,
                r_squared=r_squared,
                _polar=False,
                )
        return self.__class__(
            r=self.r,
            theta=normalize_angle(self.theta + theta),
            r_squared=r_squared,
            _polar=True,
            )

    def dot(self, other):
        assert self._validate()
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        assert other._validate()
        return sqrt(self.r_squared + other.r_squared) * cos(self.theta - other.theta)

    def cross(self, other):
        # technically, there is no "cross product" for 2D vectors.
        # so, technically, this is the "perpendicular dot product",
        #   or "perp dot product" for short.
        if self == other:
            return 0
        assert self._validate()
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        else:
            assert other._validate()
        return (self.x * other.y) - (self.y * other.x)

    def polar(self):
        return (self.r, self.theta)

    def lerp(self, other, ratio):
        assert self._validate()
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        else:
            assert other._validate()
        if ratio == 0:
            return self
        if ratio == 1:
            return other
        return self.__class__(
            self.x + ((other.x - self.x) * ratio),
            self.y + ((other.y - self.y) * ratio),
            )

    # .slerp()
    def slerp(self, other, ratio):
        raise NotImplementedError("no slerp yet")



# Our special singleton zero vector
# is hand-crafted by artisans for your pleasure.
vector2_zero = Vector2(1, 1)
object.__setattr__(vector2_zero, "x", 0)
object.__setattr__(vector2_zero, "y", 0)
object.__setattr__(vector2_zero, "r", 0)
object.__setattr__(vector2_zero, "r_squared", 0)
object.__setattr__(vector2_zero, "theta", None)
object.__setattr__(vector2_zero, "_polar", False)
