#!/usr/bin/env python3
#
# vec
# Copyright 2019-2023 by Larry Hastings
# See LICENSE for license information.
#
# A collection of classes for vectors and rects
# designed to be convenient for use in video games.
#
# Features:
#   * Vectors are immutable.
#   * Vectors support both cartesian and polar coordinates.
#   * If you ask a vector for a value it doesn't have,
#     it lazily calculates it on demand and caches the result.
#   * There is a singleton zero vector (vector2_zero).
#     All zero vectors are this same vector ("is" test will pass).
#   * The APIs accept a wide range of values where sensible.
#     For example, these all work:
#          Vector2(1, 2) + [3, 4]
#          Vector2(1, 2) + (3, 4)
#          Vector2(1, 2) + types.SimpleNamespace(x=3, y=4)
#
# Future plans:
#   * rewrite giant nest of boolean logic in metaclass.__call__
#     to be data-driven
#        * giant precomputed table of constants
#        * for each variable x, y, r, theta
#             * three states: not set, set to zero/none, set to nonzero
#             * convert into integer 0, 1, 2
#             * multiply together, (state(x) * 27) + (state(y) * 9) + (state(r) * 3) + state(theta)
#               (total 81 entries)
#             * table tells you what to do:
#                  * construct new vector
#                  * return vector2_zero
#                  * raise ValueError with this message
#                  * raise TypeError with this message
#        * compute table by splitting off the nest of boolean logic
#          into its own function, driving it with every value,
#          and observing what it does
#        * rewrite exception messages to need .format
#
#   * Rect2
#   * Vector3
#   * rewrite as C extension for speedy speeds

"""
A 2-dimensional vector class designed to be convenient
and performant enough for use in video games.
"""

from math import acos, atan2, cos, pi, sin, sqrt, tau
from collections.abc import Iterable, Set, Mapping

__version__ = "0.6"


pi_over_two = pi/2
negative_pi = -pi
negative_pi_over_two = -pi_over_two



def normalize_angle(theta):
    """
    we keep theta in the range
        -pi <= theta < pi
    (symmetric with two's compliment)
    """
    while theta >= pi:
        theta -= tau
    while theta < negative_pi:
        theta += tau
    return theta


def normalize_polar(r, theta):
    if r == 0:
        return 0, None
    if r < 0:
        r = -r
        theta = theta - pi
    return r, normalize_angle(theta)

def negate_angle(theta):
    return normalize_angle(theta + pi)

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

_invalid_types_for_iterable_x = (Set, Mapping)
permitted_coordinate_types = ()

def permit_coordinate_type(t):
    global permitted_coordinate_types
    permitted_coordinate_types = permitted_coordinate_types + (t,)

permit_coordinate_type(int)
permit_coordinate_type(float)

_sentinel = object()

class Vector2Metaclass(type):

    def __call__(self, x=_sentinel, y=_sentinel, *, r=_sentinel, theta=_sentinel, r_squared=_sentinel):

        # All input validation is done here.
        #
        # You must either
        #    not specify x, y, r, or theta,
        # or
        #    specify both x and y,
        #    *or* specify x but not y if x is an object that contains both an x and a y
        #    (x is either an iterable of exactly two things, or an object with .x and .y attributes),
        # or
        #    specify both r and theta.
        #
        # If you specify x and y, or specify r and theta, then you
        # may specify any additional arguments you like.
        #
        # print(f"Vector2Metaclass(self={self}, x={x}, y={y}, r={r}, theta={theta}, r_squared={r_squared})")

        x_source = "x"
        y_source = "y"

        if x is not _sentinel:
            # handle x if it's a special object
            if isinstance(x, Vector2):
                if (y is not _sentinel) or (r is not _sentinel) or (theta is not _sentinel) or (r_squared is not _sentinel):
                    raise ValueError(f"{self.__name__}: when x is an object, you must not specify y, r, or theta")
                return x
            elif (hasattr(x, 'x') and hasattr(x, 'y')):
                if (y is not _sentinel) or (r is not _sentinel) or (theta is not _sentinel) or (r_squared is not _sentinel):
                    raise ValueError(f"{self.__name__}: when x is an object, you must not specify y, r, or theta")
                x_source = "x.x"
                y_source = "x.y"
                y = x.y
                x = x.x
            elif hasattr(x, "__iter__"):
                if isinstance(x, _invalid_types_for_iterable_x):
                    raise TypeError(f"{self.__name__}: if x is an iterable, it must be ordered, not {x.__class__.__name__}")
                if (y is not _sentinel) or (r is not _sentinel) or (theta is not _sentinel) or (r_squared is not _sentinel):
                    raise ValueError(f"{self.__name__}: when x is an object, you must not specify y, r, or theta")
                i = iter(x)
                x_source = "x[0]"
                y_source = "x[1]"
                try:
                    x = next(i)
                    y = next(i)
                except StopIteration:
                    raise ValueError(f"{self.__name__}: if x is an iterable, it must contain exactly 2 items")
                try:
                    next(i)
                    raise ValueError(f"{self.__name__}: if x is an iterable, it must contain exactly 2 items")
                except StopIteration:
                    pass

        x_is_not_set = x is _sentinel
        x_is_set = not x_is_not_set

        y_is_not_set = y is _sentinel
        y_is_set = not y_is_not_set

        x_and_y = x_is_set and y_is_set
        not_x_or_y = x_is_not_set and y_is_not_set

        r_is_not_set = r is _sentinel
        r_is_set = not r_is_not_set

        theta_is_not_set = theta is _sentinel
        theta_is_set = not theta_is_not_set

        r_and_theta = r_is_set and theta_is_set
        not_r_or_theta = r_is_not_set and theta_is_not_set

        r_squared_is_set = r_squared is not _sentinel


        if not_x_or_y and not_r_or_theta:
            return vector2_zero

        if not_r_or_theta:
            # neither r nor theta is set.
            # either x or y must be set.
            # (otherwise we'd already be vector2_zero.)
            # but both must be set.
            if x_is_not_set or y_is_not_set:
                raise ValueError(f"{self.__name__}: you must specify x and y together")

        if not_x_or_y:
            # neither x nor y is set.
            # either r or theta must be set.
            # (otherwise we'd already be vector2_zero.)
            # but both must be set.
            if r_is_not_set or theta_is_not_set:
                raise ValueError(f"{self.__name__}: you must specify r and theta together")

        args = {}
        if x_is_set:
            if not isinstance(x, permitted_coordinate_types):
                raise TypeError(f"{self.__name__}: {x_source} must be int, float, Vector2, object with .x and .y, or iterable, not {x}")
            args['x'] = x
        if y_is_set:
            if not isinstance(y, permitted_coordinate_types):
                raise TypeError(f"{self.__name__}: {y_source} must be int or float, not {y}")
            args['y'] = y
        if r_is_set:
            if not isinstance(r, permitted_coordinate_types):
                raise TypeError(f"{self.__name__}: r must be int or float, not {r}")
            args['r'] = r
        if theta_is_set:
            if theta is None:
                if r_is_set and r:
                    raise TypeError(f"{self.__name__}: theta is None, can't be None when r is {r}")
            else:
                if not isinstance(theta, permitted_coordinate_types):
                    raise TypeError(f"{self.__name__}: theta must be int or float, not {theta}")
                if r_is_set and (not r):
                    raise TypeError(f"{self.__name__}: theta is {theta}, must be None when r is 0")
                theta = normalize_angle(theta)

        if r_is_set and theta_is_set:
            # normalize r
            if r < 0:
                r = -r
                args['r'] = r
                theta = negate_angle(theta)

            args['r'] = r
            args['theta'] = theta
        elif r_is_set:
            if r < 0:
                raise ValueError(f"{self.__name__}: can't specify r < 0 if you don't specify theta")
            args['r'] = r
        else:
            args['theta'] = theta

        if r_squared_is_set:
            if not isinstance(r_squared, permitted_coordinate_types):
                raise TypeError(f"{self.__name__}: r_squared must be int or float, not {r_squared}")
            args['r_squared'] = r_squared

        # at this point, if <whatever>_is_set is True, <whatever> is confirmed a legal value

        if x_and_y:
            if not (x or y):
                if r_is_set and r:
                    raise ValueError(f"{self.__name__}: r is {r}, must be 0 when vector=(0, 0)")
                if theta_is_set and (theta is not None):
                    raise TypeError(f"{self.__name__}: theta is {theta}, must be None when vector=(0, 0)")
                if r_squared_is_set and r_squared:
                    raise ValueError(f"{self.__name__}: r is {r}, must be 0 when vector=(0, 0)")
                return vector2_zero

            # at least one of x or y must be nonzero.  therefore r must be nonzero, and theta must not be None.
            if r_is_set and (not r):
                raise ValueError(f"{self.__name__}: r is {r}, must be > 0 when vector=({x}, {y})")
            if theta_is_set and (theta is None):
                raise TypeError(f"{self.__name__}: theta is {theta}, must not be None when vector=({x}, {y})")
            if r_squared_is_set and (not r_squared):
                raise ValueError(f"{self.__name__}: r_squared is {r_squared}, must be > 0 when vector=({x}, {y})")
        else:
            assert r_and_theta
            # we already checked that theta and r agree.
            # if theta is set, and it's None, and r_and_theta, then r must be set and must be 0.
            # and if theta is set, and it's not None, and r_and_theta, then r must be set and must not be 0.
            # so we don't need to double-check r here, we can check theta *or* r with confidence.
            if theta is None:
                if x_is_set and x:
                    raise ValueError(f"{self.__name__}: x is {x}, must be 0 when r is 0")
                if y_is_set and y:
                    raise ValueError(f"{self.__name__}: y is {y}, must be 0 when r is 0")
                if r_squared_is_set and r_squared:
                    raise ValueError(f"{self.__name__}: r is {r}, must be 0 when vector=(0, 0)")
                return vector2_zero

            if r_squared_is_set and (not r_squared):
                raise ValueError(f"{self.__name__}: r_squared is {r_squared}, must be > 0 when r is {r}")

        return super().__call__(**args)



class Vector2(metaclass=Vector2Metaclass):

    __slots__ = ['x', 'y', 'r', 'theta', 'r_squared', '_cartesian', '_polar', '_hash']

    def __init__(self, x=_sentinel, y=_sentinel, *, r=_sentinel, theta=_sentinel, r_squared=_sentinel):
        # we don't have to handle vector2_zero here.
        # (or any input validation whatsoever.)
        # it's all handled in the metaclass.

        _set = object.__setattr__
        def _set(self, name, value):
            object.__setattr__(self, name, value)
        def set(name, value):
            if value is not _sentinel:
                _set(self, name, value)

        set('x', x)
        set('y', y)
        set('r', r)
        set('theta', theta)
        set('r_squared', r_squared)
        # _cartesian and _polar are *counters*
        # if they're == 2, this vector has that coordinate set
        # (e.g. if _cartesian == 2, this vector has a complete set of cartesian coordinates)
        _cartesian = (x is not _sentinel) + (y is not _sentinel)
        _set(self, '_cartesian', _cartesian)
        _polar = (r is not _sentinel) + (theta is not _sentinel)
        _set(self, '_polar', _polar)

    def __repr__(self):
        fields = []

        self_as_object = super(Vector2, self)
        get = self_as_object.__getattribute__
        sentinel = object()


        if self._cartesian == 2:
            x = self.x
            y = self.y
        else:
            try:
                x = get('x')
            except AttributeError:
                x = sentinel

            try:
                y = get('y')
            except AttributeError:
                y = sentinel

        if (x is not sentinel) and (y is not sentinel):
            fields.append(f"{x}, {y}")
        else:
            if x is not sentinel:
                fields.append(f"x={x}")
            if y is not sentinel:
                fields.append(f"y={y}")

        if self._polar == 2:
            fields.append(f"r={self.r}, theta={self.theta}")
            remaining_fields = ('r_squared',)
        else:
            remaining_fields = ('r', 'theta', 'r_squared')
        for attr in remaining_fields:
            try:
                value = get(attr)
                fields.append(f"{attr}={value}")
            except AttributeError:
                pass

        text = ", ".join(fields)
        return f"{self.__class__.__name__}({text})"

    @classmethod
    def from_polar(cls, r, theta):
        return cls(r=r, theta=theta)

    def polar(self):
        return (self.r, self.theta)

    ##
    ## descriptor protocol
    ##
    ## Note the complexity here!
    ##
    ## Vector2 features lazy computation of attributes.
    ## The way we do this is, we leave the attribute unset,
    ## and if somebody asks for an unset attribute,
    ## Python calls our __getattr__ and we compute it on the fly.
    ##
    ## This method that we can't use normal methods for
    ## getting / setting / has-attr-ing attributes.
    ##
    ## hasattr(Vector2(), "x") will call __getattr__ to get it.
    ## so we can't use hasattr to test whether or not an attribute
    ## has been cached yet.
    ##
    ## How do we handle it?
    ## Part of the answer is, we carefully set and use _polar and _cartesian.
    ## The other part is, super(Vector2, o).__getattribute__ bypasses
    ## Vector2.__getattr__, allowing us to get the underlying attributes
    ## (or raise AttributeError when they're not defined).
    ##

    def __getattr__(self, name):
        if name == 'x':
            assert self._polar == 2, f"_polar should be 2 but is {self._polar!r}"
            x = cos(self.theta) * self.r
            object.__setattr__(self, 'x', x)
            object.__setattr__(self, '_cartesian', self._cartesian + 1)
            return x
        if name == 'y':
            assert self._polar == 2, f"_polar should be 2 but is {self._polar!r}"
            y = sin(self.theta) * self.r
            object.__setattr__(self, 'y', y)
            object.__setattr__(self, '_cartesian', self._cartesian + 1)
            return y
        if name == 'r_squared':
            if self._polar == 2:
                r = self.r
                r_squared = r * r
            else:
                assert self._cartesian == 2
                r_squared = (self.x * self.x) + (self.y * self.y)
            object.__setattr__(self, 'r_squared', r_squared)
            return r_squared
        if name == 'r':
            assert self._cartesian == 2, f"_cartesian should be 2 but is {self._cartesian!r}"
            r = sqrt(self.r_squared)
            object.__setattr__(self, 'r', r)
            object.__setattr__(self, '_polar', self._polar + 1)
            return r
        if name == 'theta':
            # if x and y were both zero, self would be vector2_zero.
            # but vector2_zero already has theta set.
            # since we're here, we're looking up theta, which means theta is not set,
            # which means we can't be vector2_zero, which means either x or y is nonzero.
            assert self._cartesian == 2, f"_cartesian should be 2 but is {self._cartesian!r}"
            assert self.x or self.y
            theta = normalize_angle(atan2(self.y, self.x))
            object.__setattr__(self, 'theta', theta)
            object.__setattr__(self, '_polar', self._polar + 1)
            return theta
        if name == '_hash':
            _hash = hash((self.x, self.y))
            object.__setattr__(self, '_hash', _hash)
            return _hash
        raise AttributeError(f"{self.__class__.__name__}: object has no attribute " + name)

    def __setattr__(self, name, value):
        raise TypeError(f"{self.__class__.__name__}: object is immutable")


    ##
    ## iterator protocol
    ##

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.x
        yield self.y


    ##
    ## sequence protocol
    ##

    def __getitem__(self, index):
        index = index.__index__()
        if (index != 0) and (index != 1):
            raise IndexError(f"{self.__class__.__name__}: index out of range")
        if index == 0:
            return self.x
        return self.y

    def __setitem__(self, index, value):
        raise TypeError(f"{self.__class__.__name__}: object does not support item assignment")


    ##
    ## binary operators
    ##

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Vector2):
            return False

        # if both are polar, do the comparison in polar.
        # otherwise do the comparison in cartesian.
        # (if only one of the two vectors is cartesian, force the other to be cartesian too.)
        if self._polar == other._polar == 2:
            return (self.theta == other.theta) and (self.r == other.r)
        return (self.x == other.x) and (self.y == other.y)

    def __ne__(self, other):
        if self is other:
            return False
        if not isinstance(other, Vector2):
            return True
        # if both are polar, do the comparison in polar.
        # otherwise do the comparison in cartesian.
        # (if only one of the two vectors is cartesian, force the other to be cartesian too.)
        if self._polar == other._polar == 2:
            return (self.theta != other.theta) or (self.r != other.r)
        return (self.x != other.x) or (self.y != other.y)

    def __add__(self, other):
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if other is vector2_zero:
            return self
        if self is vector2_zero:
            return other

        # if both vectors are currently in polar,
        # and they have the exact same theta,
        # we add the magnitudes.
        #
        # if both vectors are currently in polar,
        # but 'other' has the exact *opposite* theta,
        # we subtract other's magnitude.
        #
        # otherwise, punt back to cartesian.
        #
        # there *is* a way to add vectors with different angles without translating
        # to cartesian, but IIUC requires roughly the same number of trigonometry
        # operations (cos, arccos, etc) as translating to cartesian and back.
        # therefore punting back to cartesian is way easier and not particularly slower.
        if (self._polar == other._polar == 2):
            other_theta = other.theta
            other_r = other.r
            theta = self.theta

            if theta != other_theta:
                other_theta = negate_angle(other_theta)
                other_r = -other_r

            # if we have the same angle:
            if (theta == other_theta):
                r = self.r + other_r
                if r == 0:
                    return vector2_zero
                if r < 0:
                    r = -r
                    theta = negate_angle(theta)
                return self.__class__(r=r, theta=theta)

        return self.__class__(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if other is vector2_zero:
            return self
        if other is self:
            return vector2_zero

        # __sub__ works much the same as __add__,
        # see comment in __add__.
        if self._polar == other._polar == 2:
            other_theta = other.theta
            other_r = other.r
            theta = self.theta
            if theta != other_theta:
                other_theta = negate_angle(other_theta)
                other_r = -other_r
            if theta == other_theta:
                r = self.r - other_r
                if r == 0:
                    return vector2_zero
                if r < 0:
                    r = -r
                    theta = negate_angle(theta)
                return self.__class__(r=r, theta=theta)

        return self.__class__(self.x - other.x, self.y - other.y)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError(f"{self.__class__.__name__}: can only be multiplied by scalars (int, float), use .dot or .cross")
        if other == 1:
            return self
        if other == 0:
            return vector2_zero
        if other == -1:
            return -self

        # if theta or r are already defined on self,
        # we can calculate the new values cheaply.
        # (if they aren't, don't bother.)
        self_as_object = super(Vector2, self)
        get = self_as_object.__getattribute__

        # preserve all the fields we can, as cheaply as possible
        kwargs = {}

        if self._cartesian == 2:
            kwargs['x'] = self.x * other
            kwargs['y'] = self.y * other
        else:
            try:
                kwargs['x'] = get('x') * other
            except AttributeError:
                pass
            try:
                kwargs['y'] = get('y') * other
            except AttributeError:
                pass

        try:
            kwargs['r_squared'] = get('r_squared') * (other * other)
        except AttributeError:
            pass

        if self._polar == 2:
            theta = self.theta
            if other < 0:
                other = -other
                theta = negate_angle(theta)
            kwargs['r'] = self.r * other
            kwargs['theta'] = theta
        else:
            try:
                theta = get('theta')
                if other < 0:
                    theta = negate_angle(theta)

                kwargs['theta'] = theta
            except AttributeError:
                pass

            try:
                kwargs['r'] = get('r') * abs(other)
            except AttributeError:
                pass

        return self.__class__(**kwargs)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError(f"{self.__class__.__name__}: can only be divided by scalars (int, float)")
        if other == 1:
            return self
        if other == 0:
            return vector2_zero
        if other == -1:
            return -self

        self_as_object = super(Vector2, self)
        get = self_as_object.__getattribute__

        # preserve all the fields we can, as cheaply as possible
        kwargs = {}

        if self._cartesian == 2:
            kwargs['x'] = self.x / other
            kwargs['y'] = self.y / other
        else:
            try:
                kwargs['x'] = get('x') / other
            except AttributeError:
                pass
            try:
                kwargs['y'] = get('y') / other
            except AttributeError:
                pass

        try:
            kwargs['r_squared'] = get('r_squared') / (other * other)
        except AttributeError:
            pass

        if self._polar == 2:
            theta = self.theta
            if other < 0:
                other = -other
                theta = negate_angle(theta)
            kwargs['r'] = self.r / other
            kwargs['theta'] = theta
        else:
            try:
                theta = get('theta')
                if other < 0:
                    theta = negate_angle(theta)

                kwargs['theta'] = theta
            except AttributeError:
                pass

            try:
                kwargs['r'] = get('r') / abs(other)
            except AttributeError:
                pass

        return self.__class__(**kwargs)

    ##
    ## unary operators
    ##

    def __pos__(self):
        return self

    def __neg__(self):
        if self is vector2_zero:
            return vector2_zero

        # preserve all the attributes we can from the original object.
        self_as_object = super(Vector2, self)
        get = self_as_object.__getattribute__

        kwargs = {}
        try:
            kwargs['x'] = -get('x')
        except AttributeError:
            pass

        try:
            kwargs['y'] = -get('y')
        except AttributeError:
            pass

        try:
            kwargs['r'] = get('r')
        except AttributeError:
            pass

        try:
            kwargs['theta'] = normalize_angle(get('theta') + pi)
        except AttributeError:
            pass

        try:
            kwargs['r_squared'] = get('r_squared')
        except AttributeError:
            pass

        result = self.__class__(**kwargs)
        return result

    def __bool__(self):
        return self is not vector2_zero

    def __hash__(self):
        return self._hash

    ##
    ## methods
    ##

    def scaled(self, multiplier):
        """
        Multiplies vector by a scalar.
        Equivalent to v * multiplier.
        """
        return self * multiplier

    def scaled_to_length(self, r):
        if not r:
            return vector2_zero

        if self is vector2_zero:
            raise ValueError("can't scale vector2_zero to a nonzero length")

        if self._cartesian:
            # x and y can't both be zero.
            # otherwise self would be the zero vector, and we'd already have raised an error.
            # so if either x or y is zero, the other must be nonzero.
            # this is fast *and* exact!
            if not self.x:
                if self.y == r:
                    return self
                return self.__class__(0, r if self.y > 0 else -r)

            if not self.y:
                if self.x == r:
                    return self
                return self.__class__(r if self.x > 0 else -r, 0)

        if self.r == r:
            return self
        return self.__class__(r=r, theta=self.theta)

    def normalized(self):
        if self is vector2_zero:
            raise ValueError("can't normalize vector2_zero")

        if self._cartesian:
            # x and y can't both be zero.
            # otherwise self would be the zero vector, and we'd already have raised an error.
            # so if either x or y is zero, the other must be nonzero.
            # this is fast *and* exact!
            if not self.x:
                if self.y == 1:
                    return self
                return self.__class__(0, 1 if self.y > 0 else -1)

            if not self.y:
                if self.x == 1:
                    return self
                return self.__class__(1 if self.x > 0 else -1, 0)

        if self.r == 1:
            return self
        return self.__class__(r=1, theta=self.theta)

    def rotated(self, theta):
        if not theta:
            return self
        if self is vector2_zero:
            raise ValueError("can't rotate vector2_zero")


        self_as_object = super(Vector2, self)
        get = self_as_object.__getattribute__
        kwargs = {}

        try:
            kwargs['r_squared'] = get('r_squared')
        except AttributeError:
            pass

        theta = normalize_angle(theta)

        # cheapest of all:
        # rotating a cartesian vector by a multiple of ninety degrees.
        # this is fast *and* exact!
        if self._cartesian:
            handled = False
            if theta == pi_over_two:
                handled = True
                x = -self.y
                y = self.x
            elif theta == negative_pi:
                handled = True
                x = -self.x
                y = -self.y
            elif theta == negative_pi_over_two:
                handled = True
                x = self.y
                y = -self.x
            if handled:
                return self.__class__(x, y, **kwargs)

        # if we can't do the cheapest thing, but we have polar,
        # do it in polar.
        # that's cheaper than generalized rotation in cartesian.
        if self._polar == 2:
            return self.__class__(
                r=self.r,
                theta=normalize_angle(self.theta + theta),
                **kwargs,
                )

        try:
            kwargs['r'] = get('r')
        except AttributeError:
            # if r is set, theta won't be set.
            # so we only check for theta if r isn't set.
            try:
                kwargs['theta'] = normalize_angle(get('theta') + theta)
            except AttributeError:
                pass

        sin_theta = sin(theta)
        cos_theta = cos(theta)
        return self.__class__(
            x=((self.x * cos_theta) - (self.y * sin_theta)),
            y=((self.x * sin_theta) + (self.y * cos_theta)),
            **kwargs,
            )

    def dot(self, other):
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if (self is vector2_zero) or (other is vector2_zero):
            return 0
        return (self.x * other.x) + (self.y * other.y)

    def cross(self, other):
        """
        Returns the cross product of two vectors.

        Technically speaking, there *is* no "cross product" for
        2D vectors.  So, technically, this is actually the
        "perpendicular dot product", or "perp dot product" for short.
        (That's what people *actually* want when they say they want
        the "cross product" of two 2D vectors.)
        """
        if self == other:
            return 0
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if (self is vector2_zero) or (other is vector2_zero):
            return 0
        return (self.x * other.y) - (self.y * other.x)

    def lerp(self, other, ratio):
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if not isinstance(ratio, (int, float)):
            raise ValueError(f"{self.__class__.__name__}: lerp ratio must be int or float")

        if self is other:
            return self
        if ratio == 0:
            return self
        if ratio == 1:
            return other

        return self.__class__(
            self.x + ((other.x - self.x) * ratio),
            self.y + ((other.y - self.y) * ratio),
            )

    # https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    def slerp(self, other, ratio):
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if not isinstance(ratio, (int, float)):
            raise ValueError(f"{self.__class__.__name__}: slerp ratio must be int or float")

        if self is other:
            return self
        if ratio == 0:
            return self
        if ratio == 1:
            return other

        self_normalized = self.normalized()
        other_normalized = other.normalized()
        dot = self_normalized.dot(other_normalized)

        dot = max(dot, -1)
        dot = min(dot, 1)

        if dot == 1:
            # if dot is 1, theta is 0, so sin(theta) is 0
            # and we divide by sin(theta), so we divide by 0.
            # so, when dot is 1, switch to linear interpolation.
            return self.lerp(other, ratio)

        theta = acos(dot)
        sin_theta = sin(theta)

        theta_ratio = theta * ratio
        sin_theta_ratio = sin(theta_ratio)

        self_scale = sin(theta - theta_ratio) / sin_theta
        other_scale = sin_theta_ratio / sin_theta

        result = (self * self_scale) + (other * other_scale)

        return result

    # https://keithmaggio.wordpress.com/2011/02/15/math-magician-lerp-slerp-and-nlerp/
    # but adjusted to handle non-normalized vectors
    # (linearly interpolates the length of the lerped from |self| to |other|)
    def nlerp(self, other, ratio):
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if not isinstance(ratio, (int, float)):
            raise ValueError(f"{self.__class__.__name__}: nlerp ratio must be int or float")

        if self is other:
            return self
        if ratio == 0:
            return self
        if ratio == 1:
            return other

        lerped = self.lerp(other, ratio)

        self_r = self.r
        delta_r = other.r - self_r
        linearly_interpolated_r = self_r + (delta_r * ratio)

        result = lerped.scaled_to_length(linearly_interpolated_r)
        return result


# Our custom singleton zero vector
# is hand-crafted by artisans
# to envelop you in luxury.
vector2_zero = Vector2(1, 1)
object.__setattr__(vector2_zero, "x", 0)
object.__setattr__(vector2_zero, "y", 0)
object.__setattr__(vector2_zero, "r", 0)
object.__setattr__(vector2_zero, "theta", None)
object.__setattr__(vector2_zero, "r_squared", 0)
object.__setattr__(vector2_zero, "_cartesian", 2)
object.__setattr__(vector2_zero, "_polar", 2)
object.__setattr__(vector2_zero, "_hash", 0)
