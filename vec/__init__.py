#!/usr/bin/env python3
#
# vec
# Copyright 2019-2024 by Larry Hastings
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
#   * There is a singleton unit vector (vector2_1_1).
#     All unit vectors are this same vector ("is" test will pass):
#         vec.Vector2(1, 1) is vector2_1_1
#         vec.Vector2(1.0, 1.0) is vector2_1_1
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
#   * Vector2 constructor: allow reading 'r', 'theta', and 'r_squared'
#     from namespaces and mappings.  still be strict about it for mappings.
#
#   * Rect2
#   * rewrite as C extension for speedy speeds

"""
Immutable 2D and 3D vector classes designed to be convenient
and performant enough for use in video games.
"""

from math import acos, atan2, cos, fabs, fmod, pi, sin, sqrt, tau
from collections.abc import Iterable, Set, Mapping

__version__ = "0.7.1"


pi_over_two = pi/2
pi_over_four = pi/4
negative_pi = -pi
negative_pi_over_two = -pi_over_two

sqrt_2 = 2 ** 0.5


def normalize_angle(theta):
    """
    we keep theta in the range
        -pi <= theta < pi
    Infinite angles raise ValueError; NaN passes through unchanged.
    """
    if theta >= pi or theta < negative_pi:
        # Repeated subtraction can round back to the original value and
        # loop forever. Reduce modulo the floating-point tau in one step.
        # fmod preserves the sign, avoiding loss of tiny negative angles.
        theta = fmod(theta, tau)
        if theta >= pi:
            theta -= tau
        elif theta < negative_pi:
            theta += tau
    return theta

def negate_angle(theta):
    return normalize_angle(theta - pi)

def normalize_polar(r, theta):
    if r == 0:
        if theta is not None:
            raise TypeError(f"normalize_polar: theta is {theta}, must be None when r is 0")
        return (0, None)

    if theta is None:
        raise TypeError(f"normalize_polar: theta is None, can't be None when r is {r}")
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

# we check mappings separately
_invalid_types_for_iterable = (Set, Mapping)
permitted_coordinate_types = ()

def permit_coordinate_type(t):
    global permitted_coordinate_types
    permitted_coordinate_types = permitted_coordinate_types + (t,)

permit_coordinate_type(int)
permit_coordinate_type(float)

_sentinel = object()

def _vector2_from_xy(cls, x, y):
    # Fast specialized constructor for cartesian Vector2 objects.
    # Supports subclass construction and permitted user coordinate types.
    if cls is not Vector2 or type(x) not in (int, float) or type(y) not in (int, float):
        return cls(x, y)
    if not (x or y):
        return vector2_zero
    if x == 1:
        if y == 1:
            return vector2_1_1
        if not y:
            return vector2_1_0
    elif not x and y == 1:
        return vector2_0_1
    result = object.__new__(cls)
    setattr = object.__setattr__
    setattr(result, 'x', x)
    setattr(result, 'y', y)
    setattr(result, '_cartesian', 2)
    setattr(result, '_polar', 0)
    return result


class Vector2:

    __slots__ = ['x', 'y', 'r', 'theta', 'r_squared', '_cartesian', '_polar', '_hash']

    def __new__(cls, x=_sentinel, y=_sentinel, *, r=_sentinel, theta=_sentinel, r_squared=_sentinel):

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

        x_source = "x"
        y_source = "y"

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

        if x_is_set and (not isinstance(x, permitted_coordinate_types)):
            # handle x if it's a special object
            something_besides_x_is_set = y_is_set or r_is_set or theta_is_set or r_squared_is_set
            if isinstance(x, Vector2):
                if something_besides_x_is_set:
                    raise ValueError(f"{cls.__name__}: when x is an object, you must not specify y, r, or theta")
                return x
            # the while loop is here just so we can early-exit break out of processing.
            # we never actually loop.
            while True:
                if hasattr(x, 'x') and hasattr(x, 'y'):
                    if something_besides_x_is_set:
                        raise ValueError(f"{cls.__name__}: when x is an object, you must not specify y, r, or theta")
                    x_source = "x.x"
                    y_source = "x.y"
                    y = x.y
                    x = x.x
                    not_x_or_y = y_is_not_set = False
                    x_and_y = y_is_set = True
                    # x_is_set and y_is_set
                    break

                if hasattr(x, "__getitem__"):
                    if something_besides_x_is_set:
                        raise ValueError(f"{cls.__name__}: when x is an object, you must not specify y, r, or theta")
                    try:
                        maybe_x = x['x']
                        y = x['y']
                        x_source = "x['x']"
                        y_source = "x['y']"
                        if len(x) != 2:
                            raise ValueError(f"{cls.__name__}: if x is a mapping, it must contain exactly 2 items")
                        x = maybe_x
                        not_x_or_y = y_is_not_set = False
                        x_and_y = y_is_set = True
                        # x_is_set and y_is_set
                        break
                    except KeyError:
                        raise KeyError(f"{cls.__name__}: if x is an mapping, it must contain elements 'x' and 'y'") from None
                    except TypeError:
                        pass

                if hasattr(x, "__iter__"):
                    if isinstance(x, _invalid_types_for_iterable):
                        raise TypeError(f"{cls.__name__}: if x is an iterable, it must be ordered, not {x.__class__.__name__}")
                    if something_besides_x_is_set:
                        raise ValueError(f"{cls.__name__}: when x is an object, you must not specify y, r, or theta")
                    i = iter(x)
                    x_source = "x[0]"
                    y_source = "x[1]"
                    try:
                        x = next(i)
                        y = next(i)
                    except StopIteration:
                        raise ValueError(f"{cls.__name__}: if x is an iterable, it must contain exactly 2 items")
                    try:
                        next(i)
                        raise ValueError(f"{cls.__name__}: if x is an iterable, it must contain exactly 2 items")
                    except StopIteration:
                        pass
                    not_x_or_y = y_is_not_set = False
                    x_and_y = y_is_set = True
                    # x_is_set and y_is_set
                    break

                break


        if not_x_or_y and not_r_or_theta:
            return vector2_zero

        if not_r_or_theta:
            # neither r nor theta is set.
            # either x or y must be set.
            # (otherwise we'd already be vector2_zero.)
            # but both must be set.
            if x_is_not_set or y_is_not_set:
                raise ValueError(f"{cls.__name__}: you must specify x and y together")

        if not_x_or_y:
            # neither x nor y is set.
            # either r or theta must be set.
            # (otherwise we'd already be vector2_zero.)
            # but both must be set.
            if r_is_not_set or theta_is_not_set:
                raise ValueError(f"{cls.__name__}: you must specify r and theta together")

        if x_is_set:
            if not isinstance(x, permitted_coordinate_types):
                raise TypeError(f"{cls.__name__}: {x_source} must be int, float, Vector2, object with .x and .y, or iterable, not {x}")
        if y_is_set:
            if not isinstance(y, permitted_coordinate_types):
                raise TypeError(f"{cls.__name__}: {y_source} must be int or float, not {y}")
        if r_is_set:
            if not isinstance(r, permitted_coordinate_types):
                raise TypeError(f"{cls.__name__}: r must be int or float, not {r}")
        if theta_is_set:
            if theta is None:
                if r_is_set and r:
                    raise TypeError(f"{cls.__name__}: theta is None, can't be None when r is {r}")
            else:
                if not isinstance(theta, permitted_coordinate_types):
                    raise TypeError(f"{cls.__name__}: theta must be int or float, not {theta}")
                if r_is_set and (not r):
                    raise TypeError(f"{cls.__name__}: theta is {theta}, must be None when r is 0")
                theta = normalize_angle(theta)

        if r_is_set and theta_is_set:
            r, theta = normalize_polar(r, theta)
        elif r_is_set:
            if r < 0:
                raise ValueError(f"{cls.__name__}: can't specify r < 0 if you don't specify theta")

        if r_squared_is_set:
            if not isinstance(r_squared, permitted_coordinate_types):
                raise TypeError(f"{cls.__name__}: r_squared must be int or float, not {r_squared}")

        # at this point, if <whatever>_is_set is True, <whatever> is confirmed a legal value

        if x_and_y:
            if not (x or y):
                if r_is_set and r:
                    raise ValueError(f"{cls.__name__}: r is {r}, must be 0 when vector=(0, 0)")
                if theta_is_set and (theta is not None):
                    raise TypeError(f"{cls.__name__}: theta is {theta}, must be None when vector=(0, 0)")
                if r_squared_is_set and r_squared:
                    raise ValueError(f"{cls.__name__}: r is {r}, must be 0 when vector=(0, 0)")
                return vector2_zero

            # at least one of x or y must be nonzero.  therefore r must be nonzero, and theta must not be None.
            if r_is_set and (not r):
                raise ValueError(f"{cls.__name__}: r is {r}, must be > 0 when vector=({x}, {y})")
            if theta_is_set and (theta is None):
                raise TypeError(f"{cls.__name__}: theta is {theta}, must not be None when vector=({x}, {y})")
            if r_squared_is_set and (not r_squared):
                raise ValueError(f"{cls.__name__}: r_squared is {r_squared}, must be > 0 when vector=({x}, {y})")

            if (x == 1):
                if (y == 1):
                    return vector2_1_1
                if not y:
                    return vector2_1_0
            elif (not x) and (y == 1):
                return vector2_0_1

        else:
            assert r_and_theta
            # we already checked that theta and r agree.
            # if theta is set, and it's None, and r_and_theta, then r must be set and must be 0.
            # and if theta is set, and it's not None, and r_and_theta, then r must be set and must not be 0.
            # so we don't need to double-check r here, we can check theta *or* r with confidence.
            if theta is None:
                if x_is_set and x:
                    raise ValueError(f"{cls.__name__}: x is {x}, must be 0 when r is 0")
                if y_is_set and y:
                    raise ValueError(f"{cls.__name__}: y is {y}, must be 0 when r is 0")
                if r_squared_is_set and r_squared:
                    raise ValueError(f"{cls.__name__}: r is {r}, must be 0 when vector=(0, 0)")
                return vector2_zero

            if r_squared_is_set and (not r_squared):
                raise ValueError(f"{cls.__name__}: r_squared is {r_squared}, must be > 0 when r is {r}")

            if (r == 1):
                if (theta == 0):
                    return vector2_1_0
                if (theta == pi_over_two):
                    return vector2_0_1
            elif (r == sqrt_2) and (theta == pi_over_four):
                return vector2_1_1

        # Initialize only newly allocated vectors; reused instances keep their caches.
        self = object.__new__(cls)
        set_attr = object.__setattr__
        if x is not _sentinel:
            set_attr(self, 'x', x)
        if y is not _sentinel:
            set_attr(self, 'y', y)
        if r is not _sentinel:
            set_attr(self, 'r', r)
        if theta is not _sentinel:
            set_attr(self, 'theta', theta)
        if r_squared is not _sentinel:
            set_attr(self, 'r_squared', r_squared)
        # Each counter records how many coordinates of that pair are cached.
        set_attr(self, '_cartesian', (x is not _sentinel) + (y is not _sentinel))
        set_attr(self, '_polar', (r is not _sentinel) + (theta is not _sentinel))
        return self

    def __repr__(self):
        fields = []

        self_as_object = super(Vector2, self)
        get = self_as_object.__getattribute__
        sentinel = object()


        if self._cartesian == 2:
            x = self.x
            y = self.y
        elif self._cartesian:
            try:
                x = get('x')
            except AttributeError:
                x = sentinel

            try:
                y = get('y')
            except AttributeError:
                y = sentinel

        else:
            x = y = sentinel

        suppress_remaining = False
        if (x is not sentinel) and (y is not sentinel):
            fields.append(f"{x}, {y}")
            try:
                suppress_remaining = (int(x) == x) and (int(y) == y)
            except ValueError:
                pass
        else:
            if x is not sentinel:
                fields.append(f"x={x}")
            if y is not sentinel:
                fields.append(f"y={y}")

        if not suppress_remaining:
            if self._polar == 2:
                fields.append(f"r={self.r}, theta={self.theta}")
                remaining_fields = ('r_squared',)
            elif self._polar:
                remaining_fields = ('r', 'theta', 'r_squared')
            else:
                remaining_fields = ('r_squared',)
            for attr in remaining_fields:
                try:
                    value = get(attr)
                    fields.append(f"{attr}={value}")
                except AttributeError:
                    pass

        text = ", ".join(fields)
        return f"{self.__class__.__name__}({text})"

    @classmethod
    def from_polar(cls, r, theta=_sentinel):
        theta_is_not_set = theta is _sentinel
        theta_is_set = not theta_is_not_set

        r_source = "r"
        theta_source = "theta"
        if isinstance(r, permitted_coordinate_types):
            if theta_is_not_set:
                raise TypeError(f"{cls.__name__}() missing 1 required positional argument: 'theta'")
        else:
            if isinstance(r, Vector2):
                if theta_is_set:
                    raise ValueError(f"{cls.__name__}.from_polar: when r is an object, you must not specify theta")
                return r
            while True:
                if hasattr(r, 'r') and hasattr(r, 'theta'):
                    if theta_is_set:
                        raise ValueError(f"{cls.__name__}.from_polar: when r is an object, you must not specify theta")
                    r_source = "r.r"
                    theta_source = "r.theta"
                    theta = r.theta
                    r = r.r
                    break
                if hasattr(r, "__getitem__"):
                    if theta_is_set:
                        raise ValueError(f"{cls.__name__}.from_polar: when r is an object, you must not specify theta")
                    try:
                        maybe_r = r['r']
                        theta = r['theta']
                        r_source = "r['r']"
                        theta_source = "r['theta']"
                        if len(r) != 2:
                            raise ValueError(f"{cls.__name__}: if r is a mapping, it must contain exactly 2 items")
                        r = maybe_r
                        break
                    except KeyError:
                        raise KeyError(f"{cls.__name__}: if r is an mapping, it must contain elements 'r' and 'theta'") from None
                    except TypeError:
                        pass

                if hasattr(r, "__iter__"):
                    if isinstance(r, _invalid_types_for_iterable):
                        raise TypeError(f"{cls.__name__}.from_polar: if r is an iterable, it must be ordered, not {r.__class__.__name__}")
                    if theta_is_set:
                        raise ValueError(f"{cls.__name__}.from_polar: when r is an object, you must not specify theta")
                    i = iter(r)
                    r_source = "r[0]"
                    theta_source = "r[1]"
                    try:
                        r = next(i)
                        theta = next(i)
                    except StopIteration:
                        raise ValueError(f"{cls.__name__}.from_polar: if r is an iterable, it must contain exactly 2 items")
                    try:
                        next(i)
                        raise ValueError(f"{cls.__name__}.from_polar: if r is an iterable, it must contain exactly 2 items")
                    except StopIteration:
                        pass
                    break
                break

        if not isinstance(r, permitted_coordinate_types):
            raise TypeError(f"{cls.__name__}.from_polar: {r_source} must be int or float, not {r}")

        if theta is None:
            if r:
                raise TypeError(f"{cls.__name__}.from_polar: {theta_source} is {theta}, must not be None when {r_source} != 0")
            return vector2_zero

        if not isinstance(theta, permitted_coordinate_types):
            raise TypeError(f"{cls.__name__}.from_polar: {theta_source} must be int, float, or None, not {theta}")
        # we already know theta is not None
        if not r:
            raise TypeError(f"{cls.__name__}.from_polar: {theta_source} is {theta}, must be None when {r_source}=0")

        r, theta = normalize_polar(r, theta)

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

    def __delattr__(self, name):
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

        # Compare the same coordinates used by __hash__.  Polar conversion
        # can round distinct coordinates to identical values (or vice versa),
        # so populating that cache must not change equality.
        return (self.x == other.x) and (self.y == other.y)

    def __ne__(self, other):
        if self is other:
            return False
        if not isinstance(other, Vector2):
            return True
        # Use Cartesian coordinates consistently with __eq__ and __hash__.
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

        return _vector2_from_xy(self.__class__, self.x + other.x, self.y + other.y)

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

        return _vector2_from_xy(self.__class__, self.x - other.x, self.y - other.y)

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
        elif self._cartesian:
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
        elif self._polar:
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
            raise ZeroDivisionError("Vector2 division by zero")
        if other == -1:
            return -self

        self_as_object = super(Vector2, self)
        get = self_as_object.__getattribute__

        # preserve all the fields we can, as cheaply as possible
        kwargs = {}

        if self._cartesian == 2:
            kwargs['x'] = self.x / other
            kwargs['y'] = self.y / other
        elif self._cartesian:
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
        elif self._polar:
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
        get = super(Vector2, self).__getattribute__
        kwargs = {}
        if self._cartesian == 2:
            kwargs['x'] = -self.x
            kwargs['y'] = -self.y
        elif self._cartesian:
            try:
                kwargs['x'] = -get('x')
            except AttributeError:
                pass
            try:
                kwargs['y'] = -get('y')
            except AttributeError:
                pass
        if self._polar == 2:
            kwargs['r'] = self.r
            kwargs['theta'] = normalize_angle(self.theta + pi)
        elif self._polar:
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
        return self.__class__(**kwargs)

    def __bool__(self):
        return self is not vector2_zero

    def __hash__(self):
        return self._hash

    ##
    ## methods
    ##

    def almost_equal(self, other, places):
        """
        Returns True if 'self' and 'other' have matching
        cartesian coordinates down to 'places' decimal places.
        """
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if (self is other) or (self == other):
            return True

        delta = self - other
        # print(f"{places} :: {a} - {b} = {delta=} <= {fraction=} = {delta <= fraction}")
        return (not round(delta.x, places)) and (not round(delta.y, places))

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
                y = r if self.y > 0 else -r
                if self.y == y:
                    return self
                return _vector2_from_xy(self.__class__, 0, y)

            if not self.y:
                x = r if self.x > 0 else -r
                if self.x == x:
                    return self
                return _vector2_from_xy(self.__class__, x, 0)

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
                return _vector2_from_xy(self.__class__, 0, 1 if self.y > 0 else -1)

            if not self.y:
                if self.x == 1:
                    return self
                return _vector2_from_xy(self.__class__, 1 if self.x > 0 else -1, 0)

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

        if self._polar:
            try:
                kwargs['r'] = get('r')
            except AttributeError:
                # if r is set, theta won't be set.
                # so we only check for theta if r isn't set.
                # _polar is 1 here, so theta must be cached when r isn't.
                kwargs['theta'] = normalize_angle(get('theta') + theta)

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
            raise TypeError(f"{self.__class__.__name__}: lerp ratio must be int or float")

        if self is other:
            return self
        if ratio == 0:
            return self
        if ratio == 1:
            return other

        return _vector2_from_xy(self.__class__,
            self.x + ((other.x - self.x) * ratio),
            self.y + ((other.y - self.y) * ratio),
            )

    # Geometric slerp, evaluated in a basis aligned with self.
    def vec_slerp(self, other, ratio, *, opposite='raise'):
        if not isinstance(other, Vector2):
            other = self.__class__(other)
        if not isinstance(ratio, (int, float)):
            raise TypeError(f"{self.__class__.__name__}: slerp ratio must be int or float")
        if opposite not in ('raise', 'clockwise', 'counter-clockwise'):
            raise ValueError("slerp opposite must be 'raise', 'clockwise', or 'counter-clockwise'")

        if self is other:
            return self
        if ratio == 0:
            return self
        if ratio == 1:
            return other
        if self is vector2_zero or other is vector2_zero:
            raise ValueError("can't slerp a zero vector: it has no direction")

        x, y = self.x, self.y
        other_x, other_y = other.x, other.y
        dot = x * other_x + y * other_y
        cross = x * other_y - y * other_x
        antiparallel = cross == 0 and dot < 0
        if cross == 0:
            if not antiparallel:
                return self.lerp(other, ratio)
            if opposite == 'raise':
                raise ValueError("opposite vectors have no unique slerp arc; select clockwise or counter-clockwise")
            theta = -pi if opposite == 'clockwise' else pi
        else:
            # atan2 retains small departures from pi that acos(dot) loses.
            theta = atan2(cross, dot)

        self_r, other_r = self.r, other.r
        angle = theta * ratio
        cosine, sine = cos(angle), sin(angle)
        if antiparallel:
            # The geometric formula is undefined here. The selected arc
            # rotates the direction while linearly interpolating the length.
            radius = self_r + (other_r - self_r) * ratio
            along, across = radius * cosine, radius * sine
        else:
            # Rewrite the usual two sine-weighted vectors in the basis
            # (u, perpendicular(u)), where u = self / |self|:
            # along = |self| cos(t theta)
            #       + (|other| - |self|) cot(theta) sin(t theta)
            # across = |other| sin(t theta).
            # Equal lengths now describe a circle without subtracting two
            # enormous, nearly cancelling weighted vectors near pi.
            along = self_r * cosine
            if other_r != self_r:
                along += (other_r - self_r) * (dot / cross) * sine
            across = other_r * sine

        unit_x, unit_y = x / self_r, y / self_r
        return _vector2_from_xy(self.__class__, unit_x * along - unit_y * across,
                              unit_y * along + unit_x * across)


    def pygame_slerp(self, other, ratio, *, reflected=None, epsilon=1e-6, opposite='raise'):
        "slerp calculator, more or less compatible with how pygame does it."

        if not isinstance(other, Vector2):
            other = Vector2(other)
        if not isinstance(ratio, (int, float)):
            raise TypeError("ratio must be float or int")
        if not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be float or int")

        if opposite not in ('raise', 'clockwise', 'counter-clockwise'):
            raise ValueError("slerp opposite must be 'raise', 'clockwise', or 'counter-clockwise'")

        if fabs(ratio) < epsilon:
            return self
        if fabs(ratio - 1) < epsilon:
            return other

        if (self.r < epsilon) or (other.r < epsilon):
            raise ValueError("can't use slerp with very small vectors (r < epsilon)")

        dot = self.normalized().dot(other.normalized())
        dot = -1 if (dot < -1) else (1 if (dot > 1) else dot)

        subtended_theta = acos(dot)

        # if reflected is None, behave identically to pygame.
        if reflected is None:
            if ratio < 0:
                reflected = True
                ratio = -ratio
            else:
                reflected = False

        # Explicit opposite-direction choices take precedence over reflected.
        # Preserve the default pygame behavior (including its epsilon test).
        if opposite != 'raise':
            cross = self.x * other.y - self.y * other.x
            if cross == 0 and self.dot(other) < 0:
                return self.vec_slerp(other, ratio, opposite=opposite)

        # If reflected is True, use the alternate arc.
        if reflected:
            subtended_theta -= tau

        if (self.x * other.y) < (self.y * other.x):
            subtended_theta = -subtended_theta
        subtended_theta = fabs(subtended_theta)

        # as subtended_theta converges on zero
        if ((subtended_theta < epsilon)
            or (fabs(subtended_theta - tau) < epsilon)):
            # approximate with lerp
            return self.lerp(other, ratio)
        # as subtended_theta converges on -pi
        if fabs(subtended_theta - pi) < epsilon:
            raise ValueError("slerp is undefined when subtended theta is near pi")

        base_ratio = ((other.r - self.r) * ratio + self.r) / sin(subtended_theta)
        self_ratio = (sin(subtended_theta * (1 - ratio)) / self.r) * base_ratio
        other_ratio = (sin(subtended_theta * ratio) / other.r) * base_ratio
        return Vector2(
            (self.x * self_ratio) + (other.x * other_ratio),
            (self.y * self_ratio) + (other.y * other_ratio),
            )


    slerp = pygame_slerp


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


# Our custom singleton vectors
# are hand-crafted by artisans
# to envelop you in luxury.
vector2_zero = vector2_0_0 = Vector2(5, 5)
object.__setattr__(vector2_zero, "x", 0)
object.__setattr__(vector2_zero, "y", 0)
object.__setattr__(vector2_zero, "theta", None)
object.__setattr__(vector2_zero, "r", 0)
object.__setattr__(vector2_zero, "r_squared", 0)
object.__setattr__(vector2_zero, "_cartesian", 2)
object.__setattr__(vector2_zero, "_polar", 2)
object.__setattr__(vector2_zero, "_hash", 0)


vector2_1_0 = Vector2(5, 5)
object.__setattr__(vector2_1_0, "x", 1)
object.__setattr__(vector2_1_0, "y", 0)
object.__setattr__(vector2_1_0, "theta", 0)
object.__setattr__(vector2_1_0, "r", 1)
object.__setattr__(vector2_1_0, "r_squared", 1)
object.__setattr__(vector2_1_0, "_cartesian", 2)
object.__setattr__(vector2_1_0, "_polar", 2)
object.__setattr__(vector2_1_0, "_hash", hash((1, 0)))


vector2_0_1 = Vector2(5, 5)
object.__setattr__(vector2_0_1, "x", 0)
object.__setattr__(vector2_0_1, "y", 1)
object.__setattr__(vector2_0_1, "theta", pi / 2)
object.__setattr__(vector2_0_1, "r", 1)
object.__setattr__(vector2_0_1, "r_squared", 1)
object.__setattr__(vector2_0_1, "_cartesian", 2)
object.__setattr__(vector2_0_1, "_polar", 2)
object.__setattr__(vector2_0_1, "_hash", hash((0, 1)))


vector2_1_1 = Vector2(5, 5)
object.__setattr__(vector2_1_1, "x", 1)
object.__setattr__(vector2_1_1, "y", 1)
object.__setattr__(vector2_1_1, "theta", pi / 4)
object.__setattr__(vector2_1_1, "r", sqrt(2))
object.__setattr__(vector2_1_1, "r_squared", 2)
object.__setattr__(vector2_1_1, "_cartesian", 2)
object.__setattr__(vector2_1_1, "_polar", 2)
object.__setattr__(vector2_1_1, "_hash", hash((1, 1)))


def _vector3_from_xyz(cls, x, y, z):
    # Fast specialized constructor for cartesian Vector3 objects.
    # Supports subclass construction and permitted user coordinate types.
    if (cls is not Vector3 or type(x) not in (int, float)
            or type(y) not in (int, float) or type(z) not in (int, float)):
        return cls(x, y, z)
    if not (x or y or z):
        return vector3_zero
    if x == 1 and not (y or z):
        return vector3_1_0_0
    if y == 1 and not (x or z):
        return vector3_0_1_0
    if z == 1 and not (x or y):
        return vector3_0_0_1
    result = object.__new__(cls)
    setattr = object.__setattr__
    setattr(result, 'x', x)
    setattr(result, 'y', y)
    setattr(result, 'z', z)
    return result


class Vector3:
    """Immutable Cartesian 3D vector with lazily cached lengths and hash."""

    __slots__ = ('x', 'y', 'z', 'r', 'r_squared', '_hash')

    def __new__(cls, x=_sentinel, y=_sentinel, z=_sentinel):
        if x is _sentinel and y is _sentinel and z is _sentinel:
            return vector3_zero
        if y is _sentinel and z is _sentinel and x is not _sentinel:
            if isinstance(x, Vector3):
                return x
            if not isinstance(x, permitted_coordinate_types):
                source = x
                if isinstance(source, Mapping):
                    if len(source) != 3:
                        raise ValueError("Vector3: a mapping must contain exactly three coordinates")
                    x, y, z = source['x'], source['y'], source['z']
                else:
                    try:
                        x, y, z = source.x, source.y, source.z
                    except AttributeError:
                        if isinstance(source, (Set, str, bytes, bytearray)):
                            raise TypeError("Vector3: coordinates must be an ordered numeric iterable")
                        x, y, z = source
        if x is _sentinel or y is _sentinel or z is _sentinel:
            raise ValueError("Vector3: specify x, y, and z together")
        if not (isinstance(x, permitted_coordinate_types)
                and isinstance(y, permitted_coordinate_types)
                and isinstance(z, permitted_coordinate_types)):
            raise TypeError("Vector3: coordinates must be int, float, or a permitted coordinate type")
        if not (x or y or z):
            return vector3_zero
        if x == 1 and not (y or z):
            return vector3_1_0_0
        if y == 1 and not (x or z):
            return vector3_0_1_0
        if z == 1 and not (x or y):
            return vector3_0_0_1
        self = object.__new__(cls)
        set_attr = object.__setattr__
        set_attr(self, 'x', x)
        set_attr(self, 'y', y)
        set_attr(self, 'z', z)
        return self

    def __getattr__(self, name):
        if name == 'r_squared':
            value = self.x*self.x + self.y*self.y + self.z*self.z
        elif name == 'r':
            value = sqrt(self.r_squared)
        elif name == '_hash':
            value = hash((self.x, self.y, self.z))
        else:
            raise AttributeError(f"{self.__class__.__name__}: object has no attribute {name}")
        object.__setattr__(self, name, value)
        return value

    def __setattr__(self, name, value):
        raise TypeError(f"{self.__class__.__name__}: object is immutable")

    def __delattr__(self, name):
        raise TypeError(f"{self.__class__.__name__}: object is immutable")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x!r}, {self.y!r}, {self.z!r})"

    def __len__(self):
        return 3

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index):
        index = index.__index__()
        if index not in (0, 1, 2):
            raise IndexError("Vector3: index out of range")
        return (self.x, self.y, self.z)[index]

    def __setitem__(self, index, value):
        raise TypeError("Vector3: object does not support item assignment")

    def __bool__(self):
        return self is not vector3_zero

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Vector3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other):
        return not self == other

    def __pos__(self):
        return self

    def __neg__(self):
        return _vector3_from_xyz(self.__class__, -self.x, -self.y, -self.z)

    def __add__(self, other):
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        if not other:
            return self
        if not self:
            return other
        return _vector3_from_xyz(self.__class__, self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        if not other:
            return self
        return _vector3_from_xyz(self.__class__, self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise ValueError("Vector3: multiply by a scalar, or use dot() or cross()")
        if scalar == 1:
            return self
        if scalar == 0:
            return vector3_zero
        return _vector3_from_xyz(self.__class__, self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise ValueError("Vector3: divide by a scalar")
        if scalar == 0:
            raise ZeroDivisionError("Vector3 division by zero")
        if scalar == 1:
            return self
        return _vector3_from_xyz(self.__class__, self.x / scalar, self.y / scalar, self.z / scalar)

    def almost_equal(self, other, places):
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        if self == other:
            return True
        return (not round(self.x - other.x, places)
                and not round(self.y - other.y, places)
                and not round(self.z - other.z, places))

    def scaled(self, multiplier):
        return self * multiplier

    def scaled_to_length(self, r):
        if not isinstance(r, (int, float)):
            raise TypeError("Vector3: length must be int or float")
        if not r:
            return vector3_zero
        if not self:
            raise ValueError("can't scale vector3_zero to a nonzero length")
        length = self.r
        if length == r:
            return self
        # Divide each coordinate first rather than forming a reciprocal.
        result = _vector3_from_xyz(self.__class__, (self.x / length) * r,
                                (self.y / length) * r,
                                (self.z / length) * r)
        # The requested length is already known. Future reads remain native
        # slot reads, just as they are after any lazy computation.
        if (result is not vector3_zero and result is not vector3_1_0_0
                and result is not vector3_0_1_0 and result is not vector3_0_0_1):
            object.__setattr__(result, 'r', abs(r))
            object.__setattr__(result, 'r_squared', r*r)
        return result

    def normalized(self):
        return self.scaled_to_length(1)

    def dot(self, other):
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other):
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        return _vector3_from_xyz(self.__class__, self.y*other.z - self.z*other.y,
                              self.z*other.x - self.x*other.z,
                              self.x*other.y - self.y*other.x)

    def rotated(self, axis, angle):
        """Rotate about a nonzero axis using the right-hand rule."""
        if not isinstance(angle, (int, float)):
            raise TypeError("Vector3: angle must be int or float")
        if not isinstance(axis, Vector3):
            axis = self.__class__(axis)
        if not axis:
            raise ValueError("Vector3: rotation axis must be nonzero")
        if not angle:
            return self
        if not self:
            raise ValueError("can't rotate vector3_zero")
        angle = normalize_angle(angle)
        if not angle:
            return self
        if angle == pi_over_two:
            cosine, sine = 0, 1
        elif angle == negative_pi_over_two:
            cosine, sine = 0, -1
        elif angle == negative_pi:
            cosine, sine = -1, 0
        else:
            cosine, sine = cos(angle), sin(angle)
        axis = axis.normalized()
        # Rodrigues' formula: v*cos(t) + (k cross v)*sin(t)
        #                     + k*(k dot v)*(1-cos(t)).
        cross = axis.cross(self)
        parallel = axis.dot(self) * (1 - cosine)
        return _vector3_from_xyz(self.__class__, self.x*cosine + cross.x*sine + axis.x*parallel,
                              self.y*cosine + cross.y*sine + axis.y*parallel,
                              self.z*cosine + cross.z*sine + axis.z*parallel)

    def lerp(self, other, ratio):
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        if not isinstance(ratio, (int, float)):
            raise TypeError("Vector3: lerp ratio must be int or float")
        if self is other or ratio == 0:
            return self
        if ratio == 1:
            return other
        return _vector3_from_xyz(self.__class__, self.x + (other.x - self.x)*ratio,
                              self.y + (other.y - self.y)*ratio,
                              self.z + (other.z - self.z)*ratio)

    def nlerp(self, other, ratio):
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        if not isinstance(ratio, (int, float)):
            raise TypeError("Vector3: nlerp ratio must be int or float")
        if self is other or ratio == 0:
            return self
        if ratio == 1:
            return other
        result = self.lerp(other, ratio)
        return result.scaled_to_length(self.r + (other.r - self.r)*ratio)

    def slerp(self, other, ratio, *, axis=None):
        """Interpolate direction on the shorter arc and length linearly.

        Exactly opposite directions need a perpendicular axis to select the
        half-turn. Its sign selects the direction by the right-hand rule.
        """
        if not isinstance(other, Vector3):
            other = self.__class__(other)
        if not isinstance(ratio, (int, float)):
            raise TypeError("Vector3: slerp ratio must be int or float")
        if axis is not None:
            if not isinstance(axis, Vector3):
                axis = self.__class__(axis)
            if not axis:
                raise ValueError("Vector3: interpolation axis must be nonzero")
        if self is other or ratio == 0:
            return self
        if ratio == 1:
            return other
        if not self or not other:
            raise ValueError("can't slerp a zero vector: it has no direction")

        normal = self.cross(other)
        dot = self.dot(other)
        if not normal and dot >= 0:
            return self.lerp(other, ratio)
        self_r, other_r = self.r, other.r
        unit = self / self_r
        if not normal:
            if axis is None:
                raise ValueError("opposite vectors require a perpendicular interpolation axis")
            axis = axis.normalized()
            # Allow only floating-point rounding in a perpendicular axis.
            if fabs(axis.dot(unit)) > 1e-12:
                raise ValueError("interpolation axis must be perpendicular to the vectors")
            tangent = axis.cross(unit).normalized()
            theta = pi
        else:
            tangent = normal.normalized().cross(unit).normalized()
            theta = atan2(normal.r, dot)
        angle = theta * ratio
        cosine, sine = cos(angle), sin(angle)
        radius = self_r + (other_r - self_r)*ratio
        return _vector3_from_xyz(self.__class__, (unit.x*cosine + tangent.x*sine)*radius,
                              (unit.y*cosine + tangent.y*sine)*radius,
                              (unit.z*cosine + tangent.z*sine)*radius)


# Bootstrap the predefined objects without invoking the interning constructor.
for _name, (_x, _y, _z) in (
        ('vector3_zero', (0, 0, 0)),
        ('vector3_1_0_0', (1, 0, 0)),
        ('vector3_0_1_0', (0, 1, 0)),
        ('vector3_0_0_1', (0, 0, 1))):
    _vector = object.__new__(Vector3)
    for _field, _value in (('x', _x), ('y', _y), ('z', _z),
                           ('r', 0 if _name == 'vector3_zero' else 1),
                           ('r_squared', _x*_x + _y*_y + _z*_z),
                           ('_hash', hash((_x, _y, _z)))):
        object.__setattr__(_vector, _field, _value)
    globals()[_name] = _vector
vector3_0_0_0 = vector3_zero
del _name, _x, _y, _z, _vector, _field, _value
