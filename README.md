# vec

## Reasonable, performant 2D and 3D vector objects for games

##### Copyright 2019-2024 by Larry Hastings

## Overview

`vec` publishes two vector classes designed for game development:
`Vector2` for 2D and `Vector3` for 3D.

Features:

* Both vector classes are immutable.
* Both support the usual vector object features,
  including operator overloading.
* Derived attributes are lazily computed and cached where possible.
* `Vector2` supports both cartesian and polar coordinates.
* `Vector3` uses cartesian coordinates, with vector cross products and
  rotation around an arbitrary axis.

`vec` supports Python 3.6+, and passes its unit test suite with 100% statement coverage.


## Why Another Vector Class?

I've participated in four PyWeek gaming challenges.  And twice,
mid-week, I wrote my own vector class out of sheer frustration.

The biggest problem with most Python vector objects in is that they're
*mutable.* Frankly this way lies madness.  Vector objects should be
immutable--it just makes sense from an API perspective.  What if you
set the position of some game-engine pawn to be a particular vector
object, then modify that vector object?  Should the pawn update its
position automatically--and if so, how would it know the value changed?

Similarly, some vector classes use degrees for polar coordinates
instead of radians.
Again this way lies madness.  The trigonometric functions in Python's
`math` module operate in the radians domain, and having to keep track
of which domain something is in--and translate back and forth--is
a needless conceptual complication.  You've got a game to write!

(Some vector classes support *both* radians *and* degrees for polar
coordinates.  This is bad API design--it doubles the surface
area of your API, adding needless complexity and increasing maintenance
and testing overhead.  Embrace the radian, folks.)

On a related note, many vector classes make polar coordinates
second-class citizens.  Most vector classes only store vectors in
cartesian coordinates, so either the programmer must perform all polar
operations externally to the vector objects, or they incur the
overhead and cumulative error of translating to polar and
back again with every operation.

vec's `Vector2` avoids all these problems:

* `Vector2` objects are immutable,
* they make polar and cartesian coordinates both first-class citizens,
  and
* they strictly use radians for polar coordinates.

## The Conceptual Model

`Vector2` objects conceptually represent a vector.  They can be
defined using either cartesian or polar coordinates, and any `Vector2`
can be queried for both its cartesian and polar coordinates.

Most vector objects in games are defined using cartesian coordinates.
`Vector2` makes that easy, supporting any number of invocations
to create one.  Discrete parameters,
iterables, and objects that support `x` and `y` attributes all work fine:

    Vector2(0, 1)
    Vector2(x=0, y=1)
    Vector2((0, 1))
    Vector2([0, 1])
    Vector2(iter([0, 1]))
    Vector2({'x':0, 'y':1})
    Vector2(types.SimpleNamespace(x=0, y=1))

All these define the same vector.  That last example is there to demonstrate
that `Vector2` can create a vector based on any object with `x` and `y`
attributes.

Every `Vector2` object supports both cartesian and polar coordinates.
You can define a `Vector2` using cartesian coordinates, then examine
its polar coordinates.  This:

    v = vec.Vector2(0, 1)
    print(v.theta, v.r)

prints `1.5707963267948966 1`.  That first number is π/2 (approximately).

Conversely, you can define a `Vector2` object using polar
coordinates, then examine its cartesian coordinates:

    v2 = vec.Vector2(r=1, theta=math.pi/2)
    print(v2.x, v2.y)

This prints `0 1`.  `Vector2` has special-cased some specific
common vectors, like this one.  But most of the time, conversions
between polar and cartesian coordinates are subject to
floating-point rounding.

### Implementation Details

To define a valid `Vector2` object, you must have a complete
set of either cartesian or polar coordinates--either is sufficient.
All other attributes will be lazily computed on demand.

`Vector2` objects use slots, and rely on `__getattr__`
to implement this lazy computation.  Only the known values of the
vector are set when it's created.  If the user refers to an attribute
that hasn't been computed yet, Python will call `Vector2.__getattr__()`,
which computes, caches, and returns that value.  Future references to that
attribute skip this mechanism and simply return the cached value, which
is only as expensive as an attribute lookup on a conventional object.

Operations on `Vector2` objects compute their result
using the cheapest approach.  If you have a `Vector2` object
defined using polar coordinates, and you call `.rotated()` or `.scaled()`
on it, all the math is done in the polar domain.  On the other
hand, adding vectors is *almost* always done in the cartesian domain,
so if you add a polar vector to any other vector, its cartesian
coordinates will likely be computed--and the resulting vector will always
be defined using cartesian coordinates.

What's the exception?  There's a special case for adding two polar vectors
which have the exact same `theta`: just add their `r` values.
That approach is much cheaper than converting to cartesian,
and more precise as well, returning a vector defined using polar
coordinates!  `Vector2` takes advantage of many such serendipities,
performing your vector math as cheaply and accurately as possible.


## The Vector2 API

`Vector2(x=..., y=..., *, r=..., theta=..., r_squared=...)`

<dl><dd>

Constructs a `Vector2` object.  All five arguments are optional.
Calling `Vector2()` with no arguments returns the zero vector.
Otherwise, when supplying individual coordinates, you *must*
pass in *either* both `x` and `y` *or* both `r` and `theta`;
additional attributes are optional.
Any attributes not passed in at construction time will be lazily
computed when they are evaluated.

You can also pass in a single object which will initialize the
vector.  Supported objects include:

* an existing `Vector2` object (just returns that object),
* an object which has `.x` and `.y` attributes,
* a mapping object with exactly two keys, `'x'` and `'y'`, and
* an ordered iterable object with exactly two elements, which
  will be used as `'x'` and `'y'` respectively.

`Vector2` only does *some* validation of its arguments.
This is intentional; we assume you're using sane values,
not giant numbers (4e200) or special values (NaN, inf).
It ensures that `r` is non-negative, and `theta` is normalized
to `-pi <= theta < pi`.  However, it doesn't check that
`(x, y)` and `(r, theta)` describe the same vector.
If you pass in `x` and `y`, and also pass in a `theta` and `r`
that don't match, you'll get back the `Vector2` that you
asked for.  Good luck!

</dd></dl>

### Attributes

`Vector2` objects support five attributes:
`x`, `y`, `r`, `theta`, and `r_squared`.  It doesn't matter
whether the object was defined with cartesian or polar
coordinates, they'll all work.

`r_squared` is equivalent to `r*r`.  But if you have
a `Vector2` object defined with cartesian coordinates,
it's much cheaper to compute `r_squared` than `r`.
And there are many use cases where `r_squared` works
just as well as `r`.

For example, consider collision detection in a game.  One
way to decide whether two objects are colliding is to measure
the distance between them--if it's less than a certain distance
**R**, the two objects are colliding.  But computing the
actual distance is expensive--it requires a time-consuming
square root.  It's much cheaper to compute the distance-squared
between the two points.  If that's less than **R<sup>2</sup>**, the two
objects are colliding.

### Operators and protocols

`Vector2` objects support the *iterator protocol.*
`len()` on a `Vector2` object will always return 2.
You can also iterate over a `Vector2` object,
which will yield the `x` and `y` attributes in that order.

`Vector2` objects support the *sequence protocol.*  You can subscript
them, which behaves as if the `Vector2` object is a tuple of length
2 containing the `x` and `y` attributes.

`Vector2` objects support the *boolean protocol;* you may use them
with boolean operators, and you may call `bool()` on them.  When used in
a boolean context, the zero vector evaluates to `False`, and all other
vectors evaluate to `True`.

`Vector2` objects are *hashable,* but they're not *ordered*.
(You can't ask if one vector is *less than* another.)

`Vector2` objects support the following operators:

* `v1 + v2` adds the two vectors together.
* `v1 - v2` subtracts the right vector from the left vector.
* `v1 * scalar` multiplies the vector by a scalar amount, equivalent to `v1.scaled(scalar)`.
* `v1 / scalar` divides the vector by a scalar amount.  A zero divisor raises
  `ZeroDivisionError`, including when `v1` is the zero vector.
* `+v1` is exactly the same as `v1`.
* `-v1` returns the opposite of `v1`, such that `v1 + (-v1)` should be the zero vector.
   (This may not always be the case due to floating-point imprecision.)
* `v1 == v2` is `True` if the two vectors are *exactly* the same, and `False` otherwise.
  For consistency, this only compares cartesian coordinates.
  Note that floating-point imprecision may result in two vectors that *should* be the
  same failing an `==` check.  Consider using the `almost_equal` method, which allows
  for some imprecision in its comparison.
* `v1 != v2` is `False` if the two vectors are *exactly* the same, and `True` otherwise.
  For consistency, this only compares cartesian coordinates.
  Note that floating-point imprecision may result in two vectors that *should* be the
  same passing an `!=` check.  Again, consider using the `almost_equal` method and
  negating the results.
* `v[0]` and `v.x` evaluate to the same number.
* `v[1]` and `v.y` evaluate to the same number.
* `list(v)` is the same as `[v.x, v.y]`.

`Vector2` always uses Cartesian coordinates for equality testing
and hashing, for consistency.

### Class methods

`Vector2.from_polar(r, theta)`

<dl><dd>

Constructs a `Vector2` object from the two polar coordinates
`r` and `theta`.

You can also pass in a single object which will be used to
initialize the vector.  Supported objects include:

* an existing `Vector2` object (just returns that object),
* an object which has `.r` and `.theta` attributes,
* a mapping object with exactly two keys, `'r'` and `'theta'`, and
* an ordered iterable object with exactly two elements, which will
  be interpreted as `'r'` and `'theta'` in that order.

If `r` is `0`, `theta` must be `None`, and `from_polar` will
return the zero vector.  If `r` is not `0`, `theta` must not
be `None`.
</dd></dl>

### Methods

`Vector2` objects support the following methods:

`Vector2.almost_equal(other, places)`

<dl><dd>

Returns `True` if the vector and `other` are the same vector,
down to `places` decimal places.  Like the `Vector2` class's
support for the `==` operator, the comparison is only done
using cartesian coordinates, for consistency.
</dd></dl>

`Vector2.scaled(scalar)`

<dl><dd>

Returns a new `Vector2` object, equivalent to the original vector multiplied by that scalar.
</dd></dl>

`Vector2.scaled_to_length(r)`

<dl><dd>

Returns a new `Vector2` object, equivalent to the original vector with its length set to `r`.
If `r` is negative, `scaled_to_length` reverses the direction and sets the length to
`abs(r)`.  A zero `r` returns the zero vector.  Scaling the zero vector to a nonzero
length raises `ValueError`.
</dd></dl>

`Vector2.normalized()`

<dl><dd>

Returns a new `Vector2` object, equivalent to the original vector scaled to length 1.

The zero vector doesn't have a direction, so an attempt to normalize it raises `ValueError`.
</dd></dl>

`Vector2.rotated(theta)`

<dl><dd>

Returns a new `Vector2` object, equal to the original vector rotated by `theta` radians.
</dd></dl>

`Vector2.dot(other)`

<dl><dd>

Returns the "dot product" `self` • `other`.  This result is a scalar value, not a vector.
</dd></dl>

`Vector2.cross(other)`

<dl><dd><p>

Returns the "cross product" `self` ⨯ `other`.  This result is a scalar value, not a vector.

</p><p>

*Note:* technically, there is no "cross product" defined for 2-dimensional vectors.
This actually returns the "perpendicular dot product" of the two vectors,
because that's what people generally mean when they ask for the "cross product"
of two 2D vectors.

</p></dd></dl>

`Vector2.polar()`

<dl><dd>

Returns a 2-tuple of `(self.r, self.theta)`.
</dd></dl>

`Vector2.lerp(other, ratio)`

<dl><dd>

Returns a vector representing a linear interpolation between `self` and `other`, according
to the scalar ratio `ratio`.  `ratio` should be a value between (and including) `0` and `1`.
If `ratio` is `0`, this returns `self`.  If `ratio` is `1`, this returns `other`.
If `ratio` is between `0` and `1` non-inclusive, this returns a point on the line segment
defined by the two endpoints `self` and `other`, with the point being `ratio` between `self`
and `other`.  For example, if `ratio` is `0.4`, this returns `(self * 0.6) + (other * 0.4)`.

Note that it's not an error to specify a `ratio` less than `0` or greater than `1`, and
`ratio` isn't clamped to this range.
</dd></dl>

`Vector2.slerp(other, ratio, *, reflected=None, epsilon=1e-6, opposite='raise')`

<dl><dd>

Returns a vector representing a spherical interpolation between `self` and `other`, according
to the scalar ratio `ratio`.  `ratio` should be a value between (and including) `0` and `1`.
If `ratio` is `0`, this returns `self`.  If `ratio` is `1`, this returns `other`.

Note that it's not an error to specify a `ratio` less than `0` or greater than `1`, and
`ratio` isn't clamped to this range.

To facilitate compatibility with [pygame](https://www.pygame.org/news),
by default `slerp` produces identical results to
[`pygame.math.Vector2.slerp`](https://www.pygame.org/docs/ref/math.html#pygame.math.Vector2.slerp).
This includes the following behavior: when `ratio` is < `0`, `slerp` negates
`ratio` and traces an alternate path between `self` and `other`.  This makes it
impossible to use actual negative values for `ratio`.

If you want to use actual negative ratios, specify the `reflected` parameter.
This both specifies which path to trace and disables reflecting `ratio` around `0`.
`reflected=False` selects the path normally traced by positive ratios, and
`reflected=True` selects the path normally traced by negative ratios.


For exactly opposite directions, `opposite` accepts `'raise'` (the default),
`'clockwise'`, or `'counter-clockwise'`.  The direction names use the mathematical
convention with positive Y up.  An explicit direction overrides `reflected`
for exactly opposite vectors and interpolates their lengths linearly.
Negative ratios are still negated when `reflected=None`; specify a boolean
`reflected` value to use actual negative ratios.

The default preserves pygame's rejection of angles within `epsilon` of pi.
A direction choice resolves exactly opposite vectors (zero Cartesian cross
product and negative dot product); it does not override this rejection for
merely near-opposite vectors.  Invalid `opposite` settings raise `ValueError`,
including at the endpoints.

The original geometric interpolation is available as `Vector2.vec_slerp`,
documented below.  `Vector2.slerp` remains an alias of `Vector2.pygame_slerp`.

</dd></dl>


`Vector2.vec_slerp(other, ratio, *, opposite='raise')`

<dl><dd>

This is vec's original geometric interpolation, with a stabilized calculation.
It does not accept `reflected` or `epsilon`.

Returns a vector representing a spherical interpolation between `self` and `other`, according
to the scalar ratio `ratio`.  `ratio` should be a value between (and including) `0` and `1`.
If `ratio` is `0`, this returns `self`.  If `ratio` is `1`, this returns `other`.

If `self` and `other` are exactly opposing vectors,
there are two possible valid interpolations, and `Vector2` doesn't know
which one you want.  Choose the result you prefer by specifying the
keyword-only argument `opposite`:

* `'raise'` (default): raise `ValueError` because there is no unique arc.
* `'clockwise'`: follow the arc with decreasing polar angle.
* `'counter-clockwise'`: follow the arc with increasing polar angle.

For example, `Vector2(1, 0).vec_slerp(Vector2(-1, 0), 0.5,
opposite='counter-clockwise')` returns approximately `(0, 1)`.
These direction names use the mathematical convention with positive Y up.
The opposite-direction case is detected by a zero Cartesian cross product
and a negative dot product, without applying an angular tolerance.
Nearly opposite vectors follow their shorter arc regardless of the setting.
Invalid settings raise `ValueError`, including at the endpoints.

For a selected opposite-direction arc, length is interpolated linearly.
Elsewhere, `vec_slerp` retains geometric sine weighting: equal-length vectors
trace a circle, but unequal-length vectors do **not** generally interpolate
length linearly and can produce large lengths near opposite directions.

The endpoint shortcuts apply even to opposite or zero vectors.  Interpolating
between a zero vector and a nonzero vector at other ratios raises `ValueError`;
interpolating the zero vector with itself returns that same zero vector.

Note that it's not an error to specify a `ratio` less than `0` or greater than `1`, and
`ratio` is not clamped to this range.
</dd></dl>


`Vector2.nlerp(other, ratio)`

<dl><dd>

Returns a vector representing a normalized linear interpolation between `self` and `other`,
according to the scalar ratio `ratio`.  `ratio` should be a value between (and including)
`0` and `1`.  If `ratio` is `0`, this returns `self`.  If `ratio` is `1`, this returns `other`.

Note that it's not an error to specify a `ratio` less than `0` or greater than `1`, and
`ratio` is not clamped to this range.
</dd></dl>


### Constants

`vector2_zero`

<dl><dd>

The "zero" `Vector2` vector object.
`vec` guarantees that every zero vector is a reference to this object:

    >>> v = vec.Vector2(0, 0)
    >>> v is vec.vector2_zero
    True

Mathematically-speaking, the zero vector when expressed in polar coordinates
doesn't have a defined angle.  Therefore `vec` defines its zero vector as
having an angle of `None`.  The zero vector must have `r` set to zero
and `theta` set to `None`, and any other vector must have a non-zero `r`
and `theta` set to a numeric value.
</dd></dl>

`vector2_1_0`

<dl><dd>

A predefined `Vector2` vector object, equivalent to `Vector2(1, 0)`.
When constructing a `Vector2` object that is exactly equivalent to this
vector, the `Vector2` constructor will always return a reference to this
vector:

    >>> v = vec.Vector2(1, 0)
    >>> v is vec.vector2_1_0
    True
    >>> v2 = vec.Vector2(r=1, theta=0)
    >>> v2 is vec.vector2_1_0
    True

</dd></dl>

`vector2_0_1`

<dl><dd>

A predefined `Vector2` vector object, equivalent to `Vector2(0, 1)`.
When constructing a `Vector2` object that is exactly equivalent to this
vector, the `Vector2` constructor will always return a reference to this
vector:

    >>> v = vec.Vector2(0, 1)
    >>> v is vec.vector2_0_1
    True
    >>> v2 = vec.Vector2(r=1, theta=pi/2)
    >>> v2 is vec.vector2_0_1
    True

</dd></dl>

`vector2_1_1`

<dl><dd>

A predefined `Vector2` vector object, equivalent to `Vector2(1, 1)`.
When constructing a `Vector2` object that is exactly equivalent to this
vector, the `Vector2` constructor will always return a reference to this
vector:

    >>> v = vec.Vector2(1, 1)
    >>> v is vec.vector2_1_1
    True
    >>> v2 = vec.Vector2(r=2 ** 0.5, theta=pi/4)
    >>> v2 is vec.vector2_1_1
    True

</dd></dl>


## The Vector3 API

`Vector3` has the same immutable, cached-attribute design as `Vector2`, with
three cartesian coordinates.  It does not have polar or spherical coordinate
attributes.

### Construction and attributes

```python
from vec import Vector3

v = Vector3(2, 3, 6)
v = Vector3(x=2, y=3, z=6)
v = Vector3((2, 3, 6))
v = Vector3({'x': 2, 'y': 3, 'z': 6})
```

You can supply three individual coordinates, an ordered iterable of exactly
three coordinates, a mapping with exactly the keys `x`, `y`, and `z`, or an
object with those three attributes.  Passing an existing `Vector3` returns
that same object.  Calling `Vector3()` returns the zero vector.  Individual
coordinates must be supplied together; omitted coordinates do not default
to zero, and `None` is not a coordinate value.

`x`, `y`, and `z` are stored at construction.  `r` (length), `r_squared`,
and the hash are computed when needed and cached.  Repeated reads of a cached
attribute access its slot directly.  Assignment and deletion of attributes
are forbidden.  Equality and hashing depend only on cartesian coordinates.

### Operators and common methods

The arithmetic operators follow `Vector2`'s conventions: vector addition and
subtraction, scalar multiplication and division, unary `+` and `-`, and exact
equality.  Division by zero raises `ZeroDivisionError`, even for the zero
vector.  Addition and subtraction also accept objects supported by the
constructor.  Vectors are hashable but not ordered.

Iteration yields `x`, `y`, `z`, and `len(v)` is 3.  Indexing accepts `0`, `1`,
or `2`; as with `Vector2`, negative indices and slices are not supported.
Only the zero vector is false in a boolean context.

* `v.almost_equal(other, places)` compares cartesian coordinate differences
  rounded to the given number of decimal places.
* `v.scaled(scalar)` is equivalent to `v * scalar`.
* `v.scaled_to_length(r)` sets the length to `abs(r)` and reverses direction
  if `r` is negative.  A zero length returns the zero vector; scaling the zero
  vector to a nonzero length raises `ValueError`.
* `v.normalized()` scales to length 1.  Normalizing zero raises `ValueError`.
* `v.dot(other)` returns a scalar dot product.
* `v.cross(other)` returns a `Vector3` perpendicular to both inputs, with
  direction determined by the right-hand rule.  For example,
  `Vector3(1, 0, 0).cross(Vector3(0, 1, 0))` is `Vector3(0, 0, 1)`.

Methods may return the original vector or a predefined vector when the
result permits it; a new allocation is not guaranteed.

### Rotation

`v.rotated(axis, angle)` rotates around a nonzero axis by `angle` radians,
following the right-hand rule.  The axis can be any object accepted by the
`Vector3` constructor; its length does not affect the rotation.  Negating
the axis reverses the direction.

```python
from math import pi

Vector3(1, 0, 0).rotated((0, 0, 1), pi/2)  # Vector3(0, 1, 0)
```

A zero axis raises `ValueError`, even when the angle is zero.  As with
`Vector2`, rotating the zero vector by a nonzero angle raises `ValueError`;
a zero angle returns the original vector.

### Interpolation

`v.lerp(other, ratio)` interpolates coordinates linearly.
`v.nlerp(other, ratio)` follows the direction of that linear interpolation
while interpolating length linearly.  If that direction is zero but the
requested length is nonzero, `nlerp` raises `ValueError`; for example, this
happens halfway between equal-length opposite vectors.

`v.slerp(other, ratio, *, axis=None)` interpolates direction along the shorter
arc and length linearly.  Unlike `Vector2`'s pygame-compatible default, it
uses negative ratios directly for extrapolation.  It has no `reflected` or
`epsilon` arguments.

Exactly opposite vectors have no unique rotation plane.  In that case,
`slerp` raises `ValueError` unless you supply a nonzero perpendicular `axis`.
The axis's sign selects the half-turn using the right-hand rule; its length
is ignored.

```python
a, b = Vector3(1, 0, 0), Vector3(-1, 0, 0)
a.slerp(b, 0.5, axis=(0, 0, 1))   # approximately Vector3(0, 1, 0)
a.slerp(b, 0.5, axis=(0, 0, -1))  # approximately Vector3(0, -1, 0)
```

Opposite directions are detected using a zero cross product and a negative
dot product, without an angular tolerance.  Perpendicularity of a supplied
axis allows a normalized dot product of at most `1e-12` in magnitude to
accommodate floating-point rounding.  Non-opposite vectors determine their
own plane and ignore the supplied axis's direction.  A supplied zero axis
is always rejected, including at endpoints.

All three interpolation methods accept other objects supported by the
constructor.  A ratio of 0 returns `self`, and 1 returns `other` (converted
to `Vector3` if necessary).  Ratios are not clamped to `[0, 1]`.  `slerp`
between zero and a nonzero vector raises `ValueError` away from the endpoints;
interpolating zero with itself returns zero.

### Predefined vectors

`vector3_zero` (also named `vector3_0_0_0`), `vector3_1_0_0`,
`vector3_0_1_0`, and `vector3_0_0_1` represent the zero vector and the three
positive unit axes.  Construction returns these existing objects whenever
the coordinates match exactly.  Their lengths, squared lengths, and hashes
are already cached.


## Extending vec to handle other types

`vec` does some input verification on its inputs.
Coordinates--`x`, `y`, `z`, `r`, `theta`, as applicable--are required to be
either `int` or `float`.
(Technically `theta` can also be `None`.)  This best serves
the intended use case of `vec` as a 2D vector library for
game programming in Python.  `permit_coordinate_type` applies to both vector
classes; accepting a type does not guarantee that it supports every operation
(such as trigonometry or taking a square root).

If you want to experiment with `vec` for other use cases,
you may want `vec` to permit other types to be valid
coordinates.  `vec` provides a simple mechanism to allow
this.  Simply call:

```
    vec.permit_coordinate_type(T)
```

before creating your vector, passing in the type you want
to use as a coordinate as `T`, and `vec` will now accept
objects of that type as coordinates.

Note that the types you extend `vec` with in this manner
should behave like numeric types, like `int` and `float`.


## Numerical limits

`vec` is intended for use in video games, with ordinary game coordinates.
That means it's concerned with both speed and correctness.  So, `vec`
works hard to be fast, and it produces correct results... given reasonable
inputs.  But it doesn't guard every operation against overflow or underflow;
nor does `vec` clamp coordinates to enforce a fixed minimum or maximum.
All that would just slow it down, adding training wheels that a well-written
game doesn't need or want.  So it's up to you to ensure that intermediate values,
including squared coordinates, squared lengths, and squared scale factors,
all remain within a usable floating-point range.

Specifically: don't expect magnitudes such as `4e200` or `4e-200` to work
reliably.  Extreme values can produce zero, infinity, NaN, or exceptions;
results can also depend on which derived attributes have already been cached.
For example, dividing by `1e-200` can raise `ZeroDivisionError` when a
cached squared length is divided by the underflowed square of that nonzero
scalar.

## Changelog

**next version** *under development*

* Use specialized Cartesian result construction throughout `Vector3`
  arithmetic, cross products, scaling, rotation, and interpolation, preserving
  predefined vectors and constructor behavior for subclasses and custom types.
* Speed up Cartesian result construction in `Vector2` addition, subtraction,
  `lerp`, `vec_slerp`, and axis-aligned scaling and normalization. Preserve
  predefined-vector reuse and normal construction for subclasses and custom
  coordinate types.
* Replace `Vector2`'s construction metaclass with `__new__`, preserving
  input validation, predefined-vector reuse, and lazy attribute caching.
* Add `Vector3`: immutable cartesian coordinates, cached lengths and hash,
  arithmetic, normalization, dot and vector cross products, axis-based
  rotation, and linear, normalized-linear, and spherical interpolation.
  Opposite-direction spherical interpolation accepts a perpendicular axis
  to select the half-turn.  Add predefined zero and unit-axis vectors, tests,
  and API documentation.
* Breaking change: dividing a vector by zero now raises `ZeroDivisionError`
  instead of returning the zero vector.

* 🤖 ChatGPT Astra 6 XHigh did a review of the existing code,
  and added a heaping pile of optimizations, correctness fixes, and new tests:

    * Add the keyword-only `slerp` and `vec_slerp` setting `opposite`, accepting `'raise'` (the
      default), `'clockwise'`, or `'counter-clockwise'`.  Stabilize `vec_slerp` interpolation
      near opposite directions using `atan2` and an orthogonal basis, while
      preserving geometric weighting for unequal-length, non-opposite vectors.
    * Fix scaling a negative-axis vector to a negative length equal to its
      nonzero coordinate: the shortcut incorrectly returned the original vector
      instead of reversing it.  Add normalization and interpolation regressions
      for axes, quadrants, endpoints, and zero vectors.
    * Simplify construction with direct slot initialization.  Use existing cache
      counters to avoid unnecessary missing-attribute exceptions during scaling,
      negation, rotation, and `repr`, preserving lazy caching and immutability.
    * Document numerical limits for extreme magnitudes and intermediate squares.
    * Fix angle normalization hanging on large finite angles and infinities.
      Use remainder reduction, reject infinite angles with `ValueError`, and
      add regressions for boundary values, large angles, and rotation.
    * Fix `==` and `!=` to consistently compare Cartesian coordinates, as
      documented.  Populating the polar cache could previously change equality,
      causing inconsistent hashes for equal vectors or failed dictionary lookups.
    * Reject attribute deletion, just as attribute assignment is rejected, to
      preserve immutability and protect cached values and predefined vectors.
    * Add regression tests for equality, hashing, deletion, raw cache inspection,
      and cache population in different attribute-read orders.


**0.7.1** *2024/09/15*

* Added the `reflected` parameter to `Vector2.slerp`,
  allowing you to render actual negative ratios.
  For compatibility with
  [`pygame.math.Vector2.slerp`](https://www.pygame.org/docs/ref/math.html#pygame.math.Vector2.slerp),
  negative ratios normally trace an alternate path
  using a symmetric unit circle.  Specifying a boolean
  value for `reflected` disables this behavior, letting
  you explicitly select which path you want.

**0.7** *2024/09/15*

* Breaking change: Retooled support for `slerp`
  (spherical linear interpolation)
  to increase cross-library compatibility:
  *  `vec`'s old implementation has been renamed `vec_slerp`.
  * There's a new method, `pygame_slerp`, which produces
    essentially identical results to [`pygame.math.Vector2.slerp`](https://www.pygame.org/docs/ref/math.html#pygame.math.Vector2.slerp).
  * The `Vector2.slerp` method defaults to `pygame_slerp`.
  * `vec_slerp` now raises `TypeError` (instead of `ValueError`)
    if `ratio` is the wrong type.  (Makes sense, right?)
* Change the `repr` for `Vector2` objects.
  If both `x` and `y` are set, and they're both
  integers, it's more pleasant to see only those values,
  even if some/all of the other values are cached.
  This makes `repr(vector2_zero)` a lot easier to read.
* Updated copyright dates to 2024.

**0.6.3** *2023/10/26*

* Added three new predefined vectors:

  * `vector2_0_1` is `Vector2(0, 1)`
  * `vector2_1_0` is `Vector2(1, 0)`
  * `vector2_1_1` is `Vector2(1, 1)`

  Any expression that results in a vector that would be exactly
  equal to one of these vectors is guaranteed to return the
  predefined vector.  `Vector2(1, 0) is vector2_1_0` evaluates
  to `True`.

**0.6.2** *2023/06/14*

* Added `Vector2.almost_equal`, which supports testing
  for slightly-inexact equality.

**0.6.1** *2023/06/14*

* Enhanced the `Vector2` constructor: now it also accepts
  mappings.  The mapping must have exactly two elements,
  `x` and `y`.
* Enhanced `Vector2.from_polar`.  It now accepts all the same
  stuff as the `Vector2` constructor: `Vector2` objects,
  namespaces, mappings, and iterables.  Where it examines
  names (attributes, keys) it naturally uses `r` and `theta`
  instead of `x` and `y`.

**0.6** *2023/06/14*

A major improvement!

* `vec` now has a proper test suite.
* `vec` now passes its test suite with 100% coverage.
* `vec` explicitly supports Python 3.6+.
* Added more shortcut optimizations, e.g. rotating a cartesian vector by a multiple of `pi/2`.
* Tightened up the metaclass `__call__` logic a great deal.
* Implemented `Vector2.slerp`, and added `Vector2.nlerp`.
* Allowed `vec.permit_coordinate_type`, to allow extending the
  set of permissible types for coordinates.
* Internal details:

    - Now cache `_cartesian` and `_hash` internally, as well as `_polar`.
      (A vector can have a complete set of both cartesian and polar coordinates,
      so it's nice to know everything that's available--that can make some
      operations faster.)

* Bugfix: `Vector2.dot()` was simply wrong, it was adding where it should
  have been multiplying.  Fixes #3.

**0.5** *2021/03/21*

Initial version.
