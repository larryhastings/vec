# vec

## A reasonable, performant 2D vector object for games

##### Copyright 2019-2023 by Larry Hastings

## Overview

`vec` is a module currently publishing one class: `Vector2`, a 2D
vector object designed for game development.

Features:

* `Vector2` objects are immutable.
* `Vector2` support all the usual vector object features,
  including operator overloading.
* Attributes of `Vector2` objects are lazily-computed where possible.
* `Vector2` objects effortlessly support both cartesian and polar coordinates.

`vec` supports Python 3.6+, and passes its unit test suite with 100% coverage.


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
coordinates.  This is simply bad API design--it doubles the surface
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

prints `1.5707963267948966 1.0`.  That first number is π/2 (approximately).

Conversely, you can define a `Vector2` object using polar
coordinates, then examine its cartesian coordinates:

    v2 = vec.Vector2(r=1, theta=math.pi/2)
    print(v2.x, v2.y)

This prints `6.123233995736766e-17 1.0`.  Conceptually this should
print `0.0, 1.0`--but `math.pi` is only an approximation, which sadly
means our result has an infinitesimal error.

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
defined using polar coordinates, and you call `.rotate()` or `.scale()`
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


## The API

`Vector2(x=None, y=None, *, r=None, theta=None, r_squared=None)`

<dl><dd>

Constructs a `Vector2` object.  You may pass in as many or as
few of these arguments as you like; however, you *must* pass in
*either* both `x` and `y` *or* both `r` and `theta`.
Any attributes not passed in at construction time will be lazily
computed at the time they are evaluated.

You can also pass in a single object which will initialize the
vector.  Supported objects include:

* an existing `Vector2` object (just returns that object),
* an object which has `.x` and `.y` attributes,
* a mapping object with exactly two keys, `'x'` and `'y'`, and
* an ordered iterable object with exactly two elements.

`Vector2` only does *some* validation of its arguments.
It ensures that `r` and `theta` are normalized.  However,
it doesn't check that `(x, y)` and `(r, theta)` describe the
same vector.
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

`Vector2` objects also support the *boolean* protocol; you may use them
with boolean operators, and you may call `bool()` on them.  When used in
a boolean context, the zero vector evaluates to `False`, and all other
vectors evaluate to `True`.

`Vector2` objects are *hashable,* but they're not *ordered*.
(You can't ask if one vector is *less than* another.)

`Vector2` objects support the following operators:

* `v1 + v2` adds the two vectors together.
* `v1 - v2` subtracts the right vector from the left vector.
* `v1 * scalar` mulitplies the vector by a scalar amount, equivalent to `v1.scale(scalar)`.
* `v1 / scalar` divides the vector by a scalar amount.
* `+v1` is exactly the same as `v1`.
* `-v1` returns the opposite of `v1`, such that `v1 + (-v1)` should be the zero vector.
   (This may not always be the case due to compounding floating-point errors.)
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

### Class methods

`vec.from_polar(r, theta)`

<dl><dd>

Constructs a `Vector2` object from the two polar coordinates
`r` and `theta`.

You can also pass in a single object which will be used to
initialize the vector.  Supported objects include:

* an existing `Vector2` object (just returns that object),
* an object which has `.r` and `.theta` attributes,
* a mapping object with exactly two keys, `'r'` and `'theta'`, and
* an ordered iterable object with exactly two elements.

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
</dd></dl>

`Vector2.normalized()`

<dl><dd>

Returns a new `Vector2` object, equivalent to the original vector scaled to length 1.
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
In actuality this returns the "perpendicular dot product", or "perp dot product",
of the two vectors, because that's what people usually mean when they say they
want the "cross product" of two 2D vectors.

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
`ratio` is not clamped to this range.
</dd></dl>

`Vector2.slerp(other, ratio)`

<dl><dd>

Returns a vector representing a spherical interpolation between `self` and `other`, according
to the scalar ratio `ratio`.  `ratio` should be a value between (and including) `0` and `1`.
If `ratio` is `0`, this returns `self`.  If `ratio` is `1`, this returns `other`.

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
and `theta` set to a value besides `None`.
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


## Extending vec to handle other types

`vec` does some input verification on its inputs.
Coordinates--`x`, `y`, `r`, `theta`--are required to be
either `int` or `long`.
(Technically `theta` can also be `None`.)  This best serves
the intended use case of `vec` as a 2D vector library for
game programming in Python.

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


## Changelog

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
