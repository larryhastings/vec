# vec

## A reasonable, performant 2D vector object for games

##### Copyright 2019-2020 by Larry Hastings

## Overview

`vec` is a module currently containing one class: `Vector2`, a 2D
vector object designed for game development.

Features:

* `Vector2` objects are immutable.
* `Vector2` support all the usual features, including operator overloading.
* Attributes of `Vector2` objects are lazily-computed where possible.
* `Vector2` objects effortlessly support both cartesian and polar coordinates.


## Why Another Vector Class?

I've participated in three PyWeek gaming challenges.  In two of those three times,
mid-week I wrote my own vector class out of sheer frustration.

The biggest problem with most Python vector objects in is that they're mutable.
Frankly this way lies madness.  Vector objects should be immutable--it
just makes sense from an API perspective.  What if you set the position
of some game-engine pawn to be a particular vector object, then modify
that vector object?  Should the pawn update its position automatically--and if so,
how is it supposed to know the value changed?

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

`vec.Vector2` avoids all these problems.  `vec.Vector2` objects
are immutable,
they support vectors defined with either polar or cartesian coordinates,
and
they strictly use radians for polar operations.

## The Conceptual Model

`vec.Vector2` objects conceptually represent a vector.  They can be
defined using either cartesian or polar coordinates, and any `vec.Vector2`
can be queried for both its cartesian and polar coordinates.

Most vector objects in games are defined using cartesian coordinates.
`vec.Vector2` makes that easy, supporting any number of invocations
to create one.  Discrete parameters,
iterables, and objects that support `x` and `y` attributes all work fine:

    vec.Vector2(0, 1)
    vec.Vector2(x=0, y=1)
    vec.Vector2((0, 1))
    vec.Vector2([0, 1])
    vec.Vector2(types.SimpleNamespace(x=0, y=1))

All these define the same vector.  That last example is there to demonstrate
that `vec.Vector2` can create a vector based on any object with `x` and `y`
attributes.

Once you have a vector object, you can examine its attributes.
Every `vec.Vector2` object can be queried for both cartesian and
polar coordinates:

    v = vec.Vector2(0, 1)
    print(v.theta, v.r)

prints `1.5707963267948966 1.0`.  That first number is π/2 (approximately).

Conversely, you can also define `vec.Vector2` objects using polar
coordinates, and then ask for its cartesian coordinates:

    v2 = vec.Vector2(r=1, theta=1.5707963267948966)
    print(v2.x, v2.y)

This prints `6.123233995736766e-17 1.0`.  Conceptually this should
print `0.0, 1.0`--but `math.pi` is only an approximation, which means
sadly our result is off by an infinitesimal amount.

### Implementation Details

Internally `vec.Vector2` objects are either "cartesian" or "polar".
"cartesian" vector objects are defined in terms of `x` and `y`;
"polar" vector objects are defined in terms of `r` and `theta`.
All other attributes are lazily computed as needed.

`vec.Vector2` objects use slots, and rely on `__getattr__`
to implement this lazy computation.  Only the known values of the
vector are set when it's created.  If the user refers to an attribute
that hasn't been computed yet, Python will call `vec.Vector2.__getattr__()`,
which computes and then sets that value.  Future references to that
attribute skip this mechanism and simply return the cached value, which
is only as expensive as an attribute lookup on a conventional object.

Operations on `vec.Vector2` objects compute their result
using the cheapest approach.  If you have a `vec.Vector2` object
defined using polar coordinates, and you call `.rotate()` or `.scale()`
on it, all the math is done in the polar domain.  On the other
hand, adding vectors is always done in the cartesian domain, so
if you add a polar vector to any other vector, its cartesian
coordinates will be computed--and the resulting vector will always
be defined using cartesian coordinates.

Actually, that last statement isn't always true.  There's a special
case for adding two polar vectors which have the exact same `theta`:
just add their `r` values. That approach is much cheaper than
converting to cartesian, and more precise as well, returning a vector
defined using polar coordinates!  `vec.Vector2` takes advantage of
many such serendipities, computing your vectors as cheaply and accurately
as possible.


## The API

`vec.Vector2(x=None, y=None, *, r=None, theta=None, r_squared=None)`

Constructs a `vec.Vector2` object.  You may pass in as many or as
few of these arguments as you like; however, you *must* pass in
*either* both `x` and `y` *or* both `r` and `theta`.
Any attributes not passed in at construction time will be lazily
computed at the time they are evaluated.

(`vec.Vector2` only does *some* validation of its arguments.
It ensures that `r` and `theta` are normalized.  However,
it doesn't check that `(x, y)` and `(r, theta)` describe the
same vector.
If you pass in `x` and `y`, and a `theta` and `r` that don't
match, you'll get back the `vec.Vector2` that you asked for.
Good luck.)

`vec.Vector2` objects support five attributes:
`x`, `y`, `r`, `theta`, and `r_squared`.  It doesn't matter whether
the object was defined with cartesian or polar coordinates; these
all work.  `r_squared` is equivalent to `r*r` but it's much cheaper
to compute based on cartesian coordinates.

`vec.Vector2` objects support the *iterator protocol.*  You can call
`len()` on `vec.Vector2` objects--and it'll always return 2.
You can also iterate over them,
which will yield the `x` and `y` attributes in that order.

`vec.Vector2` objects support the *sequence protocol.*  You can subscript
them, which behaves as if the `vec.Vector2` object is a tuple of length
2 containing the `x` and `y` attributes.

`vec.Vector2` objects also support the *boolean* protocol; you may use them
with boolean operators, and you may call `bool()` on them.  When used in
a boolean context, the zero vector evaluates to `False`, and all other
vectors evaluate to `True`.

`vec.Vector2` objects are hashable.

`vec.Vector2` objects support the following operators:

* `v1 + v2` adds the two vectors together.
* `v1 - v2` subtracts the right vector from the left vector.
* `v1 * scalar` mulitplies the vector by a scalar amount, equivalent to `v1.scale(scalar)`.
* `v1 / scalar` divides the vector by a scalar amount.
* `+v1` is exactly the same as `v1`.
* `-v1` returns the opposite of `v1`, such that `v1 + (-v1)` should be the zero vector.
   (This may not always be the case due to compounding floating-point errors.)
* `v1 == v2` is `True` if the two vectors are *exactly* the same, and `False` otherwise.
* `v1 != v2` is `False` if the two vectors are *exactly* the same, and `True` otherwise.

`vec.Vector2` objects support the following methods:

`vec.Vector2.scaled(scalar)`

Returns a new `vec.Vector2` object, equivalent to the original vector multiplied by that scalar.

`vec.Vector2.scaled_to_length(r)`

Returns a new `vec.Vector2` object, equivalent to the original vector with its length set to `r`.

`vec.Vector2.normalized()`

Returns a new `vec.Vector2` object, equivalent to the original vector scaled to length 1.

`vec.Vector2.rotated(theta)`

Returns a new `vec.Vector2` object, equal to the original vector rotated by `theta` radians.

`vec.Vector2.dot(other)`

Returns the "dot product" `self` • `other`.  This result is a scalar value, not a vector.

`vec.Vector2.cross(other)`

Returns the "cross product" `self` ⨯ `other`.  This result is a scalar value, not a vector.

*Note:* technically, there is no "cross product" defined for 2-dimensional vectors.
In actuality this returns the "perpendicular dot product", or "perp dot product",
of the two vectors, because that's what people actually want when they ask for the "cross
product" of two 2D vectors.

`vec.Vector2.polar()`

Returns a 2-tuple of `(self.r, self.theta)`.

`vec.Vector2.lerp(other, ratio)`

Returns a vector representing a linear interpolation between `self` and `other`, according
to the scalar ratio `ratio`.  `ratio` should be a value between (and including) `0` and `1`.
If `ratio` is `0`, this returns `self`.  If `ratio` is `1`, this returns `other`.
If `ratio` is `0.4`, this returns `(self * 0.6) + (other * 0.4)`.

`vec.vector2_zero`

The immutable, eternal "zero" `vec.Vector2` vector object.
`vec` guarantees that every zero vector is a reference to this object:

    >>> v = vec.Vector2(0, 0)
    >>> v is vec.vector2_zero
    True

Mathematically-speaking, the zero vector when expressed in polar coordinates
doesn't have a defined angle.  Therefore `vec` defines its zero vector as
having an angle of `None`.
