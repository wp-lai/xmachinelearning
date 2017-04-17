"""
A decorator for caching properties in classes.

Examples:
    >>> class Foo(object):
    ...     @cached_property
    ...     def bar(self):
    ...         print("This message only print once")
    ...         return None
    >>> foo = Foo()
    >>> foo.bar
    This message only print once
    >>> foo.bar
"""
import functools


def cached_property(func):
    attr = '_cached_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self, *args, **kwargs):
        if not hasattr(self, attr):
            setattr(self, attr, func(self, *args, **kwargs))
        return getattr(self, attr)

    return decorator
