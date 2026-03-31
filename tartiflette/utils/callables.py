from inspect import (
    isasyncgenfunction,
    iscoroutinefunction,
    isfunction,
    ismethod,
)
from typing import AsyncGenerator, Callable, Coroutine


def is_valid_coroutine(coroutine: Coroutine) -> bool:
    """
    Determines whether or not the filled in parameter is a valid coroutine
    callable.
    :param coroutine: object to check
    :type coroutine: Coroutine
    :return: whether or not the filled in parameter is a valid coroutine
    callable
    :rtype: bool
    """
    return iscoroutinefunction(
        coroutine
        if isfunction(coroutine) or ismethod(coroutine)
        else coroutine.__call__
    )


def is_valid_async_generator(generator: AsyncGenerator) -> bool:
    """
    Determines whether or not the filled in parameter is a valid asynchronous
    generator.
    :param generator: object to check
    :type generator: AsyncGenerator
    :return: whether or not the filled in parameter is a valid asynchronous
    generator
    :rtype: bool
    """
    return isasyncgenfunction(
        generator
        if isfunction(generator) or ismethod(generator)
        else generator.__call__
    )


def is_sync_callable(callable_obj: Callable) -> bool:
    """
    Determines whether or not the filled in parameter is a synchronous callable.
    :param callable_obj: object to check
    :type callable_obj: Callable
    :return: whether or not the filled in parameter is a synchronous callable
    :rtype: bool
    """
    if not callable(callable_obj):
        return False
    
    # Check if it's a coroutine function (async)
    if iscoroutinefunction(
        callable_obj
        if isfunction(callable_obj) or ismethod(callable_obj)
        else callable_obj.__call__
    ):
        return False
    
    # Check if it's an async generator function
    if isasyncgenfunction(
        callable_obj
        if isfunction(callable_obj) or ismethod(callable_obj)
        else callable_obj.__call__
    ):
        return False
    
    return True
