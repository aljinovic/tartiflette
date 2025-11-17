from functools import partial
from typing import Callable

from tartiflette.coercers.outputs.list_coercer import (
    list_coercer_concurrently,
    list_coercer_sequentially,
)
from tartiflette.coercers.outputs.non_null_coercer import non_null_coercer

__all__ = ("get_output_coercer",)


def get_output_coercer(
    graphql_type: "GraphQLType", concurrently: bool
) -> Callable:
    """
    Computes and returns the output coercer to use for the filled in schema
    type.
    :param graphql_type: the schema type for which compute the coercer
    :param concurrently: whether list should be coerced concurrently
    :type graphql_type: GraphQLType
    :type concurrently: bool
    :return: the computed coercer wrap with directives if defined
    :rtype: Callable
    """
    inner_type = graphql_type
    wrapper_coercers = []
    while inner_type.is_wrapping_type:
        wrapped_type = inner_type.wrapped_type
        if inner_type.is_list_type:
            wrapper_coercers.append(
                partial(
                    list_coercer_concurrently
                    if concurrently
                    else list_coercer_sequentially,
                    item_type=wrapped_type,
                )
            )
        elif inner_type.is_non_null_type:
            wrapper_coercers.append(non_null_coercer)
        inner_type = wrapped_type

    try:
        # When concurrently=False (serial mode), use serial coercers for objects
        # We need to construct the same structure as inner_type.output_coercer but with object_coercer_serial
        if not concurrently and hasattr(inner_type, 'kind') and inner_type.kind == "OBJECT":
            from tartiflette.coercers.outputs.object_coercer import object_coercer_serial
            from tartiflette.coercers.outputs.directives_coercer import output_directives_coercer
            # Match the structure of inner_type.output_coercer:
            # partial(output_directives_coercer, coercer=partial(object_coercer, object_type=self), directives=...)
            coercer = partial(
                output_directives_coercer,
                coercer=partial(object_coercer_serial, object_type=inner_type),
                directives=inner_type.pre_output_coercion_directives,
            )
        else:
            coercer = inner_type.output_coercer
    except AttributeError:
        # This case should never happen and raise an exception at schema
        # validation time.
        coercer = lambda *args, **kwargs: None

    for wrapper_coercer in reversed(wrapper_coercers):
        coercer = partial(wrapper_coercer, inner_coercer=coercer)

    return coercer
