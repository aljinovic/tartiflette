from typing import Any, Callable, Dict, List

from tartiflette.execution.collect import collect_subfields
from tartiflette.execution.execute import execute_fields, execute_fields_serial
from tartiflette.utils.errors import located_error

__all__ = (
    "complete_value_catching_error",
    "complete_value_catching_error_serial",
    "complete_object_value",
    "complete_object_value_serial",
)


def handle_field_error(
    raw_error: Exception,
    field_nodes: List["FieldNode"],
    path: "Path",
    return_type: "GraphQLOutputType",
    execution_context: "ExecutionContext",
) -> None:
    """
    Computes the raw error to a TartifletteError and add it to the execution
    context or bubble up the error if the field can't be null.
    :param raw_error: the raw exception to be treated
    :param field_nodes: AST nodes related to the resolved field
    :param path: the path traveled until this field
    :param return_type: GraphQLOutputType instance of the resolved field
    :param execution_context: instance of the query execution context
    :type raw_error: Exception
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type return_type: GraphQLOutputType
    :type execution_context: ExecutionContext
    """
    error = located_error(raw_error, field_nodes, path.as_list())

    # If the field type is non-nullable, then it is resolved without any
    # protection from errors, however it still properly locates the error.
    if return_type.is_non_null_type:
        raise error

    # Otherwise, error protection is applied, logging the error and resolving
    # a null value for this field if one is encountered.
    execution_context.add_error(error)
    return None


async def complete_value_catching_error(
    result: Any,
    info: "ResolveInfo",
    execution_context: "ExecutionContext",
    field_nodes: List["FieldNode"],
    path: "Path",
    return_type: "GraphQLOutputType",
    output_coercer: Callable,
) -> Any:
    """
    Coerce the resolved field value or catch the resolver exception to add it
    to the execution context.
    :param result: resolved field value
    :param info: information related to the execution and the resolved field
    :param execution_context: instance of the query execution context
    :param field_nodes: AST nodes related to the resolved field
    :param path: the path traveled until this resolver
    :param return_type: GraphQLOutputType instance of the resolved field
    :param output_coercer: pre-computed callable to coerce the result value
    :type result: Any
    :type info: ResolveInfo
    :type execution_context: ExecutionContext
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type return_type: GraphQLOutputType
    :type output_coercer: Callable
    :return: the coerced resolved field value
    :rtype: Any
    """
    try:
        if isinstance(result, Exception):
            raise result

        return await output_coercer(
            result, info, execution_context, field_nodes, path
        )
    except Exception as raw_exception:  # pylint: disable=broad-except
        return handle_field_error(
            raw_exception, field_nodes, path, return_type, execution_context
        )


async def complete_value_catching_error_serial(
    result: Any,
    info: "ResolveInfo",
    execution_context: "ExecutionContext",
    field_nodes: List["FieldNode"],
    path: "Path",
    return_type: "GraphQLOutputType",
    output_coercer: Callable,
) -> Any:
    """
    Serial version of complete_value_catching_error that ensures sequential
    execution without concurrent patterns.
    :param result: resolved field value
    :param info: information related to the execution and the resolved field
    :param execution_context: instance of the query execution context
    :param field_nodes: AST nodes related to the resolved field
    :param path: the path traveled until this resolver
    :param return_type: GraphQLOutputType instance of the resolved field
    :param output_coercer: pre-computed callable to coerce the result value
    :type result: Any
    :type info: ResolveInfo
    :type execution_context: ExecutionContext
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type return_type: GraphQLOutputType
    :type output_coercer: Callable
    :return: the coerced resolved field value
    :rtype: Any
    """
    try:
        if isinstance(result, Exception):
            raise result

        # Use output_coercer for all types - it already includes non-null validation
        # and is configured for serial execution (concurrently=False) when used in serial mode
        return await output_coercer(
            result, info, execution_context, field_nodes, path
        )
    except Exception as raw_exception:  # pylint: disable=broad-except
        # handle_field_error will raise for non-nullable fields, so we don't return its result
        # For nullable fields, it adds the error to execution_context and returns None
        handle_field_error(
            raw_exception, field_nodes, path, return_type, execution_context
        )
        # If we get here, the field is nullable and the error was added to context
        return None


async def complete_object_value(
    result: Any,
    info: "ResolveInfo",
    execution_context: "ExecutionContext",
    field_nodes: List["FieldNode"],
    path: "Path",
    return_type: "GraphQLOutputType",
) -> Dict[str, Any]:
    """
    Complete an Object value by executing all sub-selections.
    :param result: result to treat
    :param info: information related to the execution and the resolved field
    :param execution_context: instance of the query execution context
    :param field_nodes: AST nodes related to the coerced field
    :param path: the path traveled until this coercer
    :param return_type: the GraphQLObjectType instance of the object
    :type result: Any
    :type info: ResolveInfo
    :type execution_context: ExecutionContext
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type return_type: GraphQLOutputType
    :return: the computed value
    :rtype: Dict[str, Any]
    """
    return await execute_fields(
        execution_context,
        return_type,
        result,
        path,
        await collect_subfields(execution_context, return_type, field_nodes),
        info.is_introspection,
    )


async def complete_object_value_serial(
    result: Any,
    info: "ResolveInfo",
    execution_context: "ExecutionContext",
    field_nodes: List["FieldNode"],
    path: "Path",
    return_type: "GraphQLOutputType",
) -> Dict[str, Any]:
    """
    Serial version of complete_object_value that ensures sequential execution
    without concurrent patterns.
    :param result: result to treat
    :param info: information related to the execution and the resolved field
    :param execution_context: instance of the query execution context
    :param field_nodes: AST nodes related to the coerced field
    :param path: the path traveled until this coercer
    :param return_type: the GraphQLObjectType instance of the object
    :type result: Any
    :type info: ResolveInfo
    :type execution_context: ExecutionContext
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type return_type: GraphQLOutputType
    :return: the computed value
    :rtype: Dict[str, Any]
    """
    # If result is None, return None immediately (should be handled by null_coercer_wrapper,
    # but this is a safety check to prevent resolving subfields on None)
    if result is None:
        return None
    
    subfields = await collect_subfields(execution_context, return_type, field_nodes)
    return await execute_fields_serial(
        execution_context,
        return_type,
        result,
        path,
        subfields,
        info.is_introspection,
    )
