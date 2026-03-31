import asyncio

from typing import Any, AsyncIterable, Callable, Dict, List, Optional, Union

from tartiflette.coercers.arguments import coerce_arguments
from tartiflette.coercers.common import Path
from tartiflette.coercers.outputs.common import complete_value_catching_error, complete_value_catching_error_serial
from tartiflette.constants import UNDEFINED_VALUE
from tartiflette.execution.collect import collect_fields
from tartiflette.execution.context import build_execution_context
from tartiflette.execution.helpers import get_field_definition
from tartiflette.execution.types import build_resolve_info
from tartiflette.utils.errors import extract_exceptions_from_results
from tartiflette.utils.values import is_invalid_value

__all__ = (
    "resolve_field",
    "resolve_field_serial",
    "execute_fields",
    "execute_fields_serial",
    "execute",
    "execute_serial",
    "create_source_event_stream",
)


# Sentinel object for getattr default value (faster than exception handling)
_SENTINEL = object()


def _try_get_field_from_source(source: Any, field_name: str) -> Any:
    """
    Try to get the field value directly from the source object.
    Returns the value if found, None if not found or if source is None.
    Optimized to avoid exception handling overhead in the common case.
    :param source: the source object to check
    :param field_name: the name of the field to retrieve
    :type source: Any
    :type field_name: str
    :return: the field value if found, None otherwise
    :rtype: Any
    """
    if source is None:
        return None
    
    # Fast path: try attribute access first (most common case for Python objects)
    # Using getattr with sentinel avoids exception handling overhead
    result = getattr(source, field_name, _SENTINEL)
    if result is not _SENTINEL:
        return result
    
    # Fallback: try dict-like access
    # Try .get() method first if available (dict, OrderedDict, etc.) - avoids KeyError
    get_method = getattr(source, 'get', _SENTINEL)
    if get_method is not _SENTINEL:
        result = get_method(field_name, _SENTINEL)
        if result is not _SENTINEL:
            return result
    
    return None


async def resolve_field(
    execution_context: "ExecutionContext",
    parent_type: "GraphQLObjectType",
    source: Any,
    field_nodes: List["FieldNode"],
    path: "Path",
    is_introspection_context: bool = False,
) -> Any:
    """
    Resolves the field on the given source object. In particular, this
    figures out the value that the field returns by calling its resolve
    function, then calls completeValue to complete promises, serialize scalars,
    or execute the sub-selection-set for objects.
    :param execution_context: instance of the query execution context
    :param parent_type: GraphQLObjectType of the field's parent
    :param source: default root value or field parent value
    :param field_nodes: AST nodes related to the resolved field
    :param path: the path traveled until this resolver
    :param is_introspection_context: determines whether or not the resolved
    field is in a context of an introspection query
    :type execution_context: ExecutionContext
    :type parent_type: GraphQLObjectType
    :type source: Any
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type is_introspection_context: bool
    :return: the computed field value
    :rtype: Any
    """
    field_node = field_nodes[0]
    field_name = field_node.name.value

    field_definition = get_field_definition(
        execution_context.schema, parent_type, field_name
    )
    if field_definition is None:
        return UNDEFINED_VALUE

    return await field_definition.resolver(
        execution_context,
        parent_type,
        source,
        field_nodes,
        path,
        is_introspection_context,
    )


async def resolve_field_serial(
    execution_context: "ExecutionContext",
    parent_type: "GraphQLObjectType",
    source: Any,
    field_nodes: List["FieldNode"],
    path: "Path",
    is_introspection_context: bool = False,
) -> Any:
    """
    Serial version of resolve_field that resolves fields one by one without
    concurrent execution patterns.
    :param execution_context: instance of the query execution context
    :param parent_type: GraphQLObjectType of the field's parent
    :param source: default root value or field parent value
    :param field_nodes: AST nodes related to the resolved field
    :param path: the path traveled until this resolver
    :param is_introspection_context: determines whether or not the resolved
    field is in a context of an introspection query
    :type execution_context: ExecutionContext
    :type parent_type: GraphQLObjectType
    :type source: Any
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type is_introspection_context: bool
    :return: the computed field value
    :rtype: Any
    """
    from tartiflette.resolver.factory import resolve_field_serial as factory_resolve_field_serial
    from tartiflette.execution.helpers import get_field_definition
    
    field_node = field_nodes[0]
    field_name = field_node.name.value

    field_definition = get_field_definition(
        execution_context.schema, parent_type, field_name
    )
    if field_definition is None:
        return UNDEFINED_VALUE

    # field_definition.resolver is a partial wrapping resolve_field from factory.py
    # It has keywords: {'field_definition': self, 'resolver': wraps_with_directives(...), 'output_coercer': ...}
    # The 'resolver' keyword contains the wrapped resolver from bake time (wrapped with resolver_executor)
    # We need to extract it to pass to resolve_field_serial, just like resolve_field receives it
    from functools import partial
    if isinstance(field_definition.resolver, partial) and hasattr(field_definition.resolver, 'keywords'):
        # Extract the resolver from the partial (same as what resolve_field receives)
        resolver_func = field_definition.resolver.keywords.get('resolver')
        # This should always exist - if it doesn't, something is wrong
        if resolver_func is None:
            # Fallback to raw_resolver (shouldn't normally happen)
            resolver_func = field_definition.raw_resolver
    else:
        # If resolver is not a partial, use raw_resolver (shouldn't happen in normal cases)
        resolver_func = field_definition.raw_resolver

    # Get serial output_coercer (with concurrently=False for serial execution)
    from tartiflette.coercers.outputs.compute import get_output_coercer
    output_coercer = get_output_coercer(
        field_definition.graphql_type, concurrently=False
    )

    return await factory_resolve_field_serial(
        execution_context,
        parent_type,
        source,
        field_nodes,
        path,
        is_introspection_context,
        field_definition,
        resolver_func,
        output_coercer,
    )


async def execute_fields_serial(
    execution_context: "ExecutionContext",
    parent_type: "GraphQLObjectType",
    source_value: Any,
    path: Optional["Path"],
    fields: Dict[str, List["FieldNode"]],
    is_introspection_context: bool = False,
) -> Dict[str, Any]:
    """
    Serial version of execute_fields that resolves each field one by one
    without using asyncio.gather or any concurrent execution patterns.
    :param execution_context: instance of the query execution context
    :param parent_type: GraphQLObjectType of the field's parent
    :param source_value: default root value or field parent value
    :param path: the path traveled until this resolver
    :param fields: dictionary of collected fields
    :param is_introspection_context: determines whether or not the resolved
    field is in a context of an introspection query
    :type execution_context: ExecutionContext
    :type parent_type: GraphQLObjectType
    :type source_value: Any
    :type path: Optional[Path]
    :type fields: Dict[str, List[FieldNode]]
    :type is_introspection_context: bool
    :return: the computed fields value
    :rtype: Dict[str, Any]
    """
    results = []
    for entry_key, field_nodes in fields.items():
        try:
            field_definition = get_field_definition(
                execution_context.schema, parent_type, field_nodes[0].name.value
            )
            
            # Early optimization: try to get field value from source before calling resolve_field
            # Only applies when:
            # 1. Field definition exists
            # 2. No directives on field nodes (quick check without computing)
            # 3. Field has no arguments
            # 4. Using default resolver (no custom resolver)
            # 5. Optimization flag is enabled
            # 6. Not introspection context
            if (field_definition is not None
                and not any(field_node.directives for field_node in field_nodes)
                and not field_definition.arguments
                and field_definition.raw_resolver is None
                and execution_context.schema.skip_resolved_field_default_resolver
                and not is_introspection_context):
                field_value = _try_get_field_from_source(source_value, entry_key)
                if field_value is not None:
                    # Value found in source, build info and coerce value
                    field_path = Path(path, entry_key)
                    info = build_resolve_info(
                        execution_context,
                        field_definition,
                        field_nodes,
                        parent_type,
                        field_path,
                        is_introspection_context,
                    )
                    # Get serial output_coercer (with concurrently=False)
                    from tartiflette.coercers.outputs.compute import get_output_coercer
                    output_coercer = get_output_coercer(
                        field_definition.graphql_type, concurrently=False
                    )
                    result = await complete_value_catching_error_serial(
                        field_value,
                        info,
                        execution_context,
                        field_nodes,
                        field_path,
                        field_definition.graphql_type,
                        output_coercer,
                    )
                    results.append(result)
                    continue
            
            result = await resolve_field_serial(
                execution_context,
                parent_type,
                source_value,
                field_nodes,
                Path(path, entry_key),
                is_introspection_context,
            )
        except Exception as e:  # pylint: disable=broad-except
            # Wrap exception in MultipleException if it's not already
            from tartiflette.utils.errors import located_error, MultipleException
            if isinstance(e, MultipleException):
                result = e
            else:
                field_definition = get_field_definition(
                    execution_context.schema, parent_type, field_nodes[0].name.value
                )
                if field_definition:
                    result = located_error(e, field_nodes, Path(path, entry_key).as_list())
                else:
                    result = e
        results.append(result)

    # Check for exceptions in results (like the regular executor does)
    exceptions = extract_exceptions_from_results(results)
    if exceptions:
        raise exceptions

    # Build results dictionary, filtering out invalid values
    return {
        entry_key: result
        for entry_key, result in zip(fields, results)
        if not is_invalid_value(result)
    }


async def execute_fields_serially(
    execution_context: "ExecutionContext",
    parent_type: "GraphQLObjectType",
    source_value: Any,
    path: Optional["Path"],
    fields: Dict[str, List["FieldNode"]],
) -> Dict[str, Any]:
    """
    Implements the "Evaluating selection sets" section of the spec for "write"
    mode.
    :param execution_context: instance of the query execution context
    :param parent_type: GraphQLObjectType of the field's parent
    :param source_value: default root value or field parent value
    :param path: the path traveled until this resolver
    :param fields: dictionary of collected fields
    :type execution_context: ExecutionContext
    :type parent_type: GraphQLObjectType
    :type source_value: Any
    :type path: Optional[Path]
    :type fields: Dict[str, List[FieldNode]]
    :return: the computed fields value
    :rtype: Dict[str, Any]
    """
    results = {}
    for entry_key, field_nodes in fields.items():
        result = await resolve_field(
            execution_context,
            parent_type,
            source_value,
            field_nodes,
            Path(path, entry_key),
        )
        if not is_invalid_value(result):
            results[entry_key] = result
    return results


async def execute_fields(
    execution_context: "ExecutionContext",
    parent_type: "GraphQLObjectType",
    source_value: Any,
    path: Optional["Path"],
    fields: Dict[str, List["FieldNode"]],
    is_introspection_context: bool = False,
) -> Dict[str, Any]:
    """
    Implements the "Evaluating selection sets" section of the spec for "read"
    mode.
    :param execution_context: instance of the query execution context
    :param parent_type: GraphQLObjectType of the field's parent
    :param source_value: default root value or field parent value
    :param path: the path traveled until this resolver
    :param fields: dictionary of collected fields
    :param is_introspection_context: determines whether or not the resolved
    field is in a context of an introspection query
    :type execution_context: ExecutionContext
    :type parent_type: GraphQLObjectType
    :type source_value: Any
    :type path: Optional[Path]
    :type fields: Dict[str, List[FieldNode]]
    :type is_introspection_context: bool
    :return: the computed fields value
    :rtype: Dict[str, Any]
    """
    results = []
    to_await = {}
    for index, (entry_key, field_nodes) in enumerate(fields.items()):
        field_definition = get_field_definition(
            execution_context.schema, parent_type, field_nodes[0].name.value
        )
        if field_definition is None:
            results.append(UNDEFINED_VALUE)
            continue

        # Early optimization: try to get field value from source before calling resolve_field
        # Only applies when:
        # 1. No directives on field nodes (quick check without computing)
        # 2. Field has no arguments
        # 3. Using default resolver (no custom resolver)
        # 4. Optimization flag is enabled
        # 5. Not introspection context
        if (not any(field_node.directives for field_node in field_nodes)
            and not field_definition.arguments
            and field_definition.raw_resolver is None
            and execution_context.schema.skip_resolved_field_default_resolver
            and not is_introspection_context):
            field_value = _try_get_field_from_source(source_value, entry_key)
            if field_value is not None:
                # Value found in source, build info and coerce value
                field_path = Path(path, entry_key)
                info = build_resolve_info(
                    execution_context,
                    field_definition,
                    field_nodes,
                    parent_type,
                    field_path,
                    is_introspection_context,
                )
                # Get output_coercer (with concurrently based on parent_concurrently)
                from tartiflette.coercers.outputs.compute import get_output_coercer
                output_coercer = get_output_coercer(
                    field_definition.graphql_type, concurrently=field_definition.parent_concurrently
                )
                coerce_result = complete_value_catching_error(
                    field_value,
                    info,
                    execution_context,
                    field_nodes,
                    field_path,
                    field_definition.graphql_type,
                    output_coercer,
                )
                # Handle concurrent vs non-concurrent execution
                if field_definition.parent_concurrently:
                    to_await[index] = coerce_result
                    results.append(None)
                else:
                    results.append(await coerce_result)
                continue

        result = field_definition.resolver(
            execution_context,
            parent_type,
            source_value,
            field_nodes,
            Path(path, entry_key),
            is_introspection_context,
        )

        if field_definition.parent_concurrently:
            to_await[index] = result
            results.append(None)
        else:
            results.append(await result)

    if to_await:
        awaited = await asyncio.gather(
            *list(to_await.values()), return_exceptions=True
        )
        for index, result in zip(to_await, awaited):
            results[index] = result

    exceptions = extract_exceptions_from_results(results)
    if exceptions:
        raise exceptions

    return {
        entry_key: result
        for entry_key, result in zip(fields, results)
        if not is_invalid_value(result)
    }


async def execute_operation(
    execution_context: "ExecutionContext",
    operation: "OperationDefinitionNode",
    root_value: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """
    Implements the "Evaluating operations" section of the spec.
    :param execution_context: instance of the query execution context
    :param operation: AST operation definition node to execute
    :param root_value: default value for root fields
    :type execution_context: ExecutionContext
    :type operation: OperationDefinitionNode
    :type root_value: Optional[Any]
    :return: Optional[Dict[str, Any]]
    :rtype: the computed value
    """
    operation_root_type = execution_context.schema.get_operation_root_type(
        operation
    )

    fields = await collect_fields(
        execution_context, operation_root_type, operation.selection_set
    )

    try:
        return await (
            execute_fields_serially(
                execution_context,
                operation_root_type,
                root_value,
                None,
                fields,
            )
            if operation.operation_type == "mutation"
            else execute_fields(
                execution_context,
                operation_root_type,
                root_value,
                None,
                fields,
            )
        )
    except Exception as e:  # pylint: disable=broad-except
        execution_context.add_error(e)
        return None


async def execute_operation_serial(
    execution_context: "ExecutionContext",
    operation: "OperationDefinitionNode",
    root_value: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """
    Implements the "Evaluating operations" section of the spec with serial
    field resolution (no asyncio.gather, fields resolved one by one).
    :param execution_context: instance of the query execution context
    :param operation: AST operation definition node to execute
    :param root_value: default value for root fields
    :type execution_context: ExecutionContext
    :type operation: OperationDefinitionNode
    :type root_value: Optional[Any]
    :return: Optional[Dict[str, Any]]
    :rtype: the computed value
    """
    operation_root_type = execution_context.schema.get_operation_root_type(
        operation
    )

    fields = await collect_fields(
        execution_context, operation_root_type, operation.selection_set
    )

    try:
        return await execute_fields_serial(
            execution_context,
            operation_root_type,
            root_value,
            None,
            fields,
        )
    except Exception as e:  # pylint: disable=broad-except
        execution_context.add_error(e)
        return None


async def execute(
    schema: "GraphQLSchema",
    document: "DocumentNode",
    response_builder: Callable,
    root_value: Optional[Any],
    context: Optional[Any],
    variables: Optional[Dict[str, Any]],
    operation_name: Optional[str],
) -> Dict[str, Any]:
    """
    Runs the execution of the executable operation.
    :param schema: the GraphQLSchema instance linked to the engine
    :param document: the DocumentNode instance linked to the GraphQL request
    :param response_builder: callable in charge of returning the formatted
    GraphQL response
    :param root_value: an initial value corresponding to the root type being
    executed
    :param context: value that can contain everything you need and that will be
    accessible from the resolvers
    :param variables: the variables provided in the GraphQL request
    :param operation_name: the operation name to execute
    :type schema: GraphQLSchema
    :type document: DocumentNode
    :type response_builder: Callable
    :type root_value: Optional[Any]
    :type context: Optional[Any]
    :type variables: Optional[Dict[str, Any]]
    :type operation_name: str
    :return: the GraphQL response linked to the operation execution
    :rtype: Dict[str, Any]
    """
    execution_context, errors = await build_execution_context(
        schema, document, root_value, context, variables, operation_name
    )

    if errors:
        return await response_builder(errors=errors)

    data = await execute_operation(
        execution_context, execution_context.operation, root_value
    )
    return await response_builder(data=data, errors=execution_context.errors)


async def execute_serial(
    schema: "GraphQLSchema",
    document: "DocumentNode",
    response_builder: Callable,
    root_value: Optional[Any],
    context: Optional[Any],
    variables: Optional[Dict[str, Any]],
    operation_name: Optional[str],
) -> Dict[str, Any]:
    """
    Runs the execution of the executable operation with serial field resolution.
    All fields are resolved one by one, without using asyncio.gather for
    concurrent execution.
    :param schema: the GraphQLSchema instance linked to the engine
    :param document: the DocumentNode instance linked to the GraphQL request
    :param response_builder: callable in charge of returning the formatted
    GraphQL response
    :param root_value: an initial value corresponding to the root type being
    executed
    :param context: value that can contain everything you need and that will be
    accessible from the resolvers
    :param variables: the variables provided in the GraphQL request
    :param operation_name: the operation name to execute
    :type schema: GraphQLSchema
    :type document: DocumentNode
    :type response_builder: Callable
    :type root_value: Optional[Any]
    :type context: Optional[Any]
    :type variables: Optional[Dict[str, Any]]
    :type operation_name: str
    :return: the GraphQL response linked to the operation execution
    :rtype: Dict[str, Any]
    """
    execution_context, errors = await build_execution_context(
        schema, document, root_value, context, variables, operation_name
    )

    if errors:
        return await response_builder(errors=errors)

    data = await execute_operation_serial(
        execution_context, execution_context.operation, root_value
    )
    return await response_builder(data=data, errors=execution_context.errors)


async def create_source_event_stream(
    schema: "GraphQLSchema",
    document: "DocumentNode",
    response_builder: Callable,
    root_value: Optional[Any],
    context: Optional[Any],
    variables: Optional[Dict[str, Any]],
    operation_name: Optional[str],
) -> Union[AsyncIterable[Dict[str, Any]], Dict[str, Any]]:
    """
    Resolves the subscription source event stream.
    :param schema: the GraphQLSchema instance linked to the engine
    :param document: the DocumentNode instance linked to the GraphQL request
    :param response_builder: callable in charge of returning the formatted
    GraphQL response
    :param root_value: an initial value corresponding to the root type being
    executed
    :param context: value that can contain everything you need and that will be
    accessible from the resolvers
    :param variables: the variables provided in the GraphQL request
    :param operation_name: the operation name to execute
    :type schema: GraphQLSchema
    :type document: DocumentNode
    :type response_builder: Callable
    :type root_value: Optional[Any]
    :type context: Optional[Any]
    :type variables: Optional[Dict[str, Any]]
    :type operation_name: str
    :return: an error or an async iterable
    :rtype: Union[AsyncIterable[Dict[str, Any]], Dict[str, Any]]
    """
    # pylint: disable=too-many-locals
    execution_context, errors = await build_execution_context(
        schema, document, root_value, context, variables, operation_name
    )

    if errors:
        return await response_builder(errors=errors)

    operation_root_type = schema.get_operation_root_type(
        execution_context.operation
    )

    fields = await collect_fields(
        execution_context,
        operation_root_type,
        execution_context.operation.selection_set,
    )

    response_name = list(fields.keys())[0]
    field_nodes = fields[response_name]
    field_name = field_nodes[0].name.value
    field_definition = get_field_definition(
        schema, operation_root_type, field_name
    )

    if not field_definition:
        raise Exception(
            f"The subscription field < {field_name} > is not defined."
        )

    if not field_definition.subscribe:
        raise Exception(
            "Can't execute a subscription query on a field which doesn't "
            "provide a source event stream with < @Subscription >."
        )

    info = build_resolve_info(
        execution_context,
        field_definition,
        field_nodes,
        operation_root_type,
        Path(None, response_name),
    )

    return field_definition.subscribe(
        root_value,
        await coerce_arguments(
            field_definition.arguments,
            field_nodes[0],
            execution_context.variable_values,
            execution_context.context,
            coercer=field_definition.arguments_coercer,
        ),
        execution_context.context,
        info,
    )
