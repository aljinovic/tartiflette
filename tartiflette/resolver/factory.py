from functools import partial
from typing import Any, Callable, List, Union

from tartiflette.coercers.arguments import coerce_arguments
from tartiflette.coercers.outputs.common import (
    complete_value_catching_error,
    complete_value_catching_error_serial,
)
from tartiflette.coercers.outputs.common import handle_field_error
from tartiflette.execution.types import build_resolve_info
from tartiflette.resolver.default import default_field_resolver_sync
from tartiflette.types.helpers.definition import (
    get_wrapped_type,
    is_scalar_type,
)
from tartiflette.types.helpers.get_directive_instances import (
    compute_directive_nodes,
)
from tartiflette.utils.callables import is_sync_callable, is_valid_coroutine
from tartiflette.utils.directives import (
    directive_executor,
    introspection_directives_executor,
    resolver_executor,
    wraps_with_directives,
)

__all__ = ("resolve_field", "resolve_field_serial")


# Sentinel object for getattr default value (faster than exception handling)
_SENTINEL = object()


def _can_skip_scalar_coercion(
    value: Any, scalar_type: "GraphQLScalarType"
) -> bool:
    """
    Check if we can skip coercion for a scalar value when it's already the
    correct type. This optimization works for built-in scalars where
    coerce_output is a no-op when the value is already the correct type.
    :param value: the value to check
    :param scalar_type: the GraphQLScalarType instance
    :type value: Any
    :type scalar_type: GraphQLScalarType
    :return: True if we can skip coercion, False otherwise
    :rtype: bool
    """
    if value is None:
        return False
    
    # Check for built-in scalars where coerce_output is a no-op for correct types
    scalar_name = scalar_type.name
    if scalar_name == "String":
        return isinstance(value, str)
    elif scalar_name == "Boolean":
        return isinstance(value, bool)
    # For other scalars (Int, Float, custom), we can't easily determine
    # if coercion is a no-op without calling coerce_output, so we don't skip
    return False


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


async def resolve_field_value_or_error(
    execution_context: "ExecutionContext",
    field_definition: "GraphQLField",
    field_nodes: List["FieldNode"],
    resolver: Callable,
    source: Any,
    info: "ResolveInfo",
) -> Union[Exception, Any]:
    """
    Coerce the field's arguments and then try to resolve the field.
    :param execution_context: instance of the query execution context
    :param field_definition: GraphQLField instance of the resolved field
    :param field_nodes: AST nodes related to the resolved field
    :param resolver: callable to use to resolve the field
    :param source: default root value or field parent value
    :param info: information related to the execution and the resolved field
    :type execution_context: ExecutionContext
    :type field_definition: GraphQLField
    :type field_nodes: List[FieldNode]
    :type resolver: Callable
    :type source: Any
    :type info: ResolveInfo
    :return: the resolved field value
    :rtype: Union[Exception, Any]
    """
    # pylint: disable=too-many-locals
    try:
        # Compute directives first
        computed_directives = []
        for field_node in field_nodes:
            computed_directives.extend(
                compute_directive_nodes(
                    execution_context.schema,
                    field_node.directives,
                    execution_context.variable_values,
                )
            )
        
        # Try to get the field value directly from the source first
        # Only use this optimization when:
        # 1. No directives are attached to the field
        # 2. Field has no arguments (to ensure argument validation isn't skipped)
        # 3. Using default resolver (to avoid bypassing custom resolver logic)
        # 4. skip_resolved_field_default_resolver is True (optimization enabled)
        field_value = None
        if (not computed_directives 
            and not field_definition.arguments 
            and field_definition.raw_resolver is None
            and execution_context.schema.skip_resolved_field_default_resolver):
            field_value = _try_get_field_from_source(source, info.field_name)
        if field_value is not None:
            # Value already exists in source, return it
            # Still need to handle introspection directives if present
            if info.is_introspection and computed_directives:
                return await introspection_directives_executor(
                    field_value,
                    execution_context.context,
                    info,
                    context_coercer=execution_context.context,
                )
            return field_value

        if computed_directives:
            resolver = wraps_with_directives(
                directives_definition=computed_directives,
                directive_hook="on_field_execution",
                func=resolver,
                is_resolver=True,
                with_default=True,
            )

        coerced_args = await coerce_arguments(
            field_definition.arguments,
            field_nodes[0],
            execution_context.variable_values,
            execution_context.context,
            coercer=field_definition.arguments_coercer,
        )
        
        # Check if resolver is wrapped (with resolver_executor or directive_executor)
        # Both wrappers handle sync/async detection internally and are async themselves
        is_wrapped_with_executor = (
            isinstance(resolver, partial)
            and hasattr(resolver, 'func')
            and resolver.func is resolver_executor
        )
        is_wrapped_with_directive = (
            isinstance(resolver, partial)
            and hasattr(resolver, 'func')
            and resolver.func is directive_executor
        )
        
        if computed_directives or is_wrapped_with_directive or is_wrapped_with_executor:
            # Wrapped resolver - always await (wrappers handle sync/async internally)
            result = await resolver(
                source,
                coerced_args,
                execution_context.context,
                info,
                context_coercer=execution_context.context,
            )
        elif is_sync_callable(resolver):
            # Sync resolver without directives - call directly
            result = resolver(
                source,
                coerced_args,
                execution_context.context,
                info,
            )
        else:
            # Async resolver without directives
            result = await resolver(
                source,
                coerced_args,
                execution_context.context,
                info,
                context_coercer=execution_context.context,
            )
        if info.is_introspection:
            return await introspection_directives_executor(
                result,
                execution_context.context,
                info,
                context_coercer=execution_context.context,
            )
        return result
    except Exception as e:  # pylint: disable=broad-except
        return e


async def resolve_field(
    execution_context: "ExecutionContext",
    parent_type: "GraphQLObjectType",
    source: Any,
    field_nodes: List["FieldNode"],
    path: "Path",
    is_introspection_context: bool,
    field_definition: "GraphQLField",
    resolver: Callable,
    output_coercer: Callable,
) -> Any:
    """
    Resolves the field value and coerce it before returning it.
    :param execution_context: instance of the query execution context
    :param parent_type: GraphQLObjectType of the field's parent
    :param source: default root value or field parent value
    :param field_nodes: AST nodes related to the resolved field
    :param path: the path traveled until this resolver
    :param field_definition: GraphQLField instance of the resolved field
    :param resolver: callable to use to resolve the field
    :param output_coercer: callable to use to coerce the resolved field value
    :param is_introspection_context: determines whether or not the resolved
    field is in a context of an introspection query
    :type execution_context: ExecutionContext
    :type parent_type: GraphQLObjectType
    :type source: Any
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type field_definition: GraphQLField
    :type resolver: Callable
    :type output_coercer: Callable
    :type is_introspection_context: bool
    :return: the coerced resolved field value
    :rtype: Any
    """
    # pylint: disable=too-many-arguments
    # Early optimization: try to get field value from source before computing directives
    # Only applies when:
    # 1. No directives on field nodes (quick check without computing)
    # 2. Field has no arguments
    # 3. Using default resolver (no custom resolver)
    # 4. Optimization flag is enabled
    # 5. Not introspection context (introspection may need directive processing)
    if (not any(field_node.directives for field_node in field_nodes)
        and not field_definition.arguments
        and field_definition.raw_resolver is None
        and execution_context.schema.skip_resolved_field_default_resolver
        and not is_introspection_context):
        field_value = _try_get_field_from_source(source, field_definition.name)
        if field_value is not None:
            # Value found in source, build info and return coerced value
            info = build_resolve_info(
                execution_context,
                field_definition,
                field_nodes,
                parent_type,
                path,
                is_introspection_context,
            )
            # Optimization: skip coercion for scalars when type matches
            unwrapped_type = get_wrapped_type(field_definition.graphql_type)
            if (is_scalar_type(unwrapped_type)
                and _can_skip_scalar_coercion(field_value, unwrapped_type)):
                # Handle exceptions (field_value is already known to be not None)
                if isinstance(field_value, Exception):
                    return handle_field_error(
                        field_value,
                        field_nodes,
                        path,
                        field_definition.graphql_type,
                        execution_context,
                    )
                
                return field_value
            return await complete_value_catching_error(
                field_value,
                info,
                execution_context,
                field_nodes,
                path,
                field_definition.graphql_type,
                output_coercer,
            )
    
    info = build_resolve_info(
        execution_context,
        field_definition,
        field_nodes,
        parent_type,
        path,
        is_introspection_context,
    )

    result = await resolve_field_value_or_error(
        execution_context,
        field_definition,
        field_nodes,
        resolver,
        source,
        info,
    )
    
    # Optimization: skip coercion for scalars when type matches
    unwrapped_type = get_wrapped_type(field_definition.graphql_type)
    if (is_scalar_type(unwrapped_type)
        and _can_skip_scalar_coercion(result, unwrapped_type)):
        # Handle exceptions and non-null validation
        if isinstance(result, Exception):
            return handle_field_error(
                result,
                field_nodes,
                path,
                field_definition.graphql_type,
                execution_context,
            )
        # Non-null validation
        if field_definition.graphql_type.is_non_null_type and result is None:
            return handle_field_error(
                ValueError("Non-null field cannot be null"),
                field_nodes,
                path,
                field_definition.graphql_type,
                execution_context,
            )
        return result
    
    return await complete_value_catching_error(
        result,
        info,
        execution_context,
        field_nodes,
        path,
        field_definition.graphql_type,
        output_coercer,
    )


async def resolve_field_value_or_error_serial(
    execution_context: "ExecutionContext",
    field_definition: "GraphQLField",
    field_nodes: List["FieldNode"],
    resolver: Callable,
    source: Any,
    info: "ResolveInfo",
) -> Union[Exception, Any]:
    """
    Serial version of resolve_field_value_or_error that ensures sequential
    execution without concurrent patterns. Uses sync default resolver when
    no custom resolver is defined to avoid await overhead.
    :param execution_context: instance of the query execution context
    :param field_definition: GraphQLField instance of the resolved field
    :param field_nodes: AST nodes related to the resolved field
    :param resolver: callable to use to resolve the field
    :param source: default root value or field parent value
    :param info: information related to the execution and the resolved field
    :type execution_context: ExecutionContext
    :type field_definition: GraphQLField
    :type field_nodes: List[FieldNode]
    :type resolver: Callable
    :type source: Any
    :type info: ResolveInfo
    :return: the resolved field value
    :rtype: Union[Exception, Any]
    """
    # pylint: disable=too-many-locals
    try:
        # Cache frequently accessed attributes
        raw_resolver = field_definition.raw_resolver
        is_default_resolver = raw_resolver is None
        schema = execution_context.schema
        variable_values = execution_context.variable_values
        context = execution_context.context
        field_node = field_nodes[0]  # Cache first field node (most common case)
        field_arguments = field_definition.arguments  # Cache arguments dict
        arguments_coercer = field_definition.arguments_coercer  # Cache coercer
        is_introspection = info.is_introspection  # Cache introspection flag
        
        # Compute directives first
        computed_directives = []
        for field_node_item in field_nodes:
            directive_nodes = field_node_item.directives
            if directive_nodes:  # Early exit if no directives
                computed_directives.extend(
                    compute_directive_nodes(
                        schema,
                        directive_nodes,
                        variable_values,
                    )
                )
        
        # Try to get the field value directly from the source first
        # Only use this optimization when:
        # 1. No directives are attached to the field
        # 2. Field has no arguments (to ensure argument validation isn't skipped)
        # 3. Using default resolver (to avoid bypassing custom resolver logic)
        # 4. skip_resolved_field_default_resolver is True (optimization enabled)
        field_value = None
        if (not computed_directives 
            and not field_arguments 
            and is_default_resolver
            and execution_context.schema.skip_resolved_field_default_resolver):
            field_value = _try_get_field_from_source(source, info.field_name)
        if field_value is not None:
            # Value already exists in source, return it
            # Still need to handle introspection directives if present
            if is_introspection and computed_directives:
                return await introspection_directives_executor(
                    field_value,
                    context,
                    info,
                    context_coercer=context,
                )
            return field_value
        
        has_directives = bool(computed_directives)
        
        # Fast path: default resolver without directives and not wrapped
        # Check if resolver is wrapped with resolver_executor or directive_executor at bake time
        resolver_is_wrapped = (
            isinstance(resolver, partial)
            and hasattr(resolver, 'func')
            and (resolver.func is resolver_executor or resolver.func is directive_executor)
        )
        
        # Optimize common case: default resolver, no directives, not wrapped
        if is_default_resolver and not resolver_is_wrapped and not has_directives:
            # Fast path: use sync default resolver directly
            coerced_args = await coerce_arguments(
                field_arguments,
                field_node,
                variable_values,
                context,
                coercer=arguments_coercer,
            )
            
            result = default_field_resolver_sync(
                source,
                coerced_args,
                context,
                info,
            )
            
            if is_introspection:
                return await introspection_directives_executor(
                    result,
                    context,
                    info,
                    context_coercer=context,
                )
            return result
        
        # Handle directives case
        resolver_to_use = resolver
        if has_directives:
            # Check if resolver is already wrapped with resolver_executor
            if not resolver_is_wrapped:
                resolver_to_use = partial(resolver_executor, resolver_to_use)
            
            resolver_to_use = wraps_with_directives(
                directives_definition=computed_directives,
                directive_hook="on_field_execution",
                func=resolver_to_use,
                is_resolver=False,  # Already wrapped with resolver_executor if needed
                with_default=True,
            )
        elif is_default_resolver and not resolver_is_wrapped:
            # Default resolver without directives: use sync version
            resolver_to_use = default_field_resolver_sync

        coerced_args = await coerce_arguments(
            field_arguments,
            field_node,
            variable_values,
            context,
            coercer=arguments_coercer,
        )
        
        # Determine if we can call synchronously
        # Check if resolver is wrapped with directive_executor (when directives are present)
        # directive_executor is async and always needs to be awaited
        is_wrapped_with_directive = (
            isinstance(resolver_to_use, partial)
            and hasattr(resolver_to_use, 'func')
            and resolver_to_use.func is directive_executor
        )
        is_wrapped_with_executor = (
            isinstance(resolver_to_use, partial)
            and hasattr(resolver_to_use, 'func')
            and resolver_to_use.func is resolver_executor
        )
        
        if has_directives or is_wrapped_with_directive or is_wrapped_with_executor:
            # Wrapped resolver (with directives or executor) - always await (wrapper handles sync/async)
            result = await resolver_to_use(
                source,
                coerced_args,
                context,
                info,
                context_coercer=context,
            )
        elif resolver_to_use is default_field_resolver_sync or is_sync_callable(resolver_to_use):
            # Sync resolver (default or custom) without directives - call directly without await
            result = resolver_to_use(
                source,
                coerced_args,
                context,
                info,
            )
        else:
            # Async resolver without directives - await it
            result = await resolver_to_use(
                source,
                coerced_args,
                context,
                info,
                context_coercer=context,
            )
        
        if is_introspection:
            return await introspection_directives_executor(
                result,
                context,
                info,
                context_coercer=context,
            )
        return result
    except Exception as e:  # pylint: disable=broad-except
        return e


async def resolve_field_serial(
    execution_context: "ExecutionContext",
    parent_type: "GraphQLObjectType",
    source: Any,
    field_nodes: List["FieldNode"],
    path: "Path",
    is_introspection_context: bool,
    field_definition: "GraphQLField",
    resolver: Callable,
    output_coercer: Callable,
) -> Any:
    """
    Serial version of resolve_field that ensures sequential execution without
    concurrent patterns.
    :param execution_context: instance of the query execution context
    :param parent_type: GraphQLObjectType of the field's parent
    :param source: default root value or field parent value
    :param field_nodes: AST nodes related to the resolved field
    :param path: the path traveled until this resolver
    :param field_definition: GraphQLField instance of the resolved field
    :param resolver: callable to use to resolve the field
    :param output_coercer: callable to use to coerce the resolved field value
    :param is_introspection_context: determines whether or not the resolved
    field is in a context of an introspection query
    :type execution_context: ExecutionContext
    :type parent_type: GraphQLObjectType
    :type source: Any
    :type field_nodes: List[FieldNode]
    :type path: Path
    :type field_definition: GraphQLField
    :type resolver: Callable
    :type output_coercer: Callable
    :type is_introspection_context: bool
    :return: the coerced resolved field value
    :rtype: Any
    """
    # pylint: disable=too-many-arguments
    # Early optimization: try to get field value from source before computing directives
    # Only applies when:
    # 1. No directives on field nodes (quick check without computing)
    # 2. Field has no arguments
    # 3. Using default resolver (no custom resolver)
    # 4. Optimization flag is enabled
    # 5. Not introspection context (introspection may need directive processing)
    if (not any(field_node.directives for field_node in field_nodes)
        and not field_definition.arguments
        and field_definition.raw_resolver is None
        and execution_context.schema.skip_resolved_field_default_resolver
        and not is_introspection_context):
        field_value = _try_get_field_from_source(source, field_definition.name)
        if field_value is not None:
            # Value found in source, build info and return coerced value
            info = build_resolve_info(
                execution_context,
                field_definition,
                field_nodes,
                parent_type,
                path,
                is_introspection_context,
            )
            # Optimization: skip coercion for scalars when type matches
            unwrapped_type = get_wrapped_type(field_definition.graphql_type)
            if (is_scalar_type(unwrapped_type)
                and _can_skip_scalar_coercion(field_value, unwrapped_type)):
                # Handle exceptions (field_value is already known to be not None)
                if isinstance(field_value, Exception):
                    return handle_field_error(
                        field_value,
                        field_nodes,
                        path,
                        field_definition.graphql_type,
                        execution_context,
                    )
                return field_value
            return await complete_value_catching_error_serial(
                field_value,
                info,
                execution_context,
                field_nodes,
                path,
                field_definition.graphql_type,
                output_coercer,
            )
    
    info = build_resolve_info(
        execution_context,
        field_definition,
        field_nodes,
        parent_type,
        path,
        is_introspection_context,
    )

    result = await resolve_field_value_or_error_serial(
        execution_context,
        field_definition,
        field_nodes,
        resolver,
        source,
        info,
    )
    
    # Optimization: skip coercion for scalars when type matches
    unwrapped_type = get_wrapped_type(field_definition.graphql_type)
    if (is_scalar_type(unwrapped_type)
        and _can_skip_scalar_coercion(result, unwrapped_type)):
        # Handle exceptions and non-null validation
        if isinstance(result, Exception):
            return handle_field_error(
                result,
                field_nodes,
                path,
                field_definition.graphql_type,
                execution_context,
            )
        # Non-null validation
        if field_definition.graphql_type.is_non_null_type and result is None:
            return handle_field_error(
                ValueError("Non-null field cannot be null"),
                field_nodes,
                path,
                field_definition.graphql_type,
                execution_context,
            )
        return result
    
    return await complete_value_catching_error_serial(
        result,
        info,
        execution_context,
        field_nodes,
        path,
        field_definition.graphql_type,
        output_coercer,
    )
