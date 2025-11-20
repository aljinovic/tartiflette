from functools import partial
from typing import Any, Callable, List, Union

from tartiflette.coercers.arguments import coerce_arguments
from tartiflette.coercers.outputs.common import (
    complete_value_catching_error,
    complete_value_catching_error_serial,
)
from tartiflette.execution.types import build_resolve_info
from tartiflette.resolver.default import default_field_resolver_sync
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
        computed_directives = []
        for field_node in field_nodes:
            computed_directives.extend(
                compute_directive_nodes(
                    execution_context.schema,
                    field_node.directives,
                    execution_context.variable_values,
                )
            )

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
    info = build_resolve_info(
        execution_context,
        field_definition,
        field_nodes,
        parent_type,
        path,
        is_introspection_context,
    )

    return await complete_value_catching_error(
        await resolve_field_value_or_error(
            execution_context,
            field_definition,
            field_nodes,
            resolver,
            source,
            info,
        ),
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
        
        # Optimize directive computation: use list comprehension and early exit
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
    info = build_resolve_info(
        execution_context,
        field_definition,
        field_nodes,
        parent_type,
        path,
        is_introspection_context,
    )

    return await complete_value_catching_error_serial(
        await resolve_field_value_or_error_serial(
            execution_context,
            field_definition,
            field_nodes,
            resolver,
            source,
            info,
        ),
        info,
        execution_context,
        field_nodes,
        path,
        field_definition.graphql_type,
        output_coercer,
    )
