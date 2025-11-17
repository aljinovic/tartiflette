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
from tartiflette.utils.callables import is_valid_coroutine
from tartiflette.utils.directives import (
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

        result = await resolver(
            source,
            await coerce_arguments(
                field_definition.arguments,
                field_nodes[0],
                execution_context.variable_values,
                execution_context.context,
                coercer=field_definition.arguments_coercer,
            ),
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
        computed_directives = []
        for field_node in field_nodes:
            computed_directives.extend(
                compute_directive_nodes(
                    execution_context.schema,
                    field_node.directives,
                    execution_context.variable_values,
                )
            )

        # Check if this is the default resolver (no custom resolver defined)
        is_default_resolver = field_definition.raw_resolver is None
        
        # Check if the raw resolver is async (before wrapping)
        raw_resolver_is_async = False
        if not is_default_resolver and field_definition.raw_resolver:
            raw_resolver_is_async = is_valid_coroutine(field_definition.raw_resolver)
        
        # Determine which resolver to use
        # The resolver parameter passed in is the resolver extracted from field_definition.resolver
        # which may already be wrapped with resolver_executor at bake time (via wraps_with_directives).
        # We need to check if it's wrapped to know whether to pass context_coercer.
        from functools import partial
        resolver_from_bake_is_wrapped = (
            isinstance(resolver, partial) 
            and hasattr(resolver, 'func') 
            and resolver.func is resolver_executor
        )
        
        # Check for runtime directives
        has_directives = bool(computed_directives)
        
        # Always use the resolver passed in (extracted from field_definition.resolver.keywords['resolver'])
        # At bake time, this resolver is wrapped with resolver_executor via wraps_with_directives(is_resolver=True)
        # So it should always be wrapped, except in edge cases
        # For default resolvers, we can optimize by using sync version if not wrapped and no directives
        resolver_to_use = resolver
        
        # Only replace with sync default resolver if:
        # 1. It's a default resolver (raw_resolver is None)
        # 2. The extracted resolver is NOT wrapped (shouldn't happen normally, but handle edge case)
        # 3. No runtime directives
        if is_default_resolver and not resolver_from_bake_is_wrapped and not has_directives:
            # Use sync default resolver for optimization
            resolver_to_use = default_field_resolver_sync

        if has_directives:
            # Ensure resolver_executor is applied when wrapping with directives
            # wraps_with_directives only wraps with resolver_executor if func is not a partial,
            # but we need it wrapped to handle context_coercer. So we wrap it manually first.
            # We always wrap with resolver_executor when directives are present to ensure
            # context_coercer is properly handled (popped before reaching the resolver).
            from functools import partial
            
            # Check if resolver_to_use is already wrapped with resolver_executor
            needs_wrapper = True
            if isinstance(resolver_to_use, partial) and hasattr(resolver_to_use, 'func'):
                if resolver_to_use.func is resolver_executor:
                    # Already wrapped with resolver_executor - don't double-wrap
                    needs_wrapper = False
            
            if needs_wrapper:
                # Always wrap with resolver_executor when directives are present
                # This ensures context_coercer is handled even if resolver is already a partial
                resolver_to_use = partial(resolver_executor, resolver_to_use)
            
            resolver_to_use = wraps_with_directives(
                directives_definition=computed_directives,
                directive_hook="on_field_execution",
                func=resolver_to_use,
                is_resolver=False,  # Already wrapped with resolver_executor above (or was already wrapped)
                with_default=True,
            )

        coerced_args = await coerce_arguments(
            field_definition.arguments,
            field_nodes[0],
            execution_context.variable_values,
            execution_context.context,
            coercer=field_definition.arguments_coercer,
        )
        
        # Call resolver - match regular executor behavior exactly
        # The regular executor ALWAYS passes context_coercer to ALL resolvers (line 80)
        # resolver_executor wrapper will pop it if not needed
        # Only sync default resolver without directives and without wrapper can be called synchronously
        # All other resolvers (including wrapped ones) need await and context_coercer
        is_sync_default_unwrapped = (
            resolver_to_use is default_field_resolver_sync and
            not has_directives
        )
        
        if is_sync_default_unwrapped:
            # Sync default resolver - no await, no context_coercer
            result = resolver_to_use(
                source,
                coerced_args,
                execution_context.context,
                info,
            )
        else:
            # ALL other resolvers: always pass context_coercer (matches regular executor exactly)
            # This includes:
            # - Custom resolvers (wrapped with resolver_executor at bake time)
            # - Default resolver with directives (wrapped with resolver_executor)
            # - Any async resolver
            # - Any resolver wrapped with resolver_executor
            # resolver_executor wrapper will pop context_coercer if not needed
            result = await resolver_to_use(
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
