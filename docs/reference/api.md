# Python API Reference

## Service

::: lackpy.service.LackpyService
    options:
      members:
        - validate
        - generate
        - run_program
        - delegate
        - create
        - profile_info
        - profile_list
        - profile_create
        - toolbox_list
        - docs_index
        - resolve_doc

---

## Validation

::: lackpy.lang.validator.validate

::: lackpy.lang.validator.ValidationResult

---

## Grading

::: lackpy.lang.grader.Grade

::: lackpy.lang.grader.compute_grade

---

## Grammar constants

::: lackpy.lang.grammar
    options:
      members:
        - ALLOWED_NODES
        - FORBIDDEN_NODES
        - FORBIDDEN_NAMES
        - ALLOWED_BUILTINS

---

## Toolbox

::: lackpy.tools.toolbox.Toolbox
    options:
      members:
        - register_provider
        - register_tool
        - resolve
        - resolve_docs
        - docs_index
        - list_tools
        - format_description

::: lackpy.tools.toolbox.ToolSpec

::: lackpy.tools.toolbox.ArgSpec

---

## Runner

::: lackpy.run.runner.RestrictedRunner
    options:
      members:
        - run

::: lackpy.run.base.ExecutionResult

---

## Trace

::: lackpy.run.trace.Trace

::: lackpy.run.trace.TraceEntry

---

## File access

See [Concepts: File access](../concepts/paths.md).

::: lackpy.run.paths.sources.load_policy

::: lackpy.run.paths.rules.PathPolicy

::: lackpy.run.paths.rules.PathRule

::: lackpy.run.paths.rules.parse_rule

::: lackpy.run.paths.proxy.make_path_class

---

## Inference

::: lackpy.infer.dispatch.InferenceDispatcher
    options:
      members:
        - generate

::: lackpy.infer.dispatch.GenerationResult

::: lackpy.infer.prompt.build_system_prompt

::: lackpy.infer.prompt.format_params_description

::: lackpy.infer.sanitize.sanitize_output

---

## Built-in rules

::: lackpy.lang.rules.no_loops

::: lackpy.lang.rules.max_depth

::: lackpy.lang.rules.max_calls

::: lackpy.lang.rules.no_nested_calls

---

## Policy

::: lackpy.policy.layer.PolicyLayer
    options:
      members:
        - add_source
        - resolve

::: lackpy.policy.layer.PolicySource

::: lackpy.policy.types.PolicyResult
    options:
      members:
        - replace

::: lackpy.policy.types.PolicyContext

::: lackpy.policy.types.ToolConstraints

::: lackpy.policy.types.Principal

::: lackpy.policy.types.ModelSpec

---

## Policy sources

::: lackpy.policy.sources.tools.ToolsPolicySource
    options:
      members:
        - resolve

::: lackpy.policy.sources.kibitzer.KibitzerPolicySource
    options:
      members:
        - resolve

::: lackpy.policy.sources.umwelt.UmweltPolicySource
    options:
      members:
        - resolve

---

## Configuration

::: lackpy.config.LackpyConfig

::: lackpy.config.load_config
