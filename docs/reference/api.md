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
        - kit_info
        - kit_list
        - kit_create
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

::: lackpy.kit.toolbox.Toolbox
    options:
      members:
        - register_provider
        - register_tool
        - resolve
        - resolve_docs
        - docs_index
        - list_tools
        - format_description

::: lackpy.kit.toolbox.ToolSpec

::: lackpy.kit.toolbox.ArgSpec

---

## Kit registry

::: lackpy.kit.registry.resolve_kit

::: lackpy.kit.registry.ResolvedKit

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

::: lackpy.run.trace.make_traced

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

## Configuration

::: lackpy.config.LackpyConfig

::: lackpy.config.load_config
