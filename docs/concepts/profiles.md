# Profiles

A **profile** is a named set of inference settings for one kind of task. It
selects a model and a mode. It does not select tools: lackpy has no tool kits.
A program does file access through `pathlib.Path`, under the
[path policy](paths.md).

## Define a profile

Define a profile as a `[profiles.<name>]` table in `.lackpy/config.toml`:

```toml
[profiles.careful]
model = "ollama/qwen2.5-coder:7b"
mode = "spm"
temperature = 0.2

[profile]
default = "careful"
```

Use it by name:

```bash
lackpy -c "count the TODO lines under src" --profile careful
```

With no `--profile`, lackpy uses `[profile] default` if a `[profiles.<name>]`
table defines it. Otherwise it uses the service configuration.

## Keys

| Key | Meaning |
| --- | --- |
| `model` | The model for this profile, as a `<provider>/<model>` string. |
| `mode` | The inference mode, for example `1-shot` or `spm`. |
| `order` | The inference tiers to try, in order. |
| `temperature` | The sampling temperature. |
| `rules` | Extra rule names. Parsed, but the service does not apply them yet. |
| `language` | The language axis. The default is `restricted-python`. |
| `execution` | The execution axis. The default is `one-shot`. |
| `sandbox`, `policy` | Parsed, but not enforced. |

A key that is not in this table is an error, so a misspelt key does not pass
without notice.

## Refused profiles

These profiles give a failed result. lackpy does not run the program.

| Profile | Error |
| --- | --- |
| It sets `tools` or `extra_tools` | `tool kits were removed in lackpy stepy100; use pathlib.Path (docs/concepts/paths.md)` |
| It is a list of tool names, a dict, or an unknown name such as `read_file,find_files` | the same |
| Its name is a `.kit` or `.profile` file, not a table | the same, and the file name |
| It sets `mode = "staged-dsl"` | the mode is disabled until it migrates to the path policy; use `1-shot` or `spm` |

lackpy refuses mode `staged-dsl` on every route: a profile, `lackpy --mode`
and `[inference] mode`. The refusal waits for a migration to the path policy.

A profile with `execution = "literate"` runs a [literate document](literate.md).
It uses the same path policy as a program.

## Python API

```python title="Python API: a profile object"
from lackpy.profiles import Profile

result = await svc.delegate("read the file notes.md",
                            profile=Profile(model="ollama/qwen2.5-coder:7b", mode="1-shot"))
```

## The toolbox

The toolbox is the registry of tool providers from the kit era. No run uses it.
`lackpyctl toolbox` and `lackpyctl profile` still show it, for inspection only.
