"""End-to-end tests: full documents through StreamingDriver."""

import os

import pytest

from lackpy.interpreters.literate.kernel import (
    LightweightKernel,
    NoRecoveryHandler,
    StreamingDriver,
)
from lackpy.interpreters.literate.kernel.formats import from_notebook, render_markdown, to_notebook
from lackpy.interpreters.literate.parser import Frontmatter


@pytest.fixture
def driver():
    kernel = LightweightKernel()
    return StreamingDriver(kernel=kernel, recovery=NoRecoveryHandler())


class TestFullDocumentExecution:
    @pytest.mark.asyncio
    async def test_analysis_document(self, driver):
        doc = (
            "---\necho: true\n---\n\n# Report\n\n"
            "```lackpy @hidden\ndata = [1, 2, 3, 4, 5]\ntotal = sum(data)\n```\n\n"
            "The dataset has {len(data)} items totaling {total}.\n\n"
            "```lackpy\nmean = total / len(data)\nprint(f\"Mean: {mean:.1f}\")\n```\n\n"
            "Average value: {mean:.1f}\n"
        )
        for line in doc.split("\n"):
            await driver.feed(line + "\n")
        await driver.flush()

        output = driver.rendered_output
        assert "5 items" in output
        assert "totaling 15" in output
        assert "Mean: 3.0" in output
        assert "Average value: 3.0" in output

    @pytest.mark.asyncio
    async def test_document_with_write(self, tmp_path, driver):
        os.chdir(tmp_path)
        from lackpy.interpreters.literate.tools import write_file
        driver._kernel._namespace["write_file"] = write_file

        doc = (
            "```lackpy @write(hello.py)\nprint(\"hello world\")\n```\n\n"
            "```lackpy @hidden\nimport os\nexists = os.path.exists(\"hello.py\")\n```\n\n"
            "File created: {exists}\n"
        )
        await driver.feed(doc)
        await driver.flush()

        assert "True" in driver.rendered_output
        assert (tmp_path / "hello.py").exists()


class TestNotebookRoundTrip:
    @pytest.mark.asyncio
    async def test_execute_export_reimport(self, driver):
        doc = "```lackpy @hidden\nx = 42\n```\n\nThe answer: {x}\n"
        await driver.feed(doc)
        await driver.flush()

        fm = Frontmatter()
        nb = to_notebook(driver.execution_log, fm)
        recovered_fm, recovered_cells = from_notebook(nb)

        assert len(recovered_cells) == 2
        assert recovered_cells[0].cell_type == "hidden"
        assert recovered_cells[1].cell_type == "prose"

        kernel2 = LightweightKernel()
        driver2 = StreamingDriver(kernel=kernel2, recovery=NoRecoveryHandler())
        for cell in recovered_cells:
            await driver2._execute_cells([cell])
        assert "42" in driver2.rendered_output


class TestMarkdownRoundTrip:
    @pytest.mark.asyncio
    async def test_render_clean_markdown(self, driver):
        doc = "```lackpy @hidden\nx = 1\n```\n\nValue: {x}\n"
        await driver.feed(doc)
        await driver.flush()

        md = render_markdown(driver.execution_log, Frontmatter())
        assert "```lackpy @hidden" in md
        assert "x = 1" in md
        assert "Value: {x}" in md
