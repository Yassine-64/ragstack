from pathlib import Path

from ragstack.infrastructure.parsers.text import TextParser


async def test_markdown_sections_become_chunks_with_paths(tmp_path: Path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nintro para\n\n## Part A\n\nalpha\n\nbeta\n\n## Part B\n\ngamma\n")
    parsed = await TextParser(max_chars=1000).parse(p)
    assert [c.section_path for c in parsed.chunks] == [["Title"], ["Title", "Part A"], ["Title", "Part B"]]
    assert parsed.chunks[1].text == "alpha\n\nbeta"
    assert parsed.media_type == "text/markdown"
