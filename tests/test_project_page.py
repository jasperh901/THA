from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _ResourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.resources: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.resources.update(
            value
            for key, value in attrs
            if key in {"href", "src"} and value and value.startswith("assets/")
        )


class ProjectPageTest(unittest.TestCase):
    def test_site_uses_only_public_assets(self) -> None:
        parser = _ResourceParser()
        parser.feed((ROOT / "index.html").read_text(encoding="utf-8"))
        allowed = {
            "assets/style.css",
            "assets/app.js",
            "assets/motivation.png",
            "assets/architecture.png",
            "assets/qualitative.png",
            "assets/module-necessity-mae.png",
            "assets/module-necessity-corr.png",
            "assets/query-resampler-reliability.png",
            "assets/query-resampler-rank.png",
            "assets/query-resampler-similarity.png",
        }

        self.assertEqual(parser.resources, allowed)
        files = {str(path.relative_to(ROOT)) for path in (ROOT / "assets").iterdir()}
        self.assertEqual(files, allowed)
        results = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
        for dataset in ("MOSEI", "SIMS-V2", "CHERMA"):
            self.assertIn(dataset, results)


if __name__ == "__main__":
    unittest.main()
