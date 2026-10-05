import io
import json
import unittest
from unittest.mock import patch

import pdf_processor
import rag


def make_pdf(pages: list[str]) -> bytes:
    """Build a small real PDF (test-only; needs reportlab)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


PAGES = [
    "Solar panels convert sunlight into electricity using photovoltaic cells.\nThe efficiency of modern panels is about twenty percent.",
    "Wind turbines generate power when wind spins their blades.\nOffshore wind farms produce more energy than onshore farms.",
    "Batteries store energy for later use.\nLithium ion batteries are common in electric vehicles.",
]


class PdfTests(unittest.TestCase):
    def test_extract_pages_and_numbers(self):
        pages = pdf_processor.extract_pages(make_pdf(PAGES))
        self.assertEqual([n for n, _ in pages], [1, 2, 3])
        self.assertIn("photovoltaic", pages[0][1])

    def test_chunking_overlap_and_short_text(self):
        text = " ".join(f"w{i}" for i in range(300))
        chunks = pdf_processor.chunk_text(text, words=100, overlap=20)
        self.assertGreater(len(chunks), 3)
        self.assertEqual(chunks[0].split()[-20:], chunks[1].split()[:20])
        self.assertEqual(pdf_processor.chunk_text("only a few words"), ["only a few words"])
        self.assertEqual(pdf_processor.chunk_text(""), [])

    def test_blank_pdf_has_no_chunks(self):
        self.assertEqual(pdf_processor.process_pdf("blank.pdf", make_pdf([""])), [])

    def test_garbage_file_raises(self):
        with self.assertRaises(Exception):
            pdf_processor.process_pdf("bad.pdf", b"not a pdf")


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.chunks = pdf_processor.process_pdf("energy.pdf", make_pdf(PAGES))
        self.index = rag.BM25Index(self.chunks)

    def test_finds_right_page(self):
        top = self.index.search("How do wind turbines generate power?", k=1)[0][1]
        self.assertEqual(top["page"], 2)
        top = self.index.search("lithium batteries electric vehicles", k=1)[0][1]
        self.assertEqual(top["page"], 3)

    def test_no_match_returns_empty(self):
        self.assertEqual(self.index.search("zebra quantum", k=3), [])

    def test_empty_index_is_safe(self):
        self.assertEqual(rag.BM25Index([]).search("anything"), [])

    def test_extractive_answer_has_citation(self):
        hits = self.index.search("What is the efficiency of solar panels?")
        text = rag.answer_extractive("What is the efficiency of solar panels?", hits)
        self.assertIn("twenty percent", text)
        self.assertIn("p. 1", text)

    def test_extractive_no_hits_message(self):
        self.assertIn("couldn't find", rag.answer_extractive("zebra", []))


class AnswerModeTests(unittest.TestCase):
    def setUp(self):
        chunks = pdf_processor.process_pdf("energy.pdf", make_pdf(PAGES))
        self.hits = rag.BM25Index(chunks).search("solar efficiency")

    def test_no_key_uses_extractive(self):
        _, mode = rag.answer("solar efficiency", self.hits, api_key="")
        self.assertEqual(mode, "extractive")

    @patch("rag.urllib.request.urlopen")
    def test_ai_mode_with_key(self, urlopen):
        resp = urlopen.return_value.__enter__.return_value
        resp.read.return_value = json.dumps({"content": [{"type": "text", "text": "About 20% [energy.pdf p.1]"}]}).encode()
        text, mode = rag.answer("solar efficiency", self.hits, api_key="sk-test")
        self.assertEqual((text, mode), ("About 20% [energy.pdf p.1]", "AI"))
        sent = urlopen.call_args[0][0]
        self.assertEqual(sent.get_header("X-api-key"), "sk-test")

    @patch("rag.urllib.request.urlopen", side_effect=OSError("network down"))
    def test_ai_failure_falls_back(self, _):
        text, mode = rag.answer("solar efficiency", self.hits, api_key="sk-test")
        self.assertEqual(mode, "extractive")
        self.assertIn("AI unavailable", text)


if __name__ == "__main__":
    unittest.main()
