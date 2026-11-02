import unittest

from tests.helpers import fixture_df
from voc.agent import VocAgent, build_demo_report
from voc.analysis import Corpus
from voc.llm import ScriptedLLM


class TestAgentLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus(fixture_df())

    def test_scripted_agent_report_passes_audit(self):
        ov = self.corpus.overview()
        final = (f"Korpus berisi {ov['n_reviews']} ulasan dan {ov['predicted_negative_pct']}% "
                 f"di antaranya diprediksi negatif.")
        llm = ScriptedLLM([
            ScriptedLLM.tool("get_overview", {}),
            ScriptedLLM.tool("compare_categories", {}),
            ScriptedLLM.text(final),
        ])
        res = VocAgent(self.corpus, llm).ask("Gambaran umum korpus?")
        self.assertEqual(res["tools_used"], ["get_overview", "compare_categories"])
        self.assertEqual(res["report"], final)
        self.assertTrue(res["audit_passed"], res["audit"])

    def test_unknown_review_tool_returns_error_not_crash(self):
        agent = VocAgent(self.corpus, ScriptedLLM([ScriptedLLM.text("x")]))
        out = agent._run_tool("get_review", {"review_id": "R99999"})
        self.assertIn("error", out)

    def test_demo_report_passes_audit_on_fixture(self):
        report, ev = build_demo_report(self.corpus)
        self.assertTrue(ev["audit"].passed, ev["audit"])
        self.assertIn("get_overview", ev["tools_used"])


if __name__ == "__main__":
    unittest.main()
