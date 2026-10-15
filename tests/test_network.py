import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import requests

from agent import check_api, net
from agent.llm import OpenAICompatClient, get_client


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict("os.environ", {}, clear=True)
        self.cfg_patch = patch.object(net, "_CONFIG", {})
        self.env_patch.start()
        self.cfg_patch.start()
        self.addCleanup(self.cfg_patch.stop)
        self.addCleanup(self.env_patch.stop)
        self.response = MagicMock(status_code=200)
        self.response.json.return_value = {"choices": [{"message": {"content": "正常"}}]}

    def session(self):
        session = MagicMock()
        session.__enter__.return_value = session
        session.request.return_value = self.response
        return session

    def test_proxy_mode_never_attempts_direct(self):
        net.configure({"AGENT_NETWORK_MODE": "proxy", "AGENT_PROXY": "http://user:secret@proxy.test:8080"})
        session = self.session()
        output = io.StringIO()
        with patch.object(net.requests, "Session", return_value=session), redirect_stdout(output):
            net.post("https://api.test/v1/chat/completions", timeout=25)
        self.assertEqual(session.request.call_count, 1)
        self.assertFalse(session.trust_env)
        self.assertEqual(session.request.call_args.kwargs["proxies"]["https"], "http://user:secret@proxy.test:8080")
        self.assertIn("代理 POST", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

    def test_direct_ignores_configured_proxy(self):
        net.configure({"AGENT_NETWORK_MODE": "direct", "AGENT_PROXY": "http://proxy.test:8080"})
        session = self.session()
        with patch.object(net.requests, "Session", return_value=session):
            net.get("https://api.test", timeout=25)
        self.assertEqual(session.request.call_args.kwargs["proxies"], {})

    def test_auto_retries_proxy_after_connection_failure(self):
        net.configure({"AGENT_NETWORK_MODE": "auto", "AGENT_PROXY": "http://proxy.test:8080"})
        direct, proxy = self.session(), self.session()
        direct.request.side_effect = requests.ConnectTimeout("credentials must not be logged")
        output = io.StringIO()
        with patch.object(net.requests, "Session", side_effect=[direct, proxy]), redirect_stdout(output):
            net.get("https://api.test", timeout=25)
        self.assertEqual(direct.request.call_count, 1)
        self.assertEqual(proxy.request.call_count, 1)
        self.assertIn("ConnectTimeout", output.getvalue())
        self.assertNotIn("credentials", output.getvalue())

    def test_proxy_mode_requires_address(self):
        net.configure({"AGENT_NETWORK_MODE": "proxy"})
        with patch.object(net.requests, "Session") as session:
            with self.assertRaises(ValueError):
                net.get("https://api.test")
            session.assert_not_called()

    def test_environment_overrides_file_settings(self):
        net.configure({"AGENT_NETWORK_MODE": "auto", "AGENT_PROXY": "http://old.test:8080"})
        with patch.dict("os.environ", {"AGENT_NETWORK_MODE": "proxy", "AGENT_PROXY": "http://new.test:8080"}):
            self.assertEqual(net.network_mode(), "proxy")
            self.assertEqual(net.resolve_proxy()["https"], "http://new.test:8080")

    def test_timeout_split_and_explicit_selfcheck_override(self):
        net.configure({"AGENT_CONNECT_TIMEOUT_S": "7", "AGENT_LLM_TIMEOUT_S": "90"})
        self.assertEqual(net.request_timeout(), (7.0, 90.0))
        client = OpenAICompatClient("test", "https://api.test/v1", "test-model")
        with patch.object(net, "post", return_value=self.response) as post:
            self.assertEqual(client.chat([], timeout=25), "正常")
            self.assertEqual(post.call_args.kwargs["timeout"], (7.0, 25.0))

    def test_alternate_env_config_is_used_for_network(self):
        env = {"AGENT_LLM_KIND": "openai", "AGENT_API_KEY": "test",
               "AGENT_BASE_URL": "https://api.test/v1", "AGENT_MODEL": "test-model",
               "AGENT_NETWORK_MODE": "proxy", "AGENT_PROXY": "http://alternate.test:8080"}
        get_client(env)
        self.assertEqual(net.resolve_proxy()["https"], env["AGENT_PROXY"])

    def test_http_error_summary_preserves_status_without_credentials(self):
        response = MagicMock(status_code=403)
        exc = requests.HTTPError("private details", response=response)
        self.assertEqual(net.error_summary(exc), "HTTPError HTTP 403")

    def test_selfcheck_passes_short_timeout_and_mode(self):
        env = {"AGENT_LLM_KIND": "openai", "AGENT_BASE_URL": "https://api.test/v1", "AGENT_MODEL": "test"}
        client = MagicMock()
        client.chat.return_value = "正常"
        output = io.StringIO()
        with patch("sys.argv", ["check_api.py", "--network-mode", "direct", "--timeout", "12"]), \
                patch.object(check_api, "load_env", return_value=env), \
                patch.object(check_api, "get_client", return_value=client), redirect_stdout(output):
            self.assertEqual(check_api.main(), 0)
        self.assertEqual(client.chat.call_args.kwargs["timeout"], 12)
        self.assertIn("网络模式: direct", output.getvalue())


if __name__ == "__main__":
    unittest.main()
