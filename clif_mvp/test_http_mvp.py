import unittest

from clif_mvp import http_mvp


class HTTPMVPTests(unittest.TestCase):
    def test_users_forbidden_flow_is_observable(self):
        with http_mvp.ToyServer() as server:
            flow = http_mvp.execute_http(server.base_url, "users", {
                "requester": "alice", "user_id": "bob", "scope": "all"
            })
        self.assertTrue(flow["forbidden"])
        self.assertEqual(flow["source_domain"], "cross_user")
        self.assertEqual(flow["sink"], "http_body")

    def test_normal_users_request_is_not_forbidden(self):
        with http_mvp.ToyServer() as server:
            flow = http_mvp.execute_http(server.base_url, "users", {
                "requester": "alice", "user_id": "alice", "scope": "self"
            })
        self.assertFalse(flow["forbidden"])

    def test_program_local_ucb_state_starts_empty(self):
        import random
        learner_a = http_mvp.UCBPolicy(random.Random(1))
        learner_b = http_mvp.UCBPolicy(random.Random(2))
        learner_a.update("mutate_scope", 1.0)
        self.assertEqual(learner_a.n["mutate_scope"], 1)
        self.assertEqual(learner_b.n["mutate_scope"], 0)

    def test_reports_and_tokens_have_real_http_security_flow(self):
        with http_mvp.ToyServer() as server:
            report = http_mvp.execute_http(server.base_url, "reports", {
                "requester": "alice", "report_id": "2", "mode": "debug", "page": "0"
            })
            token = http_mvp.execute_http(server.base_url, "tokens", {
                "requester": "alice", "token": "guest-b", "role": "admin"
            })
        self.assertTrue(report["forbidden"])
        self.assertTrue(token["forbidden"])


if __name__ == "__main__":
    unittest.main()
