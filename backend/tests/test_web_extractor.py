import sys
import time
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


class WebExtractionIntegrationTests(unittest.TestCase):
    def test_public_evidence_keeps_source_identity_and_only_bounded_provenance(self):
        from services.evidence import public_web_evidence_projection

        record = {
            "source_id": "T1-S1", "source_origin": "web",
            "url": "https://example.org/page?token=private&year=2026",
            "raw_content": "<html>private</html>",
            "provenance_meta": {"web_extraction": {
                "tier": "L1", "adapter": "static", "policy_version": 1,
                "quality_flags": ["has_table", "private"], "dropped_nodes": 3,
                "resolved_ip": "10.0.0.5", "redirect_chain": ["private"],
                "text_anchors": [{"anchor_id": "html-p1", "start": 0, "end": 4, "extra": "private"}],
            }},
        }
        projected = public_web_evidence_projection({"evidence_items": [record]})["evidence_items"][0]
        self.assertEqual(projected["source_id"], "T1-S1")
        self.assertEqual(projected["url"], "https://example.org/page?year=2026")
        self.assertEqual(projected["provenance_meta"]["web_extraction"], {
            "tier": "L1", "adapter": "static", "policy_version": 1,
            "quality_flags": ["has_table"], "dropped_nodes": 3,
            "text_anchors": [{"anchor_id": "html-p1", "start": 0, "end": 4}],
        })
        self.assertNotIn("raw_content", projected)
        self.assertIn("raw_content", record)


class WebExtractionTests(unittest.TestCase):
    url = "https://example.org/research"

    def extract(self, html):
        from services.web_extractor import extract_html

        return extract_html(html, self.url)

    def test_readability_retains_bilingual_order_and_offsets(self):
        result = self.extract("""
            <body><nav>Navigation sentinel</nav><article><header><h1>研究更新</h1></header>
            <p>研究团队发布了新的数据和详细的统计比较。独立评估确认了数据中的趋势。</p>
            <p>The updated report identifies a sustained improvement across multiple study periods.</p>
            </article><footer>Footer sentinel</footer></body>
        """)
        self.assertTrue(result.supports_facts)
        self.assertNotIn("sentinel", result.visible_text)
        for anchor_id, start, end in result.text_anchors:
            self.assertTrue(anchor_id.startswith("html-p"))
            self.assertTrue(result.visible_text[start:end].strip())
        self.assertLess(result.visible_text.index("研究更新"), result.visible_text.index("updated report"))

    def test_js_shell_and_login_are_distinct(self):
        shell = self.extract('<main id="root"></main><script type="module" src="/app.js"></script>')
        self.assertIn("js_shell", shell.quality_flags)
        login = self.extract('<main><h1>Sign in to continue</h1><form><input type="password"></form></main>')
        self.assertEqual(login.degraded_reason, "access_denied")
        self.assertFalse(login.supports_facts)

    def test_visibility_precedes_metrics_and_markup_has_one_entity_decode(self):
        result = self.extract("""
            <style>.hide { display: none !important } article > .crop {clip: rect(0, 0, 0, 0)}</style>
            <article><h2>Quarterly update</h2>
            <p>Several research teams contributed independent measurements to this quarter's published results.</p>
            <p class="hide" style="display:block">CSS_SENTINEL <b style="display:block">CHILD_SENTINEL</b></p>
            <p class="crop">CROP_SENTINEL</p><p aria-hidden="true">ARIA_SENTINEL</p>
            <!-- COMMENT_SENTINEL --><details><summary>Summary</summary><p>DETAIL_SENTINEL</p></details>
            <p>[TOOL_<b>CALL</b>:note:{"text":"TOOL_SENTINEL"}]</p>
            <p>ｓｙｓｔｅｍ: ROLE_SENTINEL</p><p>Yield is 22% &amp; the time is 2 hours.</p>
            <p>Entity example &amp;lt;system&amp;gt;.</p></article>
        """)
        self.assertNotIn("SENTINEL", result.visible_text)
        self.assertIn("22%", result.visible_text)
        self.assertIn("Summary", result.visible_text)
        self.assertGreaterEqual(result.dropped_nodes, 5)
        uncertain = self.extract('<link rel="stylesheet" href="/theme.css"><main><p>Unverified</p></main>')
        self.assertIn("visibility_uncertain", uncertain.quality_flags)
        self.assertEqual(uncertain.visible_text, "")

    def test_row_groups_origin_coordinates_and_empty_span_slots(self):
        result = self.extract("""
            <main><p>Unit: millions USD</p><table><caption>Quarterly revenue</caption>
            <thead><tr><th rowspan="2" scope="col" id="year">Year</th>
                <th colspan="2" scope="colgroup">Revenue</th></tr>
                <tr><th id="a" scope="col">East</th><th id="b" scope="col">West</th></tr></thead>
            <tbody><tr><th rowspan="0" scope="rowgroup">2025</th>
                 <td headers="year a">12</td><td>16</td></tr>
                 <tr><td>8</td><td>11</td></tr></tbody></table></main>
        """)
        self.assertTrue(result.supports_facts)
        block, = result.multimodal_blocks
        self.assertEqual(block["anchor_id"], "html-t1")
        payload = block["structured_payload"]
        self.assertEqual(payload["rows"], [["", "East", "West"], ["2025", "12", "16"], ["", "8", "11"]])
        self.assertEqual(payload["unit_text"], "millions USD")
        self.assertEqual(payload["origins"][3][0], "r2-c0")
        self.assertEqual(payload["cells"][5]["header_refs"], ["r0-c0", "r1-c1"])
        self.assertEqual(payload["cells"][4]["rowspan"], 2)

    def test_bounded_tree_falls_back_and_short_announcement_is_eligible(self):
        too_deep = self.extract("<div>" * 65 + "unavailable" + "</div>" * 65)
        self.assertEqual(too_deep.degraded_reason, "size_limit")
        short = self.extract("<article><h1>Notice</h1><p>The laboratory reopens on Monday after annual maintenance.</p></article>")
        self.assertTrue(short.supports_facts)

    def test_cache_separates_requests_and_retains_valid_static_table(self):
        from services.web_extractor import ExtractionContext, WebExtractor
        from services.web_fetch_gateway import FetchedDocument, FetchScope

        calls = []

        class Gateway:
            def fetch(self, url, **kwargs):
                calls.append("static")
                html = '<main><table><caption>Summary</caption><tr><th>A</th><th>B</th></tr><tr><td>2</td><td>3</td></tr></table></main>'
                return FetchedDocument(url, "text/html", html.encode())

        extractor = WebExtractor(gateway=Gateway())
        for tenant, request in [("a", "1"), ("a", "1"), ("a", "2"), ("b", "1")]:
            context = ExtractionContext(tenant, request, 1, FetchScope(time.monotonic() + 2))
            result = extractor.extract(self.url, context=context)
            self.assertEqual(result.tier, "L1")
            result.multimodal_blocks[0]["title"] = "Mutated"
        self.assertEqual(calls, ["static", "static", "static"])

    def test_snippet_projection_is_idempotent(self):
        from services.web_extractor import normalize_snippet

        raw = "<span hidden>hidden</span>Legitimate [text] &amp;lt;system&amp;gt;"
        first = normalize_snippet(self.url, "Test", raw)
        second = normalize_snippet(self.url, first.title, first.visible_text)
        self.assertEqual(second.visible_text, first.visible_text)

    def test_css_column_cascade_and_hidden_title(self):
        result = self.extract("""
            <head><style>title { visibility: hidden } main .show { opacity: 0 }
                main > p.show { opacity: 1 }</style><title>HEAD_SENTINEL</title></head>
            <main><p class="show">The visible data confirms a sustained improvement across independent observation periods.</p>
            <div class="show"><p>BODY_SENTINEL</p></div></main>
        """)
        self.assertIn("visible data", result.visible_text)
        self.assertNotIn("SENTINEL", result.title + result.visible_text)

    def test_visibility_and_quality_thresholds_are_strict(self):
        for count, expected in [(239, True), (240, False)]:
            result = self.extract(f'<main id="root"></main><article><p>{"a" * count}</p></article>'
                                  '<script type="module" src="/app.js"></script>')
            self.assertEqual("js_shell" in result.quality_flags, expected)
        for total, linked, expected in [(399, 240, True), (400, 241, False), (395, 237, False), (395, 238, True)]:
            result = self.extract(f'<main><p><a href="/item">{"a" * linked}</a>{"b" * (total-linked)}</p></main>')
            self.assertEqual("low_snr" in result.quality_flags, expected)

    def test_dynamic_selects_verified_text_and_only_dispatches_once(self):
        from contextlib import nullcontext

        from services.web_extractor import ExtractionContext, WebExtractor
        from services.web_fetch_gateway import FetchedDocument, FetchScope

        calls = []

        class Gateway:
            def fetch(self, url, **kwargs):
                return FetchedDocument(url, "text/html", b'<main id="app"></main><script type="module"></script>')

        class Reader:
            name = "reader_http"

            def can_reuse(self, url, *, tenant_id):
                return tenant_id == "team"

            def extract(self, url, *, scope, tenant_id, request_id, on_dispatch, authorize_redirect):
                on_dispatch()
                calls.append(url)
                return FetchedDocument(url, "text/html", b'<main><p>' + b"Research confirms a sustained improvement. " * 5 + b"</p></main>")

        scope = FetchScope(time.monotonic() + 2, attempt=lambda demand: nullcontext())
        context = ExtractionContext("team", "request", 1, scope, needs_coverage=True)
        extractor = WebExtractor(gateway=Gateway(), adapter=Reader())
        self.assertEqual(extractor.extract(self.url, context=context).tier, "L2")
        self.assertEqual(extractor.extract(self.url, context=context).tier, "L2")
        self.assertEqual(calls, [self.url])


class WebFetchGatewayTests(unittest.TestCase):
    def test_canonical_url_preserves_request_target_bytes(self):
        from services.web_fetch_gateway import normalize_url

        self.assertEqual(
            normalize_url("HTTPS://BÜCHER.example.com.:443/a%2fb?q=a+b&b=%2B&q=%2f#omit"),
            "https://xn--bcher-kva.example.com/a%2fb?q=a+b&b=%2B&q=%2f",
        )
        self.assertEqual(normalize_url("http://example.com?"), "http://example.com/?")

    def test_url_policy_denies_ambiguous_and_nonpublic_authorities(self):
        from services.web_fetch_gateway import GatewayError, normalize_url

        for url in (
            "ftp://example.com/file",
            "https://example.com:80/",
            "https://example.com:/",
            "https://example.com:0443/",
            "https://example.com../",
            "https://localhost/",
            "https://demo.local/",
            "https://name@example.com/",
            "https://example.com\\bad",
            "https://example.com/%q0",
            "https://example.com/\r",
            "https://[fe80::1%25local]/",
            "https://[::ffff:192.0.2.4]/",
            "https://10.1/",
        ):
            with self.subTest(url=url), self.assertRaises(GatewayError) as denied:
                normalize_url(url)
            self.assertEqual(denied.exception.reason, "url_denied")

    def test_mixed_dns_prevents_transport_and_accounting(self):
        from services.web_fetch_gateway import GatewayError

        gateway, scope, calls, attempts = self._gateway(
            addresses=("93.184.216.34", "192.0.2.3"),
        )
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("http://example.com", scope=scope)
        self.assertEqual(denied.exception.reason, "dns_denied")
        self.assertEqual((calls, attempts), ([], []))

    def test_redirect_and_connection_fallback_each_receive_a_lease(self):
        gateway, scope, calls, attempts = self._gateway(
            OSError("synthetic connect failure"),
            self._response(status=b"302 Found", headers=b"Location: /new?b=%2F\r\n"),
            self._response(body=b"document"),
        )
        result = gateway.fetch("http://example.com/start", scope=scope)
        self.assertEqual(result.final_url, "http://example.com/new?b=%2F")
        self.assertEqual(result.redirects, ("http://example.com/new?b=%2F",))
        self.assertEqual(result.body, b"document")
        self.assertEqual([address for _, address in calls], ["93.184.216.34", "93.184.216.35", "93.184.216.34"])
        self.assertEqual(attempts, [{"web_fetches": 1}] * 3)

    def test_redirect_rechecks_authorization_and_does_not_connect_when_denied(self):
        from services.web_fetch_gateway import GatewayError

        gateway, scope, calls, attempts = self._gateway(
            self._response(status=b"302 Found", headers=b"Location: https://other.example.com/\r\n"),
        )
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("http://example.com/", scope=scope, authorize=lambda url: url.startswith("http://example.com/"))
        self.assertEqual(denied.exception.reason, "origin_not_allowed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(attempts), 1)

    def test_unverified_peer_cannot_receive_an_http_request(self):
        from services.web_fetch_gateway import GatewayError

        stream = self._Stream(self._response(body=b"document"), peer="93.184.216.99")
        gateway, scope, calls, attempts = self._gateway(stream)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("http://example.com", scope=scope)
        self.assertEqual(denied.exception.reason, "network_error")
        self.assertEqual(stream.writes, [])
        self.assertTrue(stream.closed)
        self.assertEqual((len(calls), len(attempts)), (1, 1))

    def test_proxy_agnostic_request_has_fixed_host_only_and_preserves_query(self):
        stream = self._Stream(self._response(body=b"document"))
        gateway, scope, _, _ = self._gateway(stream)

        gateway.fetch("http://Example.com:80/a%2Fb?z=1&x=%2b", scope=scope)

        request = b"".join(stream.writes).lower()
        self.assertTrue(request.startswith(b"get /a%2fb?z=1&x=%2b http/1.1\r\n"))
        self.assertIn(b"\r\nhost: example.com\r\n", request)
        for header in (b"cookie:", b"referer:", b"authorization:"):
            self.assertNotIn(header, request)

    def test_compressed_body_cannot_exceed_decoded_cap(self):
        import gzip

        from services.web_fetch_gateway import GatewayError

        stream = self._Stream(self._response(headers=b"Content-Encoding: gzip\r\n", body=gzip.compress(b"z" * 1024)))
        gateway, scope, _, _ = self._gateway(stream)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com/", scope=scope, max_body_bytes=32)
        self.assertEqual(denied.exception.reason, "size_limit")
        self.assertTrue(stream.closed)

    def test_request_count_for_resource_retries_and_redirects_is_shared(self):
        from services.web_fetch_gateway import FetchQuota, GatewayError

        gateway, scope, calls, attempts = self._gateway(
            self._response(status=b"302 Found", headers=b"Location: /another\r\n"),
        )
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com", scope=scope, quota=FetchQuota(1000, 1000, 1), count_request=True)
        self.assertEqual(denied.exception.reason, "size_limit")
        self.assertEqual((len(calls), len(attempts)), (1, 1))

    def test_wire_limit_includes_response_headers_and_chunk_framing(self):
        from services.web_fetch_gateway import GatewayError

        raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n\r\n"
        gateway, scope, _, _ = self._gateway(raw)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com/", scope=scope, max_wire_bytes=len(raw) - 1)
        self.assertEqual(denied.exception.reason, "size_limit")

    def test_truncated_response_is_not_returned(self):
        from services.web_fetch_gateway import GatewayError

        raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 50\r\n\r\nshort"
        gateway, scope, _, _ = self._gateway(raw)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com/", scope=scope)
        self.assertEqual(denied.exception.reason, "response_invalid")

    def test_cancelled_during_read_closes_transport(self):
        from dataclasses import replace

        from services.web_fetch_gateway import GatewayError

        stream = self._Stream(self._response(body=b"document"))
        gateway, scope, _, _ = self._gateway(stream)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com/", scope=replace(scope, cancelled=lambda: stream.reads > 0))
        self.assertEqual(denied.exception.reason, "cancelled")
        self.assertTrue(stream.closed)

    def test_deadline_is_shared_across_redirect_resolution(self):
        from dataclasses import replace

        from services.web_fetch_gateway import GatewayError

        stream = self._Stream(self._response(status=b"302 Found", headers=b"Location: /next\r\n"))
        gateway, scope, calls, attempts = self._gateway(stream)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com/", scope=replace(scope, deadline=2, monotonic=lambda: 2 if stream.closed else 1))
        self.assertEqual(denied.exception.reason, "timeout")
        self.assertEqual((len(calls), len(attempts)), (1, 1))

    def test_dns_io_cancellation_stops_before_physical_http_attempt(self):
        from dataclasses import replace

        from services.web_fetch_gateway import GatewayError

        gateway, scope, calls, attempts = self._gateway()
        scope = replace(scope, check_owner=lambda: False)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com/", scope=scope)
        self.assertEqual(denied.exception.reason, "lease_lost")
        self.assertEqual((calls, attempts), ([], []))

    class _Stream:
        def __init__(self, data, *, peer="93.184.216.34"):
            self.data = data
            self.peer = peer
            self.writes = []
            self.reads = 0
            self.closed = False

        def send(self, data):
            self.writes.append(bytes(data))
            return len(data)

        def recv(self, limit):
            self.reads += 1
            prefix, self.data = self.data[:limit], self.data[limit:]
            return prefix

        def getpeername(self):
            return self.peer, self.port

    @staticmethod
    def _response(*, status=b"200 OK", headers=b"", body=b""):
        return (
            b"HTTP/1.1 " + status + b"\r\nContent-Type: text/plain\r\n"
            + b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            + headers + b"\r\n" + body
        )

    @classmethod
    def _gateway(cls, *script, addresses=("93.184.216.34", "93.184.216.35")):
        from contextlib import contextmanager
        from time import monotonic

        from services.web_fetch_gateway import FetchScope, WebFetchGateway

        steps = iter(script)
        calls = []
        attempts = []

        @contextmanager
        def attempt(demand):
            attempts.append(demand)
            yield

        @contextmanager
        def connect(target, address, scope):
            calls.append((target, address))
            step = next(steps)
            if isinstance(step, OSError):
                raise step
            stream = step if isinstance(step, cls._Stream) else cls._Stream(step, peer=address)
            stream.port = target.port
            try:
                yield stream
            finally:
                stream.closed = True

        gateway = WebFetchGateway(resolver=lambda hostname, scope: addresses, connect=connect)
        return gateway, FetchScope(deadline=monotonic() + 2, attempt=attempt), calls, attempts

    def test_cumulative_wire_budget_prevents_reading_past_limit(self):
        from services.web_fetch_gateway import FetchQuota, GatewayError

        raw = self._response(body=b"document")
        stream = self._Stream(raw)
        gateway, scope, _, _ = self._gateway(stream)
        quota = FetchQuota(len(raw) - 2, 1000)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com", scope=scope, quota=quota)
        self.assertEqual(denied.exception.reason, "size_limit")
        self.assertEqual(quota.wire_bytes, len(raw) - 2)
        self.assertEqual(stream.data, b"nt")
        self.assertTrue(stream.closed)

    def test_dns_requires_consistent_alias_chain_across_address_families(self):
        from unittest.mock import patch

        from services.web_fetch_gateway import GatewayError, WebFetchGateway

        _, scope, calls, _ = self._gateway()

        def query(host, rr_type, scope):
            alias = f"cdn{rr_type}.example.com"
            return {host: alias}, {alias: ["93.184.216.34" if rr_type == 1 else "2606:4700::1"]}

        with patch("services.web_fetch_gateway._dns_query", side_effect=query):
            with self.assertRaises(GatewayError) as denied:
                WebFetchGateway().fetch("http://example.com", scope=scope)
        self.assertEqual(denied.exception.reason, "dns_denied")
        self.assertEqual(calls, [])

    def test_dns_filters_nonpublic_cname_and_mixed_address_answers(self):
        from struct import pack

        from services.web_fetch_gateway import GatewayError, _dns_answer

        name = b"\x07example\x03com\0"
        qname_pointer = b"\xc0\x0c"
        query_id = 1729

        def packet(records):
            return pack("!6H", query_id, 0x8180, 1, len(records), 0, 0) + name + pack("!2H", 1, 1) + b"".join(records)

        def record(rr_type, data):
            return qname_pointer + pack("!HHIH", rr_type, 1, 60, len(data)) + data

        public_a = record(1, bytes([93, 184, 216, 34]))
        private_a = record(1, bytes([192, 0, 2, 7]))
        reserved_alias = record(5, b"\x04demo\x05local\0")
        self.assertEqual(_dns_answer(packet([public_a]), "example.com", 1, query_id), ({}, {"example.com": ["93.184.216.34"]}))
        for records in ([public_a, private_a], [reserved_alias]):
            with self.subTest(records=len(records)), self.assertRaises(GatewayError):
                _dns_answer(packet(records), "example.com", 1, query_id)

    def test_tls_uses_pinned_numeric_peer_and_original_server_name(self):
        import ssl
        from unittest.mock import MagicMock, patch

        from services.web_fetch_gateway import GatewayError, WebFetchGateway

        _, scope, _, attempts = self._gateway()
        raw = MagicMock()
        raw.__enter__.return_value = raw
        raw.connect_ex.return_value = 0
        raw.getpeername.return_value = ("93.184.216.34", 443)
        tls = MagicMock()
        tls.__enter__.return_value = tls
        tls.do_handshake.side_effect = ssl.SSLCertVerificationError("synthetic certificate mismatch")
        context = MagicMock()
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.wrap_socket.return_value = tls

        with (
            patch("services.web_fetch_gateway.socket.socket", return_value=raw),
            patch("services.web_fetch_gateway._tls_context", return_value=context),
            self.assertRaises(GatewayError) as denied,
        ):
            WebFetchGateway(resolver=lambda host, scope: ("93.184.216.34", "93.184.216.35")).fetch(
                "https://example.com/", scope=scope,
            )

        self.assertEqual(denied.exception.reason, "network_error")
        raw.connect_ex.assert_called_once_with(("93.184.216.34", 443))
        context.wrap_socket.assert_called_once_with(
            raw, server_hostname="example.com", do_handshake_on_connect=False, suppress_ragged_eofs=False,
        )
        tls.send.assert_not_called()
        tls.__exit__.assert_called_once()
        self.assertEqual(attempts, [{"web_fetches": 1}])

    def test_response_rejects_conflicting_transfer_framing(self):
        from services.web_fetch_gateway import GatewayError

        raw = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 1\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n\r\n"
        )
        gateway, scope, _, _ = self._gateway(raw)
        with self.assertRaises(GatewayError) as denied:
            gateway.fetch("https://example.com/", scope=scope)
        self.assertEqual(denied.exception.reason, "response_invalid")

    def test_tcp_reset_during_tls_does_not_trigger_ip_fallback(self):
        import errno
        import ssl
        from unittest.mock import MagicMock, patch

        from services.web_fetch_gateway import GatewayError, WebFetchGateway

        _, scope, _, attempts = self._gateway()
        raw = MagicMock()
        raw.__enter__.return_value = raw
        raw.connect_ex.return_value = 0
        raw.getpeername.return_value = ("93.184.216.34", 443)
        tls = MagicMock()
        tls.__enter__.return_value = tls
        tls.do_handshake.side_effect = ConnectionResetError(errno.ECONNRESET, "synthetic TLS interruption")
        context = MagicMock(check_hostname=True, verify_mode=ssl.CERT_REQUIRED)
        context.wrap_socket.return_value = tls

        with (
            patch("services.web_fetch_gateway.socket.socket", return_value=raw),
            patch("services.web_fetch_gateway._tls_context", return_value=context),
            self.assertRaises(GatewayError) as denied,
        ):
            WebFetchGateway(resolver=lambda host, scope: ("93.184.216.34", "93.184.216.35")).fetch(
                "https://example.com/", scope=scope,
            )
        self.assertEqual(denied.exception.reason, "network_error")
        self.assertEqual(attempts, [{"web_fetches": 1}])
        tls.__exit__.assert_called_once()

    def test_expired_owner_while_waiting_on_nonblocking_dns_io_closes_socket(self):
        import socket
        from contextlib import contextmanager
        from dataclasses import replace
        from unittest.mock import patch

        from services.web_fetch_gateway import GatewayError, WebFetchGateway

        _, scope, _, attempts = self._gateway()
        cancellation = False
        first, dns_socket = socket.socketpair()
        with first, dns_socket:
            dns_socket.setblocking(False)

            @contextmanager
            def udp_socket(*args):
                with dns_socket:
                    yield dns_socket

            def cancelled_after_empty_poll(*args, **kwargs):
                nonlocal cancellation
                cancellation = True
                return [], [], []

            with (
                patch("services.web_fetch_gateway._dns_server", return_value="127.0.0.53"),
                patch("services.web_fetch_gateway._socket_address", return_value=(socket.AF_INET, ("127.0.0.53", 53))),
                patch("services.web_fetch_gateway.socket.socket", side_effect=udp_socket),
                patch("services.web_fetch_gateway._connect_socket"),
                patch("services.web_fetch_gateway.select.select", side_effect=cancelled_after_empty_poll),
                self.assertRaises(GatewayError) as denied,
            ):
                WebFetchGateway().fetch("https://example.com/", scope=replace(scope, check_owner=lambda: not cancellation))
            self.assertEqual(dns_socket.fileno(), -1)
        self.assertEqual(denied.exception.reason, "lease_lost")
        self.assertEqual(attempts, [])


class DynamicAdapterTests(unittest.TestCase):
    def setUp(self):
        from services.web_dynamic_adapter import DynamicTargetPolicy

        self.policy = DynamicTargetPolicy.from_config({
            'tenant-a': [{
                'origin': 'https://example.org',
                'path_prefix': '/study',
                'resource_origins': ['https://assets.example.org'],
            }],
        })
        self.url = 'https://example.org/study/article'

    def test_origin_policy_uses_segment_boundaries_and_resource_scope(self):
        from services.web_fetch_gateway import GatewayError

        rule = self.policy.admit(self.url, 'tenant-a')
        self.assertTrue(rule.allows('https://assets.example.org/file.js', resource=True))
        self.assertFalse(rule.allows('https://assets.example.org/file.js', resource=False))
        for url, tenant in (
            ('https://example.org/studying', 'tenant-a'),
            ('https://example.org/study%2Farticle', 'tenant-a'),
            ('https://example.org/study?access_token=fixture', 'tenant-a'),
            ('https://example.org/study/../article', 'tenant-a'),
            (self.url, 'tenant-b'),
        ):
            with self.subTest(url=url, tenant=tenant), self.assertRaises(GatewayError):
                self.policy.admit(url, tenant)

    def _run_reader(self, *, changes=None, capabilities=True):
        import json
        import time
        from contextlib import contextmanager

        from services.web_dynamic_adapter import (
            DynamicCapabilities,
            ReaderEndpoint,
            ReaderHttpAdapter,
            ReaderResponse,
        )
        from services.web_fetch_gateway import FetchedDocument, FetchScope

        events = []
        target = self.url

        class Gateway:
            def fetch(self, url, *, scope, **kwargs):
                scope.check()
                with scope.attempt({'web_fetches': 1}):
                    return FetchedDocument(url, 'text/html', b'<main>example</main>')

        class Deployment:
            def capabilities(self):
                return DynamicCapabilities(1, True, True, True, True) if capabilities else None

            @contextmanager
            def bind(self, egress):
                events.append('bound')
                try:
                    yield egress
                finally:
                    events.append('closed')

            def post(self, endpoint, *, body, binding, scope, max_response_bytes):
                self_endpoint = endpoint.url
                if self_endpoint != 'https://reader.internal/render':
                    raise AssertionError(self_endpoint)
                scope.check()
                self_url = json.loads(body)['url']
                if self_url != target:
                    raise AssertionError(self_url)
                binding.fetch(self_url, kind='document', method='GET')
                response = {
                    'version': 1,
                    'final_url': self_url,
                    'redirects': [],
                    'resources': [],
                    'content_type': 'text/html',
                    'visible_dom': {'tag': 'main', 'visible': True, 'children': [
                        {'tag': 'p', 'visible': True, 'children': ['Reading results & data']},
                        {'tag': 'section', 'visible': False, 'children': ['hidden contents']},
                        {'tag': 'table', 'visible': True, 'children': [
                            {'tag': 'tr', 'visible': True, 'children': [
                                {'tag': 'th', 'visible': True, 'children': ['Value']},
                            ]},
                        ]},
                    ]},
                }
                response.update(changes or {})
                return ReaderResponse(200, 'application/json', json.dumps(response).encode())

        @contextmanager
        def attempt(demand):
            events.append(demand)
            yield

        reader = ReaderHttpAdapter(
            Gateway(), self.policy, ReaderEndpoint('https://reader.internal/render', 'fixture'),
            deployment=Deployment(),
        )
        scope = FetchScope(deadline=time.monotonic() + 2, attempt=attempt)
        return reader, scope, events

    def test_reader_binds_one_attempt_and_checks_visible_structure(self):
        reader, scope, events = self._run_reader()
        document = reader.extract(self.url, scope=scope, tenant_id='tenant-a', request_id='request-a')
        self.assertEqual(document.final_url, self.url)
        self.assertIn(b'Reading results &amp; data', document.body)
        self.assertIn(b'<th>Value</th>', document.body)
        self.assertNotIn(b'hidden contents', document.body)
        self.assertEqual(events, [
            'bound', {'web_fetches': 1, 'dynamic_fetches': 1}, {'web_fetches': 1}, 'closed',
        ])

    def test_reader_rejects_unattested_final_url_and_text_only_output(self):
        from services.web_fetch_gateway import GatewayError

        for changes, reason in (
            ({'final_url': 'https://example.org/study/other'}, 'response_invalid'),
            ({'visible_dom': 'text-only rendering'}, 'visibility_uncertain'),
            ({'resources': ['https://assets.example.org/script.js']}, 'response_invalid'),
        ):
            with self.subTest(changes=changes):
                reader, scope, events = self._run_reader(changes=changes)
                with self.assertRaises(GatewayError) as caught:
                    reader.extract(self.url, scope=scope, tenant_id='tenant-a', request_id='request-a')
                self.assertEqual(caught.exception.reason, reason)
                self.assertEqual(events[-1], 'closed')

    def test_reader_requires_attested_deployment_before_binding_or_dispatch(self):
        from services.web_fetch_gateway import GatewayError

        reader, scope, events = self._run_reader(capabilities=False)
        with self.assertRaises(GatewayError) as caught:
            reader.extract(self.url, scope=scope, tenant_id='tenant-a', request_id='request-a')
        self.assertEqual(caught.exception.reason, 'adapter_unavailable')
        self.assertEqual(events, [])

    def test_playwright_closes_isolation_and_charges_one_worker(self):
        import time
        from contextlib import contextmanager

        from services.web_dynamic_adapter import DynamicCapabilities, PlaywrightAdapter
        from services.web_fetch_gateway import FetchedDocument, FetchScope, GatewayError

        events = []

        class Gateway:
            def fetch(self, url, *, scope, **kwargs):
                with scope.attempt({'web_fetches': 1}):
                    return FetchedDocument(url, 'text/html', b'<main>fixture</main>')

        class Deployment:
            def capabilities(self):
                return DynamicCapabilities(1, True, True, True, True)

            @contextmanager
            def open_worker(self, egress, *, scope):
                events.append('opened')
                try:
                    yield self
                finally:
                    events.append('closed')
                    with self_case.assertRaises(GatewayError):
                        egress.fetch(egress.url, kind='document', method='GET')

            def render(self, url, *, egress, scope):
                egress.fetch(url, kind='document', method='GET')
                with self_case.assertRaises(GatewayError):
                    egress.fetch('https://other.example.org/private', kind='subresource', method='GET')
                return {
                    'version': 1, 'final_url': url, 'redirects': [], 'resources': [], 'content_type': 'text/html',
                    'visible_dom': {'tag': 'main', 'visible': True, 'children': ['Rendered publication']},
                }

        @contextmanager
        def attempt(demand):
            events.append(demand)
            yield

        self_case = self
        adapter = PlaywrightAdapter(Gateway(), self.policy, deployment=Deployment(), enabled=True)
        scope = FetchScope(deadline=time.monotonic() + 2, attempt=attempt)
        document = adapter.extract(
            self.url, scope=scope, tenant_id='tenant-a', request_id='request-a', can_dispatch=lambda demand: True,
        )
        self.assertIn(b'Rendered publication', document.body)
        self.assertEqual(events, [{'dynamic_fetches': 1}, 'opened', {'web_fetches': 1}, 'closed'])

    def test_playwright_web_budget_preflight_keeps_rendering_closed(self):
        import time

        from services.web_dynamic_adapter import DynamicCapabilities, PlaywrightAdapter
        from services.web_fetch_gateway import FetchScope, GatewayError

        class Deployment:
            def capabilities(self):
                return DynamicCapabilities(1, True, True, True, True)

            def open_worker(self, egress, *, scope):
                raise AssertionError('worker opened after denied budget')

        adapter = PlaywrightAdapter(None, self.policy, deployment=Deployment(), enabled=True)
        scope = FetchScope(deadline=time.monotonic() + 2)
        with self.assertRaises(GatewayError) as caught:
            adapter.extract(
                self.url, scope=scope, tenant_id='tenant-a', request_id='request-a', can_dispatch=lambda demand: False,
            )
        self.assertEqual(caught.exception.reason, 'budget_denied')

    def test_reader_service_transport_uses_one_fixed_post_and_refuses_redirect(self):
        import ssl
        import time
        from contextlib import contextmanager
        from unittest.mock import patch

        from services.web_dynamic_adapter import ReaderEndpoint, ReaderHttpsTransport
        from services.web_fetch_gateway import FetchScope, GatewayError

        class Connection:
            def __init__(self):
                self.outbound = b''
                self.inbound = b'HTTP/1.1 302 Found\r\nLocation: https://external.example.org/next\r\nContent-Length: 0\r\n\r\n'

            def send(self, value):
                self.outbound += value
                return len(value)

            def recv(self, length):
                result, self.inbound = self.inbound[:length], self.inbound[length:]
                return result

        connection = Connection()
        destinations = []

        @contextmanager
        def connect(target, address, scope, *, tls_context):
            destinations.append((target.host, address, target.request_target))
            self.assertTrue(tls_context.check_hostname)
            self.assertEqual(tls_context.verify_mode, ssl.CERT_REQUIRED)
            yield connection

        transport = ReaderHttpsTransport('192.0.2.10', ssl.create_default_context())
        scope = FetchScope(deadline=time.monotonic() + 2)
        with patch('services.web_dynamic_adapter._open_pinned', connect):
            with self.assertRaises(GatewayError) as caught:
                transport.post(
                    ReaderEndpoint('https://reader.internal/render', 'service-secret'), body=b'{}',
                    binding='ephemeral-binding', scope=scope, max_response_bytes=1024,
                )
        self.assertEqual(caught.exception.reason, 'response_invalid')
        self.assertEqual(destinations, [('reader.internal', '192.0.2.10', '/render')])
        self.assertIn(b'POST /render HTTP/1.1\r\n', connection.outbound)
        self.assertIn(b'Authorization: Bearer service-secret\r\n', connection.outbound)
        self.assertIn(b'X-Research-Egress: ephemeral-binding\r\n', connection.outbound)
        self.assertTrue(connection.outbound.endswith(b'\r\n\r\n{}'))

    def test_isolated_chromium_requires_capability_before_loading_dependency(self):
        import asyncio
        import time
        from unittest.mock import patch

        from services.web_dynamic_adapter import (
            DynamicCapabilities,
            sample_isolated_chromium,
        )
        from services.web_fetch_gateway import FetchScope, GatewayError

        scope = FetchScope(deadline=time.monotonic() + 2)
        with patch.dict('sys.modules', {'playwright.async_api': None}):
            for capabilities in (
                DynamicCapabilities(1, False, True, True, True),
                DynamicCapabilities(1, True, True, True, True),
            ):
                with self.subTest(capabilities=capabilities), self.assertRaises(GatewayError) as caught:
                    asyncio.run(sample_isolated_chromium(
                        self.url, scope=scope, capabilities=capabilities, proxy_server='http://127.0.0.1:9123',
                    ))
                self.assertEqual(caught.exception.reason, 'adapter_unavailable')

    def test_egress_binding_rejects_cross_tenant_and_expired_capabilities(self):
        from services.web_dynamic_adapter import DynamicEgressBroker
        from services.web_fetch_gateway import GatewayError

        class Egress:
            tenant_id = 'tenant-a'
            request_id = 'request-a'

            def fetch(self, url, *, kind, method):
                return (url, kind, method)

        broker = DynamicEgressBroker()
        with broker.bind(Egress()) as token:
            kwargs = {'request_id': 'request-a', 'url': self.url, 'kind': 'document', 'method': 'GET'}
            with self.assertRaises(GatewayError):
                broker.fetch(token, tenant_id='tenant-b', **kwargs)
            self.assertEqual(broker.fetch(token, tenant_id='tenant-a', **kwargs), (self.url, 'document', 'GET'))
        with self.assertRaises(GatewayError):
            broker.fetch(token, tenant_id='tenant-a', **kwargs)

    def test_dynamic_fifo_removes_cancelled_head_before_admitting_next_tenant(self):
        import time
        from threading import Event, Thread, get_ident

        from services.web_dynamic_adapter import DynamicConcurrency
        from services.web_fetch_gateway import FetchScope, GatewayError

        concurrency = DynamicConcurrency()
        first_waiting, second_waiting, cancelled, second_entered = (Event() for _ in range(4))
        results = []
        deadline = time.monotonic() + 3

        def contender(tenant, waiting, cancelled_check):
            checks = 0

            def owner():
                nonlocal checks
                checks += 1
                if checks == 2:
                    waiting.set()
                return True

            scope = FetchScope(deadline=deadline, check_owner=owner, cancelled=cancelled_check)
            try:
                with concurrency.acquire(tenant, scope, adapter='reader_http'):
                    results.append((tenant, get_ident()))
                    second_entered.set()
            except GatewayError as error:
                results.append((tenant, error.reason))

        first = Thread(target=contender, args=('tenant-a', first_waiting, cancelled.is_set))
        second = Thread(target=contender, args=('tenant-b', second_waiting, lambda: False))
        with concurrency.acquire('tenant-a', FetchScope(deadline), adapter='reader_http'):
            first.start()
            self.assertTrue(first_waiting.wait(1))
            second.start()
            self.assertTrue(second_waiting.wait(1))
            self.assertFalse(second_entered.is_set())
            cancelled.set()
            self.assertTrue(second_entered.wait(1))
        first.join(1)
        second.join(1)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertIn(('tenant-a', 'cancelled'), results)
        self.assertTrue(any(tenant == 'tenant-b' and type(outcome) is int for tenant, outcome in results))


class WebExtractionBudgetIntegrationTests(unittest.TestCase):
    def test_reader_preserves_committed_dispatch_and_separate_upstream_charge(self):
        from dataclasses import replace
        from tempfile import TemporaryDirectory

        from config import Configuration
        from models import SummaryState
        from services.budget import BudgetGovernor, build_request_budget
        from services.request_state import RequestStateStore
        from services.web_extractor import ExtractionContext, WebExtractor
        from services.web_fetch_gateway import FetchedDocument
        from state_manager import (
            RequestBudgetController,
            StateManagerMixin,
            freeze_web_extraction,
        )

        fixture = DynamicAdapterTests()
        fixture.setUp()
        reader, scope, events = fixture._run_reader()
        config = Configuration(
            v4_web_extractor_enabled=True, v4_web_extractor_tenants=["tenant-a"],
            v4_web_l2_adapter="reader_http",
            v4_web_l2_origin_rules={"tenant-a": [{"origin": "https://example.org", "path_prefix": "/study"}]},
            v4_web_reader_url="https://reader.internal/render", v4_web_reader_token="fixture",
        )

        class Gateway:
            def fetch(self, url, *, scope, **kwargs):
                with scope.attempt({"web_fetches": 1}):
                    return FetchedDocument(url, "text/html", b'<main id="app"></main><script type="module"></script>')

        with TemporaryDirectory() as directory:
            manager = StateManagerMixin()
            manager._request_state_store = RequestStateStore(directory)
            manager.tenant_id, manager.request_id = "tenant-a", "web-budget"
            manager._is_cancelled = lambda: False
            v4 = {"schema_version": 1, "web_extraction": freeze_web_extraction(config, manager.tenant_id)}
            snapshot = manager._request_state_store.save(manager.request_id, {"v4": v4}, tenant_id=manager.tenant_id)
            budget = build_request_budget("quick")

            def run(controller):
                context = ExtractionContext(
                    manager.tenant_id, manager.request_id, 1,
                    replace(scope, attempt=lambda demand: controller.attempt(demand, required=False)),
                    needs_coverage=True,
                    dynamic_attempt=lambda demand, urls: controller.attempt(demand, required=False, dynamic_urls=urls),
                )
                return WebExtractor(gateway=Gateway(), adapter=reader).extract(fixture.url, context=context)

            governor = BudgetGovernor(budget, tenant_id=manager.tenant_id, request_id=manager.request_id)
            first = RequestBudgetController(manager, SummaryState(
                v4=snapshot["v4"], state_version=snapshot["state_version"],
            ), governor)
            run(first)
            self.assertEqual(first.snapshot().external_attempts, 4)
            self.assertEqual(first.snapshot().spent["web_fetches"], 3)
            self.assertEqual(first.snapshot().spent["dynamic_fetches"], 1)
            self.assertEqual(events, ["bound", "closed"])
            snapshot = manager._request_state_store.load(manager.request_id, tenant_id=manager.tenant_id)
            restored = BudgetGovernor(budget, tenant_id=manager.tenant_id, request_id=manager.request_id)
            restored.restore_ledger(snapshot["v4"]["budget"]["ledger"])
            second = RequestBudgetController(manager, SummaryState(
                v4=snapshot["v4"], state_version=snapshot["state_version"],
            ), restored)
            run(second)
            self.assertEqual(second.snapshot().external_attempts, 5)
            self.assertEqual(second.snapshot().spent["dynamic_fetches"], 1)
