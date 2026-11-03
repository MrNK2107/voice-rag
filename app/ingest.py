"""Turn uploaded files and URLs into plain-text documents.

Supported: PDF (one document per page, so citations can name the page; needs
the optional ``pypdf`` package), plain text, Markdown, HTML, and http(s) URLs.

URL fetching refuses private, loopback and link-local addresses (a basic SSRF
guard for a server that fetches on a user's behalf), caps the download size
and follows at most 3 redirects, re-checking each hop.
"""
from __future__ import annotations

import io
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, List, Optional
from urllib.parse import urljoin, urlparse

import requests

MAX_URL_BYTES = 5 * 1024 * 1024
TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".rst", ".csv"}
HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | HTML_EXTENSIONS | {".pdf"}


class IngestError(ValueError):
    pass


@dataclass
class ExtractedDoc:
    title: str
    text: str
    source: str
    page: Optional[int] = None


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form"}
    _BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "section", "article", "td"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """(title, text) of an HTML page, scripts/styles/navigation removed."""
    p = _HTMLText()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    paras = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # Lines without sentence punctuation (menus, buttons) get a full stop so
    # the sentence splitter doesn't glue them onto the next paragraph.
    paras = [ln if re.search(r"[.!?。！？।؟:;]$", ln) else ln + "." for ln in paras if len(ln) > 2]
    return p.title.strip(), " ".join(paras)


_MD_RULES = [
    (re.compile(r"```.*?```", re.S), " "),          # code fences
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),     # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # links -> text
    (re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$", re.M), r"\1."),  # headings
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),       # bullets
    (re.compile(r"[*_`]{1,3}"), ""),               # emphasis / inline code
    (re.compile(r"^\s*>\s?", re.M), ""),           # quotes
]


def markdown_to_text(md: str) -> str:
    for rx, rep in _MD_RULES:
        md = rx.sub(rep, md)
    return md


def _decode(data: bytes) -> str:
    # UTF-16 only with a BOM, or when NUL bytes show it (BOM-less ASCII-range
    # UTF-16): almost any even-length byte string "decodes" as UTF-16, which
    # turned Windows-1252 French / Spanish files into CJK mojibake.
    candidates = ["utf-8-sig"]
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.insert(0, "utf-16")
    elif data and data.count(b"\x00") * 4 > len(data):
        candidates.append("utf-16-le" if data[1:2] == b"\x00" else "utf-16-be")
    candidates.append("cp1252")
    for enc in candidates:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _ext(filename: str) -> str:
    m = re.search(r"(\.[A-Za-z0-9]+)$", filename or "")
    return m.group(1).lower() if m else ""


def extract_file(filename: str, data: bytes) -> List[ExtractedDoc]:
    """Documents in an uploaded file. Raises IngestError for unsupported or empty input."""
    ext = _ext(filename)
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename or "document").replace("_", " ").strip() or "document"
    if ext == ".pdf":
        return _extract_pdf(filename, stem, data)
    if ext in HTML_EXTENSIONS:
        title, text = html_to_text(_decode(data))
        docs = [ExtractedDoc(title or stem, text, filename)]
    elif ext in TEXT_EXTENSIONS:
        text = _decode(data)
        if ext in {".md", ".markdown"}:
            text = markdown_to_text(text)
        docs = [ExtractedDoc(stem, text, filename)]
    else:
        raise IngestError(f"Unsupported file type '{ext or filename}'. Use PDF, TXT, MD or HTML.")
    docs = [d for d in docs if d.text.strip()]
    if not docs:
        raise IngestError("No text found in the file.")
    return docs


def _extract_pdf(filename: str, stem: str, data: bytes) -> List[ExtractedDoc]:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError as e:
        raise IngestError("PDF support needs the 'pypdf' package (pip install pypdf).") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(ExtractedDoc(f"{stem} (p. {i})", text, filename, page=i))
    except Exception as e:  # malformed PDF
        raise IngestError(f"Could not read the PDF: {e}") from e
    if not docs:
        raise IngestError("The PDF has no extractable text (scanned images need OCR first).")
    return docs


def _check_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestError("Only http(s) URLs can be ingested.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise IngestError(f"Cannot resolve host '{parsed.hostname}'.") from e
    for info in infos:
        if _blocked_ip(info[4][0]):
            raise IngestError("URLs that point to private or local network addresses are not allowed.")


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _blocked_ip(addr: str) -> bool:
    """True unless ``addr`` is a public unicast address. ``is_global`` also
    rejects shared/CGNAT space (100.64.0.0/10, where some clouds serve instance
    metadata) and other special ranges the older flag list missed; IPv6 forms
    that embed an IPv4 address (mapped, 6to4, Teredo, NAT64) are judged by it."""
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return True
    embedded = None
    if ip.version == 6:
        embedded = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is None and ip in _NAT64:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    for a in (ip, embedded):
        if a is None:
            continue
        if (not a.is_global or a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
                or a.is_multicast or a.is_unspecified):
            return True
    return False


def _peer_ip(resp) -> Optional[str]:
    """IP address of the server a streamed requests response is connected to
    (None if it cannot be determined, e.g. for a test double)."""
    raw = getattr(resp, "raw", None)
    for path in (("_connection", "sock"), ("_fp", "fp", "raw", "_sock")):
        obj = raw
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            try:
                return obj.getpeername()[0]
            except (OSError, AttributeError, IndexError, TypeError):
                continue
    return None


def fetch_url(url: str, get: Optional[Callable] = None, check_host: bool = True) -> List[ExtractedDoc]:
    """Fetch and extract a web page or PDF. ``get`` is injectable for tests."""
    get = get or requests.get
    current = url
    for _ in range(4):
        if check_host:
            _check_host(current)
        resp = get(current, timeout=10, stream=True, allow_redirects=False,
                   headers={"User-Agent": "Vouch-ingest/1.0"})
        # _check_host resolved the name once; the connection resolved it again.
        # A rebinding DNS name can answer differently the second time, so check
        # the address actually connected to before reading anything.
        peer = _peer_ip(resp)
        if check_host and peer and _blocked_ip(peer):
            resp.close()
            raise IngestError("URLs that point to private or local network addresses are not allowed.")
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
            current = urljoin(current, resp.headers["location"])
            continue
        break
    else:
        raise IngestError("Too many redirects.")
    if resp.status_code >= 400:
        raise IngestError(f"The URL returned HTTP {resp.status_code}.")
    data = b""
    for chunk in resp.iter_content(65536):
        data += chunk
        if len(data) > MAX_URL_BYTES:
            raise IngestError("The page is larger than 5 MB.")
    ctype = (resp.headers.get("content-type") or "").lower()
    name = urlparse(current).path.rsplit("/", 1)[-1] or urlparse(current).hostname
    if "pdf" in ctype or name.lower().endswith(".pdf"):
        docs = _extract_pdf(current, name, data)
    elif "html" in ctype or not ctype:
        title, text = html_to_text(_decode(data))
        docs = [ExtractedDoc(title or name, text, current)]
    elif ctype.startswith("text/"):
        docs = [ExtractedDoc(name, _decode(data), current)]
    else:
        raise IngestError(f"Unsupported content type '{ctype}'.")
    for d in docs:
        d.source = current
    docs = [d for d in docs if d.text.strip()]
    if not docs:
        raise IngestError("No text found at that URL.")
    return docs
