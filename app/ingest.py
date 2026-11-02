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
    for enc in ("utf-8-sig", "utf-16"):
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
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise IngestError("URLs that point to private or local network addresses are not allowed.")


def fetch_url(url: str, get: Optional[Callable] = None, check_host: bool = True) -> List[ExtractedDoc]:
    """Fetch and extract a web page or PDF. ``get`` is injectable for tests."""
    get = get or requests.get
    current = url
    for _ in range(4):
        if check_host:
            _check_host(current)
        resp = get(current, timeout=10, stream=True, allow_redirects=False,
                   headers={"User-Agent": "Vouch-ingest/1.0"})
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
