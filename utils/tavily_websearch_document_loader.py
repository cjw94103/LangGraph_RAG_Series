import re, html
from typing import Dict, List
import trafilatura
from readability import Document
from bs4 import BeautifulSoup
from langchain.schema import Document

def clean_html_to_text(html_str: str) -> str:
    
    # 1) trafilatura 시도
    txt = trafilatura.extract(html_str, include_comments=False, include_tables=False)
    if txt and txt.strip():
        return txt.strip()
    # 2) readability 백업
    try:
        doc = Document(html_str)
        soup = BeautifulSoup(doc.summary(html_partial=True), "lxml")
        for tag in soup(["script","style","noscript","iframe","img","svg","picture","source","video","audio"]):
            tag.decompose()
        txt = soup.get_text(" ", strip=True)
        if txt:
            return txt
    except Exception:
        pass
    # 3) 마지막 백업: 단순 HTML 제거
    soup = BeautifulSoup(html_str, "lxml")
    for tag in soup(["script","style","noscript","iframe","img","svg","picture","source","video","audio"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")

def sanitize(text: str, max_chars: int = 5000) -> str:
    
    text = html.unescape(text)
    text = URL_RE.sub("", text)            # URL 제거
    text = WS_RE.sub(" ", text).strip()    # 공백 정규화
    return text[:max_chars]                # 길이 제한

def postprocess_tavily(results: List[Dict]) -> List[Dict]:
    
    cleaned = []
    for r in results:
        if r.get("raw_content") is not None:
            content = r.get("raw_content")
        else:
            content = r.get("content")
        # 혹시 HTML 섞여오면 정제
        content = clean_html_to_text(content)
        content = sanitize(content)
        # 짧거나 광고성은 걸러내기
        if len(content) < 200:       # 임계값은 상황에 맞게
            continue
        # cleaned.append({
        #     "title": r.get("title",""),
        #     "url": r.get("url",""),
        #     "content": content
        # })
        cleaned.append(Document(page_content=content, metadata={"page" : "", "title" : r.get("title",""), "source" : r.get("url", "")}))
    # 중복 제거(유사 URL 기준)
    seen = set()
    deduped = []
    for r in cleaned:
        key = r.metadata["source"].split("?")[0]
        if key in seen: 
            continue
        seen.add(key)
        deduped.append(r)
    return deduped