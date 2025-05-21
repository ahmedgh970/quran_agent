import logging
from typing import List, Dict, Any, Iterator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from langchain.schema import Document

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=0.5,
                          status_forcelist=[429, 500, 502, 503, 504])
    )
)

class QuranClient:
    def __init__(self, base_url: str, token: str, client_id: str):
        self.base_url     = base_url.rstrip("/")
        self.headers      = {
            "Authorization": f"Bearer {token}",
            "x-auth-token":  token,
            "x-client-id":   client_id,
        }

    def get_chapters(self) -> List[Dict[str, Any]]:
        resp = _session.get(
            f"{self.base_url}/chapters",
            headers=self.headers,
            params={"per_page": 200},
        )
        resp.raise_for_status()
        return resp.json()["chapters"]

    def get_verses(self, chapter_id: int) -> List[Dict[str, Any]]:
        resp = _session.get(
            f"{self.base_url}/verses/by_chapter/{chapter_id}",
            headers=self.headers,
            params={"language": "en", "words": False, "per_page": 300},
        )
        resp.raise_for_status()
        return resp.json()["verses"]

    def iter_documents(self) -> Iterator[Document]:
        chapters = self.get_chapters()
        for chap in tqdm(chapters, desc="Chapters"):
            try:
                verses = self.get_verses(chap["id"])
            except Exception as e:
                logger.error(f"Chapter {chap['id']} failed: {e}")
                continue

            for v in verses:
                translations = [
                    w["translation"]["text"]
                    for w in v["words"]
                    if w["char_type_name"] == "word" and w["translation"]["text"]
                ]
                english = " ".join(translations)
                meta = {
                    "surah_id":         str(chap["id"]),
                    "surah_name":       chap["name_simple"],
                    "pages":            str(chap["pages"]),
                    "revelation_place": chap["revelation_place"],
                    "revelation_order": str(chap["revelation_order"]),
                    "ayah":             str(v["verse_number"]),
                    "verse_key":        str(v["verse_key"]),
                    "juz_number":       str(v["juz_number"]),
                    "page_number":      str(v["page_number"]),
                }
                yield Document(page_content=english, metadata=meta)
