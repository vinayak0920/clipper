"""Country / region filters.

Neither Twitch nor Kick tells you where a streamer lives. What both APIs do expose
is the stream language, and Twitch also exposes free-form tags. So a region is
defined as: a set of stream languages + optional tag/title keywords that pull in
English-language streamers from that country (e.g. Indian streamers on English).
"""
from __future__ import annotations

REGIONS: dict[str, dict] = {
    "india": {
        "label": "India",
        "languages": ["hi", "ta", "te", "bn", "mr", "kn", "ml", "pa", "gu"],
        "keywords": ["india", "indian", "hindi", "desi", "hinglish", "tamil", "telugu", "bgmi", "bharat"],
        "keyword_languages": ["en"],
    },
    "pakistan": {
        "label": "Pakistan",
        "languages": ["ur"],
        "keywords": ["pakistan", "pakistani", "urdu"],
        "keyword_languages": ["en"],
    },
    "usa_uk": {"label": "USA / UK / English", "languages": ["en"], "keywords": [], "keyword_languages": []},
    "brazil_portugal": {"label": "Brazil / Portugal", "languages": ["pt"], "keywords": [], "keyword_languages": []},
    "spain_latam": {"label": "Spain / Latin America", "languages": ["es"], "keywords": [], "keyword_languages": []},
    "france": {"label": "France", "languages": ["fr"], "keywords": [], "keyword_languages": []},
    "germany": {"label": "Germany / Austria", "languages": ["de"], "keywords": [], "keyword_languages": []},
    "italy": {"label": "Italy", "languages": ["it"], "keywords": [], "keyword_languages": []},
    "turkey": {"label": "Turkey", "languages": ["tr"], "keywords": [], "keyword_languages": []},
    "russia": {"label": "Russia / CIS", "languages": ["ru"], "keywords": [], "keyword_languages": []},
    "poland": {"label": "Poland", "languages": ["pl"], "keywords": [], "keyword_languages": []},
    "arabic": {"label": "Middle East / Arabic", "languages": ["ar"], "keywords": [], "keyword_languages": []},
    "japan": {"label": "Japan", "languages": ["ja"], "keywords": [], "keyword_languages": []},
    "korea": {"label": "South Korea", "languages": ["ko"], "keywords": [], "keyword_languages": []},
    "indonesia": {"label": "Indonesia", "languages": ["id"], "keywords": [], "keyword_languages": []},
    "philippines": {
        "label": "Philippines",
        "languages": ["tl"],
        "keywords": ["philippines", "pinoy", "filipino", "tagalog"],
        "keyword_languages": ["en"],
    },
    "vietnam_thai": {"label": "Vietnam / Thailand", "languages": ["vi", "th"], "keywords": [], "keyword_languages": []},
}


def matches_keywords(region: dict, title: str, tags: list[str] | None) -> bool:
    kws = region.get("keywords") or []
    if not kws:
        return False
    hay = (title or "").lower() + " " + " ".join(t.lower() for t in (tags or []))
    return any(k in hay for k in kws)
