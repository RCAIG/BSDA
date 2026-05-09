#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Tuple

import requests

_CACHE: Dict[Tuple[float, float, str], Dict[str, Any]] = {}
_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0


def _cached_key(lat: float, lon: float, language: str) -> Tuple[float, float, str]:
    return (round(float(lat), 6), round(float(lon), 6), language.strip() or "en")


def _nominatim_http_reverse(lat: float, lon: float, language: str = "en") -> Dict[str, Any]:
    global _LAST_CALL_TS

    with _LOCK:
        now = time.time()
        elapsed = now - _LAST_CALL_TS
        # Be conservative with Nominatim public endpoint.
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _LAST_CALL_TS = time.time()

    url = "http<LOCAL_PATH>"
    headers = {"User-Agent": "DDA-PerceptionAgent/1.0 (research-use)"}
    params = {
        "format": "jsonv2",
        "lat": float(lat),
        "lon": float(lon),
        "zoom": 18,
        "addressdetails": 1,
        "accept-language": language or "en",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=12)
    resp.raise_for_status()
    obj = resp.json() if resp.content else {}
    addr = obj.get("address", {}) if isinstance(obj, dict) else {}

    return {
        "success": True,
        "source": "nominatim_http",
        "display_name": obj.get("display_name", "") if isinstance(obj, dict) else "",
        "address": addr,
        "country": addr.get("country", ""),
        "state": addr.get("state", ""),
        "city": addr.get("city", "") or addr.get("town", "") or addr.get("village", ""),
        "county": addr.get("county", ""),
        "suburb": addr.get("suburb", "") or addr.get("neighbourhood", ""),
        "road": addr.get("road", ""),
        "postcode": addr.get("postcode", ""),
        "raw": obj if isinstance(obj, dict) else {},
    }


def reverse_geocode_location(lat: float, lon: float, language: str = "en") -> Dict[str, Any]:
    """
    Reverse geocode via Nominatim with in-process cache.
    Returns a normalized dict and never raises.
    """
    try:
        key = _cached_key(lat, lon, language)
    except Exception:
        return {
            "success": False,
            "source": "none",
            "error": "invalid_lat_lon",
            "display_name": "",
            "address": {},
        }

    if key in _CACHE:
        cached = dict(_CACHE[key])
        cached["cache_hit"] = True
        return cached

    try:
        out = _nominatim_http_reverse(lat=float(lat), lon=float(lon), language=language)
        out["cache_hit"] = False
        _CACHE[key] = dict(out)
        return out
    except Exception as e:
        return {
            "success": False,
            "source": "nominatim_http",
            "error": str(e),
            "display_name": "",
            "address": {},
            "cache_hit": False,
        }

