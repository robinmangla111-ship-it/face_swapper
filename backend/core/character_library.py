"""
Character Library Manager
=========================
Manages curated Hollywood, Bollywood, Avengers, and Mythology character portraits
plus user-added custom characters stored persistently in backend/library/.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CharacterLibrary:
    """Manages persistent character presets for one-click face swapping."""

    def __init__(self, library_dir: Optional[Path] = None):
        from config import LIBRARY_DIR, LIBRARY_IMAGES_DIR
        self.library_dir = Path(library_dir) if library_dir else LIBRARY_DIR
        self.images_dir = self.library_dir / "images" if library_dir else LIBRARY_IMAGES_DIR
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.library_dir / "characters.json"

    def _load_manifest(self) -> List[Dict]:
        if not self.manifest_path.exists():
            return []
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Failed to read character library manifest: {e}")
            return []

    def _save_manifest(self, items: List[Dict]):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

    def list_characters(self, category: Optional[str] = None) -> List[Dict]:
        """Return all characters whose image file exists on disk."""
        items = self._load_manifest()
        valid = []
        for item in items:
            img_path = self.images_dir / item.get("filename", "")
            if img_path.exists():
                if category and category.lower() != "all" and item.get("category", "").lower() != category.lower():
                    continue
                valid.append({
                    "id": item["id"],
                    "name": item["name"],
                    "category": item.get("category", "Custom"),
                    "swap_mode": item.get("swap_mode", "auto"),
                    "filename": item["filename"],
                    "image_url": f"/library-images/{item['filename']}",
                    "is_custom": bool(item.get("is_custom", False)),
                })
        return valid

    def get_character(self, char_id: str) -> Optional[Dict]:
        """Find a character by ID and return its metadata + full image_path."""
        for item in self._load_manifest():
            if item.get("id") == char_id:
                img_path = self.images_dir / item.get("filename", "")
                if img_path.exists():
                    return {
                        **item,
                        "image_path": str(img_path),
                        "image_url": f"/library-images/{item['filename']}",
                    }
        return None

    def add_character(self, name: str, category: str, filename: str,
                      swap_mode: str = "auto", is_custom: bool = True,
                      char_id: Optional[str] = None) -> Dict:
        """Add a new character entry to the persistent library."""
        items = self._load_manifest()
        cid = char_id or f"lib_{str(uuid.uuid4())[:8]}"
        entry = {
            "id": cid,
            "name": name.strip(),
            "category": category.strip() or "Custom",
            "swap_mode": swap_mode,
            "filename": filename,
            "is_custom": is_custom,
        }
        # Replace if ID already exists
        items = [it for it in items if it.get("id") != cid]
        items.append(entry)
        self._save_manifest(items)
        return {
            **entry,
            "image_url": f"/library-images/{filename}",
        }

    def delete_character(self, char_id: str) -> bool:
        """Remove a character from the library and delete its image if custom."""
        items = self._load_manifest()
        target = next((it for it in items if it.get("id") == char_id), None)
        if not target:
            return False
        items = [it for it in items if it.get("id") != char_id]
        self._save_manifest(items)
        img_path = self.images_dir / target.get("filename", "")
        if img_path.exists() and target.get("is_custom", True):
            img_path.unlink(missing_ok=True)
        return True
