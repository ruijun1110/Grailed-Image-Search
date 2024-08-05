import json
from datetime import datetime, timezone
from typing import Dict, List, Set
from csv_converter import ScrapingUtils


class JSONProcessor:
    @staticmethod
    def process_data(raw_data: List[Dict]) -> List[Dict]:
        processed_items = []
        for result in raw_data:
            hits = result["results"][0]["hits"]
            for hit in hits:
                designers = [designer["name"] for designer in hit.get("designers", [])]
                processed_item = {
                    "id": hit.get("id"),
                    "title": ScrapingUtils.clean_title(hit.get("title"), designers),
                    # "description": hit.get("description"),
                    "category": {
                        "main": ScrapingUtils.clean_category(hit.get("category")),
                        "sub": ScrapingUtils.get_sub_category(hit.get("category_path")),
                    },
                    "designers": designers,
                    "price": {
                        "current": hit.get("price"),
                        # "original": (
                        #     hit["price_drops"][0]
                        #     if hit.get("price_drops")
                        #     else hit.get("price")
                        # ),
                        # "drops": hit.get("price_drops", []),
                        "updated_at": hit.get("price_updated_at"),
                    },
                    # "condition": hit.get("condition"),
                    "color": (
                        hit.get("traits", [])[0]["value"]
                        if hit.get("traits")
                        else "unspecified"
                    ),
                    "size": hit.get("size"),
                    "image": {
                        "url": hit.get("cover_photo", {}).get("url"),
                        "width": hit.get("cover_photo", {}).get("width"),
                        "height": hit.get("cover_photo", {}).get("height"),
                    },
                    # "location": hit.get("location"),
                    # "seller": {
                    #     "id": hit.get("user", {}).get("id"),
                    #     "username": hit.get("user", {}).get("username"),
                    #     "rating": hit.get("user", {})
                    #     .get("seller_score", {})
                    #     .get("rating_average"),
                    #     "avatar_url": hit.get("user", {}).get("avatar_url"),
                    # },
                    # "traits": hit.get("traits", []),
                    "created_at": hit.get("created_at"),
                    # "tags": hit.get("hashtags", []),
                    "department": hit.get("department"),
                    # "make_offer": hit.get("makeoffer"),
                    # "badges": hit.get("badges", []),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
                processed_items.append(processed_item)
        return processed_items

    def remove_duplicates(self, items: List[Dict], seen_ids: Set[str]) -> List[Dict]:
        unique_items = []
        for item in items:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                unique_items.append(item)
        return unique_items
