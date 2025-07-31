import json

INPUT_FILE = "../data/menu_contents.json"
OUTPUT_REFINED_FILE = "../data/menu_contents_refined.json"


def refine_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    refined = []
    for item in data:
        # Bỏ bài viết nếu không có nội dung
        if not item.get("content_cleaned", "").strip():
            continue

        refined.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "category": item.get("category", ""),
            "date": item.get("date", ""),
            "content": item.get("content_cleaned", "")
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu file đã lọc: {output_path} (Tổng {len(refined)} bài)")


if __name__ == "__main__":
    refine_json(INPUT_FILE, OUTPUT_REFINED_FILE)
