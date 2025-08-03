import json, re
from collections import Counter

# Đổi path tới file thật của bạn
PATH = r"D:\airdrop\CuocThiAI\HOU\ChuongTrinhHoc\Chuongtrinhhoc.json"

COURSE_NAME_KEYS = {"ten_hp","tenhocphan","tên học phần","ten mon","ten_mon","course_name","hocphan","tên","học phần","mon_hoc","môn học","tenmon"}
CREDIT_KEYS = {"so_tin_chi","tin_chi","sotinchi","credits","tc","số_tín_chỉ"}

def lower_keys(d): return {str(k).lower(): v for k,v in d.items()}

def looks_like_course(o):
    if not isinstance(o, dict): return False
    L = lower_keys(o)
    has_name = any(k in L for k in COURSE_NAME_KEYS)
    has_other = any(k in L for k in CREDIT_KEYS)
    return has_name and has_other

def walk(o, acc):
    if isinstance(o, dict):
        acc["dicts"] += 1
        if looks_like_course(o): acc["course_like"] += 1
        for v in o.values(): walk(v, acc)
    elif isinstance(o, list):
        for v in o: walk(v, acc)

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

acc = Counter()
walk(data, acc)
print("Course-like (đếm lại từ JSON):", acc["course_like"])
