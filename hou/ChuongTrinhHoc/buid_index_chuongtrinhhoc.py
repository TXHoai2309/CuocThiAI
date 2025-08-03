#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build FAISS index cho Chuongtrinhhoc.json theo phong cách:
LangChain + FAISS + HuggingFaceEmbeddings

• Nhận diện khóa TIẾNG VIỆT trong dữ liệu thực tế:
  - Học phần: 'tên', 'số_tín_chỉ' (mã có thể không có)
  - Mô tả CTĐT/PLO: 'thông_tin_chi_tiết' và các khóa mô tả khác (gom từ str/dict/list)
  - Cấu trúc: 'khối_kiến_thức' → 'học_phần'/'học_phần_bắt_buộc'/'học_phần_tự_chọn'
• Sinh Document cho:
  - Học phần (section = "học phần")
  - Mô tả chương trình / PLO (section = "mô tả chương trình / PLO")
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ================== CLI ==================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build FAISS index cho Chuongtrinhhoc.json (LangChain)")
    p.add_argument("--input", default=r"D:\airdrop\CuocThiAI\HOU\ChuongTrinhHoc\Chuongtrinhhoc.json", help="Đường dẫn JSON đầu vào")
    p.add_argument("--out_dir", default="./data/ctdt_index", help="Thư mục lưu FAISS index")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                   help="HuggingFace sentence-transformers model")
    p.add_argument("--chunk_size", type=int, default=800)
    p.add_argument("--chunk_overlap", type=int, default=120)
    return p.parse_args()


# ================== tiện ích chuẩn hoá ==================
def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _join_nonempty(lines: List[str], sep: str = "\n") -> str:
    return sep.join([_normalize_space(l) for l in lines if _normalize_space(l)])


def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}


# ================== bộ khóa nhận diện ==================
MAJOR_KEYS = {
    "ngành", "nganh", "tennganh", "tên ngành", "program", "program_name",
    "chuongtrinh", "chương trình",
}
DEGREE_KEYS = {"bậc", "bac", "degree", "trinhdo", "trình độ"}
YEAR_KEYS = {"năm", "nam", "year", "khoahoc", "khoá", "khóa"}
FACULTY_KEYS = {"khoa", "khoa/viện", "viện", "faculty"}

# Học phần trong dữ liệu thực tế: có 'tên', 'số_tín_chỉ' (mã có thể không có)
COURSE_NAME_KEYS = {
    "ten_hp", "tenhocphan", "tên học phần", "ten mon", "ten_mon", "course_name",
    "hocphan", "tên", "học phần", "mon_hoc", "môn học", "tenmon"
}
COURSE_CODE_KEYS = {"ma_hp", "mahocphan", "mã học phần", "course_code", "ma_mon", "ma mon", "ma_mh", "mã môn"}
CREDIT_KEYS = {"so_tin_chi", "tin_chi", "sotinchi", "credits", "tc", "số_tín_chỉ"}
SEMESTER_KEYS = {"hoc_ky", "hocky", "semester", "ky", "kì"}
DESC_KEYS = {"mo_ta", "mota", "description", "gioi_thieu", "giới thiệu", "thông_tin_chi_tiết"}
PLO_KEYS = {"plo", "chuandaura_ctdt", "chuẩn đầu ra", "chuan_dau_ra", "program_learning_outcomes"}
CLO_KEYS = {"clo", "chuandaura_hocphan", "course_learning_outcomes"}
PREREQ_KEYS = {"hoc_phan_tien_quyet", "tienquyet", "prerequisite", "điều kiện tiên quyết"}


def _get_first(d: Dict[str, Any], keys: set) -> Optional[Any]:
    for k, v in d.items():
        if k in keys:
            return v
    return None


def _looks_like_course(obj: Dict[str, Any]) -> bool:
    L = _lower_keys(obj)
    has_name = any(k in L for k in COURSE_NAME_KEYS)
    has_credit_or_desc_or_clo = any(k in L for k in CREDIT_KEYS | DESC_KEYS | CLO_KEYS)
    # nới lỏng: chỉ cần 'tên' + (tín chỉ / mô tả / CLO)
    return has_name and has_credit_or_desc_or_clo


def _extract_course(obj: Dict[str, Any]) -> Dict[str, Any]:
    L = _lower_keys(obj)
    course = {
        "course_name": _safe_str(_get_first(L, COURSE_NAME_KEYS)) or "",
        "course_code": _safe_str(_get_first(L, COURSE_CODE_KEYS)) or "",  # có thể trống
        "credits": _get_first(L, CREDIT_KEYS),
        "semester": _get_first(L, SEMESTER_KEYS),
        "desc": _safe_str(_get_first(L, DESC_KEYS)) or "",
        "clo": _get_first(L, CLO_KEYS) or [],
        "prereq": _get_first(L, PREREQ_KEYS) or "",
    }
    # chuẩn hoá CLO về list[str]
    if isinstance(course["clo"], dict):
        course["clo"] = list(course["clo"].values())
    if isinstance(course["clo"], str):
        course["clo"] = [course["clo"]]
    return course


def _extract_context(obj_stack: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gom bối cảnh từ các node cha (khoa, ngành, bậc, năm, PLO chương trình...)."""
    ctx: Dict[str, Any] = {}
    for o in obj_stack:
        L = _lower_keys(o)
        ctx.setdefault("faculty", _safe_str(_get_first(L, FACULTY_KEYS)) or None)
        ctx.setdefault("major", _safe_str(_get_first(L, MAJOR_KEYS)) or None)
        ctx.setdefault("degree", _safe_str(_get_first(L, DEGREE_KEYS)) or None)
        ctx.setdefault("year", _get_first(L, YEAR_KEYS) or None)
        # PLO cấp chương trình
        plo = _get_first(L, PLO_KEYS)
        if plo is not None and ctx.get("plo") is None:
            if isinstance(plo, dict):
                plo = list(plo.values())
            if isinstance(plo, str):
                plo = [plo]
            ctx["plo"] = plo
    return ctx


# ================== duyệt JSON → Documents ==================
def _walk(
    data: Any,
    stack: List[Dict[str, Any]],
    collected: List[Document],
    stats: Dict[str, int]
) -> None:
    if isinstance(data, dict):
        stack.append(data)
        L = _lower_keys(data)

        # Node học phần
        if _looks_like_course(data):
            stats["course_like"] += 1
            ctx = _extract_context(stack)
            course = _extract_course(data)

            page_content = _join_nonempty([
                f"Ngành/CTĐT: {ctx.get('major')}",
                f"Bậc: {ctx.get('degree')}",
                f"Khoa/Viện: {ctx.get('faculty')}",
                f"Năm/Khoá: {ctx.get('year')}",
                "---",
                f"Học phần: {course['course_name']}" + (f" ({course['course_code']})" if course['course_code'] else ""),
                f"Tín chỉ: {course['credits']}",
                f"Học kỳ: {course['semester']}",
                f"Tiên quyết: {course['prereq']}",
                "Mô tả:",
                course["desc"],
                "CLO:" if course["clo"] else "",
                *[f"- {c}" for c in course["clo"]],
            ])

            metadata = {
                "section": "học phần",
                "major": ctx.get("major"),
                "degree": ctx.get("degree"),
                "faculty": ctx.get("faculty"),
                "year": ctx.get("year"),
                "course_name": course["course_name"],
                "course_code": course["course_code"],
                "credits": course["credits"],
                "semester": course["semester"],
            }
            collected.append(Document(page_content=page_content, metadata=metadata))

        else:
            # Node mô tả/PLO cấp CTĐT – gom text từ str/dict/list
            desc_vals: List[str] = []

            def _collect_strs(x):
                if isinstance(x, str):
                    s = _normalize_space(x)
                    if s:
                        desc_vals.append(s)
                elif isinstance(x, dict):
                    for vv in x.values():
                        _collect_strs(vv)
                elif isinstance(x, list):
                    for vv in x:
                        _collect_strs(vv)

            for k, v in L.items():
                if k in DESC_KEYS:
                    _collect_strs(v)

            plo_here = _get_first(L, PLO_KEYS)
            has_any_program_text = bool(desc_vals or plo_here is not None)

            if has_any_program_text:
                stats["program_desc_like"] += 1
                ctx = _extract_context(stack)
                plo_list = ctx.get("plo")
                if plo_here is not None:
                    plo_list = plo_here
                    if isinstance(plo_list, dict):
                        plo_list = list(plo_list.values())
                    if isinstance(plo_list, str):
                        plo_list = [plo_list]

                page_content = _join_nonempty([
                    f"Ngành/CTĐT: {ctx.get('major')}",
                    f"Bậc: {ctx.get('degree')}",
                    f"Khoa/Viện: {ctx.get('faculty')}",
                    f"Năm/Khoá: {ctx.get('year')}",
                    "---",
                    "Mô tả chương trình:",
                    _join_nonempty(desc_vals, sep="\n"),
                    "PLO:" if plo_list else "",
                    *[f"- {p}" for p in (plo_list or [])],
                ])

                metadata = {
                    "section": "mô tả chương trình / PLO",
                    "major": ctx.get("major"),
                    "degree": ctx.get("degree"),
                    "faculty": ctx.get("faculty"),
                    "year": ctx.get("year"),
                }
                collected.append(Document(page_content=page_content, metadata=metadata))

        # Duyệt sâu
        for v in data.values():
            _walk(v, stack, collected, stats)

        stack.pop()
        return

    if isinstance(data, list):
        for item in data:
            _walk(item, stack, collected, stats)
        return
    # primitive: bỏ qua


def load_documents(json_path: str) -> Tuple[List[Document], Dict[str, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    docs: List[Document] = []
    stats = {"course_like": 0, "program_desc_like": 0}
    _walk(payload, stack=[], collected=docs, stats=stats)

    # Khử trùng lặp
    seen = set()
    uniq_docs: List[Document] = []
    for d in docs:
        key = (
            d.page_content,
            d.metadata.get("section"),
            d.metadata.get("major"),
            d.metadata.get("course_code"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq_docs.append(d)

    return uniq_docs, stats


# ================== main pipeline ==================
def main():
    args = parse_args()
    input_file = args.input
    out_dir = os.path.abspath(args.out_dir)

    print("📥 Đang tải và xử lý dữ liệu chương trình học...")
    documents, stats = load_documents(input_file)
    print(f"🔎 Phát hiện: course-like={stats['course_like']}, program-desc-like={stats['program_desc_like']}")
    print(f"✅ Số document trước khi chia: {len(documents)}")

    print("🔪 Chia nhỏ tài liệu...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(documents)
    print(f"📄 Tổng số đoạn sau chia: {len(split_docs)}")

    if not split_docs:
        print("⚠️ Không có đoạn nào để nhúng. Kiểm tra lại cấu trúc JSON hoặc bộ khóa nhận diện.")
        return

    print("🧠 Nhúng dữ liệu (HuggingFaceEmbeddings, multilingual)...")
    embedding_model = HuggingFaceEmbeddings(model_name=args.model)
    vectordb = FAISS.from_documents(split_docs, embedding_model)

    os.makedirs(out_dir, exist_ok=True)
    print(f"💾 Lưu FAISS index vào: {out_dir}")
    vectordb.save_local(out_dir)

    print("🎉 Hoàn tất build index chương trình học!")


if __name__ == "__main__":
    main()
