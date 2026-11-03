"""Task 2 data: COCO captions, annotation-derived grounding pairs and VQAv2 complementary pairs.

    python -m task2_vlm.data download      # annotation JSONs (COCO 2014 + VQAv2) into data/task2_raw
    python -m task2_vlm.data select        # fixed selections -> data/task2/*.json (seed 6304)
    python -m task2_vlm.data images        # download only the selected images by URL

Every eval (dev/test) set uses val2014 images; a single `used_val` set keeps all dev/test sets image-disjoint
from one another (training uses train2014 only, so it is disjoint by construction).
Grounding questions are generated programmatically:
  presence  "Does this image contain a {category}?"  image with >=1 annotation (yes) vs image with none (no)
  counting  "How many {category plural} can you see?"  two images with different counts in [1, 4]
Counting only uses images whose instances of the category are all non-crowd (iscrowd=0); presence "no" images
have no annotation of the category at all (crowd included). These choices are recorded in selection_info.json.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common.utils import ensure_dir, load_cfg, load_json, save_json
from task2_vlm.vqa_norm import normalize_answer

URLS = {
    "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "v2_Questions_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "v2_Questions_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_Annotations_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "v2_Complementary_Pairs_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "v2_Complementary_Pairs_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}
PLURAL_EXCEPTIONS = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
                     "bus": "buses", "couch": "couches", "wine glass": "wine glasses", "bench": "benches",
                     "sandwich": "sandwiches", "toothbrush": "toothbrushes", "knife": "knives", "skateboard": "skateboards",
                     "dining table": "dining tables", "potted plant": "potted plants", "hair drier": "hair driers"}


def plural(name: str) -> str:
    if name in PLURAL_EXCEPTIONS:
        return PLURAL_EXCEPTIONS[name]
    return name + ("es" if name.endswith(("s", "x", "ch", "sh")) else "s")


def article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def image_url(split: str, image_id: int) -> str:
    return f"http://images.cocodataset.org/{split}/COCO_{split}_{image_id:012d}.jpg"


# ------------------------------------------------------------------ download
def download(cfg):
    import urllib.request

    raw = ensure_dir(cfg["data"]["raw_dir"])
    for fn, url in URLS.items():
        dst = raw / fn
        if not dst.exists():
            print("downloading", url, flush=True)
            urllib.request.urlretrieve(url, dst)
        with zipfile.ZipFile(dst) as z:
            for m in z.namelist():
                if m.endswith(".json") and not (raw / Path(m).name).exists():
                    with z.open(m) as src, open(raw / Path(m).name, "wb") as out:
                        out.write(src.read())
    print(sorted(p.name for p in raw.glob("*.json")))


# ------------------------------------------------------------------ selection helpers
def load_instances(raw: Path, split: str):
    d = json.load(open(raw / f"instances_{split}.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    counts = defaultdict(lambda: defaultdict(int))   # image -> cat -> non-crowd count
    crowd = defaultdict(set)                          # image -> cats with a crowd annotation
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            crowd[a["image_id"]].add(a["category_id"])
        else:
            counts[a["image_id"]][a["category_id"]] += 1
    images = sorted(im["id"] for im in d["images"])
    return cats, counts, crowd, images


def presence_pairs(rng, cats, counts, crowd, images, n_pairs, used, allowed=None):
    """n_pairs (yes, no) pairs; each pair uses two images not in `used` (updated in place)."""
    pool = [i for i in images if (allowed is None or i in allowed)]
    with_cat = defaultdict(list)
    for i in pool:
        for c, n in counts[i].items():
            if n > 0:
                with_cat[c].append(i)
    out, tries = [], 0
    cat_ids = sorted(c for c in cats if with_cat[c])   # category drawn uniformly, then a positive image
    seen = set()
    while len(out) < n_pairs and tries < 200000:
        tries += 1
        c = cat_ids[rng.integers(len(cat_ids))]
        a, b = with_cat[c][rng.integers(len(with_cat[c]))], pool[rng.integers(len(pool))]
        if a == b or (used is not None and (a in used or b in used)):
            continue
        has_a = counts[a].get(c, 0) > 0
        has_b = counts[b].get(c, 0) > 0 or c in crowd[b]
        if has_a and not has_b and (c, a, b) not in seen:
            seen.add((c, a, b))
            q = f"Does this image contain {article(cats[c])}?"
            out.append({"pair_id": f"pres_{len(out)}", "kind": "presence", "category": cats[c], "question": q,
                        "items": [{"image_id": a, "answer": "yes"}, {"image_id": b, "answer": "no"}]})
            if used is not None:
                used.update([a, b])
    return out


def counting_pairs(rng, cats, counts, crowd, images, n_pairs, used, lo, hi, allowed=None):
    pool = [i for i in images if (allowed is None or i in allowed)]
    by_cat_count = defaultdict(lambda: defaultdict(list))
    for i in pool:
        for c, n in counts[i].items():
            if lo <= n <= hi and c not in crowd[i]:
                by_cat_count[c][n].append(i)
    cat_ids = sorted(c for c in by_cat_count if len(by_cat_count[c]) >= 2)
    out, tries, seen = [], 0, set()
    while len(out) < n_pairs and tries < 200000:
        tries += 1
        c = cat_ids[rng.integers(len(cat_ids))]
        ns = sorted(by_cat_count[c])
        n1, n2 = rng.choice(ns, 2, replace=False)
        a = by_cat_count[c][n1][rng.integers(len(by_cat_count[c][n1]))]
        b = by_cat_count[c][n2][rng.integers(len(by_cat_count[c][n2]))]
        if a == b or (used is not None and (a in used or b in used)):
            continue
        if used is None and (c, a, b) in seen:
            continue
        seen.add((c, a, b))
        q = f"How many {plural(cats[c])} can you see?"
        out.append({"pair_id": f"count_{len(out)}", "kind": "counting", "category": cats[c], "question": q,
                    "items": [{"image_id": a, "answer": str(int(n1))}, {"image_id": b, "answer": str(int(n2))}]})
        if used is not None:
            used.update([a, b])
    return out


def vqa_candidates(raw: Path, split: str, min_agree: int):
    """Complementary pairs whose normalised consensus answers differ, each with >= min_agree/10 agreement."""
    tag = "train2014" if split == "train" else "val2014"
    qs = {q["question_id"]: q for q in json.load(open(raw / f"v2_OpenEnded_mscoco_{tag}_questions.json"))["questions"]}
    anns = {a["question_id"]: a for a in json.load(open(raw / f"v2_mscoco_{tag}_annotations.json"))["annotations"]}
    pairs = json.load(open(raw / f"v2_mscoco_{tag}_complementary_pairs.json"))
    out = []
    for q1, q2 in pairs:
        items = []
        for qid in (q1, q2):
            a = anns[qid]
            raw_answers = [x["answer"] for x in a["answers"]]
            norm = [normalize_answer(x) for x in raw_answers]
            cons, n = Counter(norm).most_common(1)[0]
            if n < min_agree:
                break
            # original answer string from the consensus group (most frequent raw spelling, ties -> first)
            target = Counter([r for r, nn in zip(raw_answers, norm) if nn == cons]).most_common(1)[0][0]
            items.append({"question_id": qid, "image_id": a["image_id"], "answer": target, "consensus": cons,
                          "answers": raw_answers, "answer_type": a["answer_type"]})
        if len(items) < 2 or items[0]["consensus"] == items[1]["consensus"]:
            continue
        if qs[q1]["question"] != qs[q2]["question"] or items[0]["image_id"] == items[1]["image_id"]:
            continue
        out.append({"pair_id": f"vqa_{q1}_{q2}", "kind": "vqa", "question": qs[q1]["question"],
                    "answer_type": items[0]["answer_type"], "items": items})
    return out


def sample_vqa(rng, cands, quotas: dict, used: set | None):
    by_type = defaultdict(list)
    for p in cands:
        by_type[p["answer_type"]].append(p)
    out = []
    for t, n in quotas.items():
        pool = by_type[t]
        order = rng.permutation(len(pool))
        got = 0
        for i in order:
            p = pool[i]
            ims = {it["image_id"] for it in p["items"]}
            if used is not None and ims & used:
                continue
            out.append(p)
            if used is not None:
                used.update(ims)
            got += 1
            if got == n:
                break
        assert got == n, f"only {got} {t} pairs available"
    return out


def select(cfg):
    dc = cfg["data"]
    raw, sd = Path(dc["raw_dir"]), ensure_dir(dc["split_dir"])
    rng = np.random.default_rng(cfg["seed"])
    used_val: set[int] = set()
    info = {"seed": cfg["seed"], "counting_rule": "non-crowd instance count in [1,4]; no crowd annotation of category",
            "presence_rule": "yes: >=1 non-crowd instance; no: no annotation (incl. crowd) of category"}

    # final-test sets first, then dev sets (all val2014, mutually image-disjoint)
    vqa_val = vqa_candidates(raw, "val", dc["min_agreement"])
    vqa_test = sample_vqa(rng, vqa_val, dc["vqa_test_pairs"], used_val)
    cats, vcounts, vcrowd, vimages = load_instances(raw, "val2014")
    g_test = (presence_pairs(rng, cats, vcounts, vcrowd, vimages, dc["grounding_test_pairs"]["presence"], used_val)
              + counting_pairs(rng, cats, vcounts, vcrowd, vimages, dc["grounding_test_pairs"]["counting"], used_val,
                               *dc["count_range"]))
    vqa_dev = sample_vqa(rng, vqa_val, dc["vqa_dev_pairs"], used_val)
    g_dev = (presence_pairs(rng, cats, vcounts, vcrowd, vimages, dc["grounding_dev_pairs"]["presence"], used_val)
             + counting_pairs(rng, cats, vcounts, vcrowd, vimages, dc["grounding_dev_pairs"]["counting"], used_val,
                              *dc["count_range"]))
    vcap = json.load(open(raw / "captions_val2014.json"))
    cap_by_img = defaultdict(list)
    for a in sorted(vcap["annotations"], key=lambda a: a["id"]):
        cap_by_img[a["image_id"]].append(a["caption"].strip())
    free = [i for i in sorted(cap_by_img) if i not in used_val]
    dev_imgs = [free[i] for i in rng.choice(len(free), dc["caption_dev_images"], replace=False)]
    used_val.update(dev_imgs)
    cap_dev = [{"image_id": i, "split": "val2014", "caption": cap_by_img[i][0]} for i in dev_imgs]

    # training (train2014)
    tcap = json.load(open(raw / "captions_train2014.json"))
    tcap_by = defaultdict(list)
    for a in sorted(tcap["annotations"], key=lambda a: a["id"]):
        tcap_by[a["image_id"]].append(a["caption"].strip())
    timgs = sorted(i for i, c in tcap_by.items() if len(c) >= dc["captions_per_train_image"])
    sel = [timgs[i] for i in rng.choice(len(timgs), dc["caption_train_images"], replace=False)]
    cap_train = [{"image_id": i, "split": "train2014", "caption": c}
                 for i in sel for c in tcap_by[i][: dc["captions_per_train_image"]]]
    tcats, tcounts, tcrowd, timages = load_instances(raw, "train2014")
    gpool = set(timages[i] for i in rng.choice(len(timages), dc["grounding_train_pool_images"], replace=False))
    # reusable training pools: images may recur across pairs, identical (category, images) pairs are not repeated
    pres_pool = presence_pairs(rng, tcats, tcounts, tcrowd, timages, 4000, used=None, allowed=gpool)
    count_pool = counting_pairs(rng, tcats, tcounts, tcrowd, timages, 4000, used=None, lo=dc["count_range"][0],
                                hi=dc["count_range"][1], allowed=gpool)
    vqa_tr = sample_vqa(rng, vqa_candidates(raw, "train", dc["min_agreement"]), dc["vqa_train_pairs"], used=None)

    sets = {"caption_train": cap_train, "caption_dev": cap_dev, "presence_train_pool": pres_pool,
            "counting_train_pool": count_pool, "vqa_train": vqa_tr, "grounding_dev": g_dev, "grounding_test": g_test,
            "vqa_dev": vqa_dev, "vqa_test": vqa_test}
    for name, s in sets.items():
        save_json(s, sd / f"{name}.json", indent=None)
        info[name] = len(s)
    for name in ["grounding_dev", "grounding_test", "vqa_dev", "vqa_test"]:
        info[f"{name}_by_kind"] = dict(Counter(p.get("answer_type", p["kind"]) for p in sets[name]))
    info["n_val_images_used"] = len(used_val)
    save_json(info, sd / "selection_info.json")
    print(info)


def all_images(cfg) -> list[tuple[str, int]]:
    sd = Path(cfg["data"]["split_dir"])
    ims = set()
    for name in ["caption_train", "caption_dev"]:
        ims |= {(x["split"], x["image_id"]) for x in load_json(sd / f"{name}.json")}
    for name, split in [("presence_train_pool", "train2014"), ("counting_train_pool", "train2014"),
                        ("vqa_train", "train2014"), ("grounding_dev", "val2014"), ("grounding_test", "val2014"),
                        ("vqa_dev", "val2014"), ("vqa_test", "val2014")]:
        for p in load_json(sd / f"{name}.json"):
            ims |= {(split, it["image_id"]) for it in p["items"]}
    return sorted(ims)


def download_images(cfg, workers: int = 32):
    import urllib.request

    root = ensure_dir(cfg["data"]["image_dir"])
    todo = [(s, i) for s, i in all_images(cfg) if not (root / f"{i}.jpg").exists()]
    print(f"{len(todo)} images to download")

    def get(si):
        s, i = si
        for _ in range(3):
            try:
                urllib.request.urlretrieve(image_url(s, i), root / f"{i}.jpg.part")
                os.replace(root / f"{i}.jpg.part", root / f"{i}.jpg")
                return None
            except Exception as e:  # noqa
                err = e
        return (s, i, str(err))

    with ThreadPoolExecutor(workers) as ex:
        fails = [f for f in ex.map(get, todo) if f]
    print("failed:", fails[:10], len(fails))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("step", choices=["download", "select", "images"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    {"download": download, "select": select, "images": download_images}[args.step](cfg)


if __name__ == "__main__":
    main()
