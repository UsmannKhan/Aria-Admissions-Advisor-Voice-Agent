import argparse
from collections import Counter, defaultdict

import config as cfg
import chromadb


def get_collection():
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DB_PATH))
    return client.get_collection(cfg.CHROMA_COLLECTION_NAME)


def load_all(col):
    got = col.get(include=["documents", "metadatas"])
    return got["ids"], got["documents"], got["metadatas"]


def summary(col):
    ids, docs, metas = load_all(col)
    print(f"collection: {cfg.CHROMA_COLLECTION_NAME}   total chunks: {len(ids)}\n")

    by_entity = Counter(m.get("entity_id", "?") for m in metas)
    print("chunks per entity:")
    for e, n in sorted(by_entity.items(), key=lambda x: -x[1]):
        print(f"  {e:14} {n:5}")

    print("\nfreshness_class distribution (watch for cycle_bound inflation):")
    for k, n in Counter(m.get("freshness_class", "?") for m in metas).most_common():
        print(f"  {k:16} {n:5}")

    print("\ncontent_category distribution:")
    for k, n in Counter(m.get("content_category", "?") for m in metas).most_common():
        print(f"  {k:22} {n:5}")

    # chunk-length shape: tiny chunks (<300) or maxed chunks (near MAX) are the
    # signals for over-splitting / merge failures worth eyeballing
    lens = sorted(len(d) for d in docs)
    if lens:
        p = lambda q: lens[int(q * (len(lens) - 1))]
        print(f"\nchunk length chars: min={lens[0]} p25={p(.25)} "
              f"median={p(.5)} p75={p(.75)} max={lens[-1]}")
        print(f"  tiny (<300): {sum(1 for l in lens if l < 300)}   "
              f"near-max (>{cfg.MAX_CHUNK_CHARS-200}): "
              f"{sum(1 for l in lens if l > cfg.MAX_CHUNK_CHARS - 200)}")


def show_entity(col, entity, full):
    ids, docs, metas = load_all(col)
    rows = [(i, d, m) for i, d, m in zip(ids, docs, metas)
            if m.get("entity_id") == entity]
    print(f"{entity}: {len(rows)} chunks\n")
    for cid, doc, m in rows:
        head = m.get("headings", "")
        print(f"\n{'='*80}\n{cid}\n  headings: {head}"
              f"\n  cat={m.get('content_category')} fresh={m.get('freshness_class')} "
              f"pos={m.get('chunk_position')}/{m.get('total_chunks_in_page')} "
              f"len={len(doc)}")
        mentions = {k: m.get(k) for k in
                    ("mentioned_amounts", "mentioned_dates", "mentioned_programs",
                     "mentioned_scholarships", "mentioned_exams") if m.get(k)}
        if mentions:
            print(f"  mentions: {mentions}")
        print("-" * 80)
        print(doc if full else doc[:400] + ("..." if len(doc) > 400 else ""))


def show_url(col, url_sub):
    """All chunks for one page, in order, with boundary view: end of each
    chunk + start of the next, so you can see if a thought split across them."""
    ids, docs, metas = load_all(col)
    rows = [(i, d, m) for i, d, m in zip(ids, docs, metas)
            if url_sub in m.get("source_url", "")]
    rows.sort(key=lambda r: r[2].get("chunk_position", 0))
    if not rows:
        print(f"no chunks whose source_url contains {url_sub!r}")
        return
    print(f"{rows[0][2].get('source_url')}\n{len(rows)} chunks in order\n")
    for idx, (cid, doc, m) in enumerate(rows):
        print(f"\n[{m.get('chunk_position')}] headings={m.get('headings')} len={len(doc)}")
        print(f"  FULL: {doc}")
        if idx < len(rows) - 1:
            nxt = rows[idx + 1][1]
            print(f"\n  >>> BOUNDARY >>> ...{doc[-160:]!r}")
            print(f"      NEXT STARTS >>> {nxt[:160]!r}")
            print("  ^ does the next chunk read as a fresh unit, or a cut-off continuation?")


def grep(col, needle):
    ids, docs, metas = load_all(col)
    hits = [(i, d, m) for i, d, m in zip(ids, docs, metas) if needle in d]
    print(f"{len(hits)} chunk(s) contain {needle!r}\n")
    for cid, doc, m in hits:
        pos = doc.find(needle)
        ctx = doc[max(0, pos - 120): pos + len(needle) + 120]
        print(f"  {m.get('entity_id')} | {m.get('headings')}")
        print(f"    source: {m.get('source_url')}")
        print(f"    ...{ctx}...\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity")
    ap.add_argument("--url", help="substring of source_url; shows chunks in order + boundaries")
    ap.add_argument("--grep", help="find which chunk holds a fact/number")
    ap.add_argument("--full", action="store_true", help="print full chunk text (with --entity)")
    ap.add_argument("--meta", action="store_true", help="metadata summary only")
    ap.add_argument("--dump", action="store_true", help="write ALL chunks (full) to all_chunks.txt")
    args = ap.parse_args()

    col = get_collection()
    if args.dump:
        dump_all(col)
    elif args.grep:
        grep(col, args.grep)
    elif args.url:
        show_url(col, args.url)
    elif args.entity:
        show_entity(col, args.entity, full=args.full or True)
    else:
        summary(col)


def dump_all(col, out_path="all_chunks.txt"):
    """Write every chunk (full text + metadata) grouped by entity, ordered by
    source_url then chunk_position, so the file reads like the source pages."""
    ids, docs, metas = load_all(col)
    rows = list(zip(ids, docs, metas))
    # group by entity, then url, then position
    rows.sort(key=lambda r: (
        r[2].get("entity_id", ""),
        r[2].get("source_url", ""),
        r[2].get("chunk_position", 0),
    ))
    with open(out_path, "w", encoding="utf-8") as f:
        cur_entity = cur_url = None
        for cid, doc, m in rows:
            e = m.get("entity_id", "?")
            u = m.get("source_url", "?")
            if e != cur_entity:
                f.write(f"\n\n{'#'*90}\n# ENTITY: {e}\n{'#'*90}\n")
                cur_entity = e
                cur_url = None
            if u != cur_url:
                f.write(f"\n\n{'='*90}\nPAGE: {u}\n{'='*90}\n")
                cur_url = u
            f.write(
                f"\n[pos {m.get('chunk_position')}/{m.get('total_chunks_in_page')}] "
                f"cat={m.get('content_category')} fresh={m.get('freshness_class')} "
                f"len={len(doc)}\n"
                f"headings: {m.get('headings','')}\n"
            )
            mentions = {k.replace('mentioned_',''): m.get(k) for k in
                        ('mentioned_amounts','mentioned_dates','mentioned_programs',
                         'mentioned_scholarships','mentioned_exams') if m.get(k)}
            if mentions:
                f.write(f"mentions: {mentions}\n")
            f.write("-"*90 + "\n")
            f.write(doc + "\n")
    print(f"wrote {len(rows)} chunks to {out_path}")


if __name__ == "__main__":
    main()