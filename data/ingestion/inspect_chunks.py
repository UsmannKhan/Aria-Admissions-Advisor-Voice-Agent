# inspect_chunks.py
import asyncio, json
import config as cfg
from crawler import crawl_entity, chunk_markdown

async def main():
    with open(cfg.ENTITIES_CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    universal = data["universal_settings"]
    for eid in ["lums", "habib"]:
        entity = data["entities"][eid]
        pages = await crawl_entity(eid, entity, universal, limit=1)
        for page in pages:
            chunks = chunk_markdown(page.markdown)
            print(f"\n{'='*70}\n{eid}: {len(chunks)} chunks from {page.url}\n{'='*70}")
            for i, ch in enumerate(chunks):
                preview = ch.text[:200].replace("\n", " ")
                print(f"\n[{i}] headings={ch.headings} | {len(ch.text)} chars")
                print(f"    {preview}")

asyncio.run(main())