from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/wg25r/review_agent/.env")

from datalab_sdk import ConvertOptions, DatalabClient


root_dir = Path(__file__).parent
pdf_dir = root_dir / "pdf"
md_dir = root_dir / "md"
pdf_paths = sorted(pdf_dir.glob("*.pdf"))

if not pdf_paths:
    raise RuntimeError(f"No PDF files found in {pdf_dir}")

md_dir.mkdir(exist_ok=True)
client = DatalabClient()
options = ConvertOptions(
    output_format="markdown",
    mode="balanced",
    paginate=True,
)

for pdf_path in pdf_paths:
    md_path = md_dir / f"{pdf_path.stem}.md"
    if md_path.exists():
        print(f"Skipping completed conversion: {md_path}")
        continue

    print(f"Converting {pdf_path} -> {md_path}")
    result = client.convert(pdf_path, options=options, poll_interval=60)
    md_path.write_text(result.markdown, encoding="utf-8")
    print(f"Saved {md_path} ({len(result.markdown):,} characters)")
