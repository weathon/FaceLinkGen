"""Evaluate 300 identities with Face++ and Amazon, reporting at-least-one."""

import argparse
import io
import json
import os
import time

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"))
FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"
EDENAI_URL = "https://api.edenai.run/v2/image/face_compare"


def compare_facepp(generated, real, source, image_index):
    while True:
        generated_buffer = io.BytesIO()
        real_buffer = io.BytesIO()
        generated.convert("RGB").resize(
            (256, 256), Image.LANCZOS
        ).save(generated_buffer, format="JPEG")
        real.resize((256, 256), Image.LANCZOS).save(
            real_buffer, format="JPEG"
        )
        generated_buffer.seek(0)
        real_buffer.seek(0)
        print("Face++ source=%s generation=%d" % (
            source, image_index
        ), flush=True)
        response = requests.post(
            FACEPP_URL,
            data={
                "api_key": os.environ["FACEPP_API_KEY"],
                "api_secret": os.environ["FACEPP_API_SECRET"],
            },
            files={
                "image_file1": (
                    "generated.jpg", generated_buffer, "image/jpeg"
                ),
                "image_file2": ("real.jpg", real_buffer, "image/jpeg"),
            },
            timeout=60,
        )
        result = response.json()
        if (
            "error_message" in result
            and "CONCURRENCY_LIMIT_EXCEEDED" in result["error_message"]
        ):
            print("Face++ concurrency limit; retrying source=%s" % source, flush=True)
            time.sleep(2)
            continue
        response.raise_for_status()
        if "error_message" in result:
            raise RuntimeError(
                "Face++ source=%s generation=%d error=%s" % (
                    source, image_index, result["error_message"]
                )
            )
        if not result["faces1"] or not result["faces2"]:
            matched = False
        else:
            matched = (
                result["confidence"] > result["thresholds"]["1e-5"]
            )
        return matched, result


def compare_amazon(generated, real, source, image_index):
    generated_buffer = io.BytesIO()
    real_buffer = io.BytesIO()
    generated.convert("RGB").resize(
        (256, 256), Image.LANCZOS
    ).save(generated_buffer, format="JPEG")
    real.resize((256, 256), Image.LANCZOS).save(
        real_buffer, format="JPEG"
    )
    generated_buffer.seek(0)
    real_buffer.seek(0)
    print("Amazon source=%s generation=%d" % (
        source, image_index
    ), flush=True)
    response = requests.post(
        EDENAI_URL,
        data={"providers": "amazon"},
        files={
            "file1": ("generated.jpg", generated_buffer, "image/jpeg"),
            "file2": ("real.jpg", real_buffer, "image/jpeg"),
        },
        headers={
            "Authorization": "Bearer %s" % os.environ["EDENAI_API_TOKEN"]
        },
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()
    node = result["amazon"]
    if node["status"] == "success":
        matched = len(node["items"]) > 0
        no_face = False
    elif node["error"]["type"] == "ProviderInvalidInputError":
        matched = False
        no_face = True
    else:
        raise RuntimeError(
            "Amazon source=%s generation=%d error=%s" % (
                source, image_index, node["error"]
            )
        )
    return matched, no_face, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=["ours", "unet"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reconstruction-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.attack == "ours":
        assert args.reconstruction_root is not None
    else:
        assert args.reconstruction_root is None

    os.makedirs(args.output, exist_ok=True)
    input_records = []
    with open(args.manifest) as f:
        for line in f:
            input_records.append(json.loads(line))
    input_records.sort(key=lambda record: record["index"])
    assert len(input_records) == 300

    records_path = os.path.join(args.output, "records.jsonl")
    completed = set()
    if os.path.exists(records_path):
        with open(records_path) as f:
            for line in f:
                completed.add(json.loads(line)["index"])

    for input_record in tqdm(input_records, desc="API evaluation"):
        if input_record["index"] in completed:
            continue

        source = input_record["source"]
        real = Image.open(source).convert("RGB")
        if args.attack == "ours":
            complete_path = os.path.join(
                args.reconstruction_root,
                "%04d" % input_record["index"],
                "complete.json",
            )
            with open(complete_path) as f:
                generated_paths = json.load(f)["images"]
            assert len(generated_paths) == 5
        else:
            generated_paths = [input_record["output"]]

        comparisons = []
        for image_index, generated_path in enumerate(generated_paths):
            generated = Image.open(generated_path).convert("RGB")
            facepp_match, facepp_response = compare_facepp(
                generated, real, source, image_index
            )
            amazon_match, amazon_no_face, amazon_response = compare_amazon(
                generated, real, source, image_index
            )
            comparisons.append({
                "image": generated_path,
                "facepp_match_1e-5": facepp_match,
                "facepp_response": facepp_response,
                "amazon_match": amazon_match,
                "amazon_no_face": amazon_no_face,
                "amazon_response": amazon_response,
            })

        record = {
            "index": input_record["index"],
            "source": source,
            "facepp_at_least_one": any(
                comparison["facepp_match_1e-5"]
                for comparison in comparisons
            ),
            "amazon_at_least_one": any(
                comparison["amazon_match"]
                for comparison in comparisons
            ),
            "comparisons": comparisons,
        }
        with open(records_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    records = []
    with open(records_path) as f:
        for line in f:
            records.append(json.loads(line))
    assert len(records) == 300
    summary = {
        "n": len(records),
        "facepp_at_least_one": sum(
            record["facepp_at_least_one"] for record in records
        ),
        "facepp_at_least_one_rate": sum(
            record["facepp_at_least_one"] for record in records
        ) / len(records),
        "amazon_at_least_one": sum(
            record["amazon_at_least_one"] for record in records
        ),
        "amazon_at_least_one_rate": sum(
            record["amazon_at_least_one"] for record in records
        ) / len(records),
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
