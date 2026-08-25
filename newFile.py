import base64
import binascii
import io
import logging
import os
import time
import zipfile as zf
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("fusion_ess_extractor")

app = FastAPI(title="Fusion ESS Report Extractor")

# ---------------------------------------------------------------------------
# ORDS endpoint configuration — only used for the Account Analysis Report
# ---------------------------------------------------------------------------
ORDS_BASE = os.environ.get(
    "ORDS_BASE",
    "https://gd6b06f1176805b-paastestdb.adb.me-dubai-1.oraclecloudapps.com/ords/WKSP_XXCUST/GL_AAR",
)
GL_BALANCES_URL = os.environ.get("GL_BALANCES_URL", f"{ORDS_BASE}/GL_Balances")
GL_ITEMS_URL = os.environ.get("GL_ITEMS_URL", f"{ORDS_BASE}/GL_Items")

# Batch size for POST inserts — ORDS AutoREST batch-load endpoints reject
# very large payloads, so items (which can run into the tens of thousands)
# are chunked.
INSERT_BATCH_SIZE = int(os.environ.get("INSERT_BATCH_SIZE", "100000"))
HTTP_TIMEOUT = float(os.environ.get("ORDS_HTTP_TIMEOUT", "60"))

# NOTE / ASSUMPTION: the "delete before insert" step issues a plain HTTP
# DELETE against the collection URL (no ID in the path), which only clears
# all rows if your ORDS module has a handler for that. Standard ORDS
# AutoREST only supports DELETE on a single row via /GL_Items/{id}. If your
# endpoints don't support bulk DELETE this way, swap _delete_all() below
# for your actual bulk-delete mechanism.


class ReportRequest(BaseModel):
    document_content: str = Field(
        ...,
        description="Base64-encoded ZIP (DocumentContent) for a single ESS report.",
    )


REPORT_MARKERS = {
    "XLAAARPT": "Account Analysis Report",
    "GLTRBAL": "Trial Balance Report",
    "DATA_DS": "Supplier Balance Aging Report",
}

# Repeating "record" element per report type — this is the element we
# clear() after each one, bounding memory to ~one record instead of
# the whole document.
RECORD_TAG = {
    "Account Analysis Report": "CCID_S",
    "Trial Balance Report": "G_DETAIL",
    "Supplier Balance Aging Report": "ACCOUNT_SUMMARY",
}


def _to_number(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _decode_base64(document_content: str) -> bytes:
    logger.info("Decoding base64 document content (%d chars)", len(document_content))
    try:
        decoded = base64.b64decode(document_content, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.error("Failed to decode base64 document content: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid base64 document_content: {exc}")
    logger.info("Decoded document content: %d bytes", len(decoded))
    return decoded


def _find_xml_entry(zip_file: zf.ZipFile) -> str:
    namelist = zip_file.namelist()
    logger.debug("ZIP contains files: %s", namelist)
    xml_file = next((name for name in namelist if name.lower().endswith(".xml")), None)
    if xml_file is None:
        logger.error("No XML file found inside ZIP. Contents: %s", namelist)
        raise HTTPException(status_code=422, detail="No XML file found inside ZIP")
    return xml_file


def _detect_report_type(zip_file: zf.ZipFile, xml_name: str) -> str:
    """Peek at the first chunk of the entry instead of decoding the whole
    document to text just to run a substring check."""
    with zip_file.open(xml_name) as f:
        head = f.read(65536)
    head_text = head.decode("utf-8", errors="replace")
    for marker, report_name in REPORT_MARKERS.items():
        if marker in head_text:
            logger.info("Detected report type: %s (%s)", report_name, marker)
            return report_name
    logger.error("Unknown report type — no known marker in first 64KB of %s", xml_name)
    raise HTTPException(
        status_code=422,
        detail="Unknown report type. XML does not contain GLTRBAL, DATA_DS, or XLAAARPT.",
    )


def _parse_ccid(elem) -> dict:
    code_combination = elem.findtext(".//ACCOUNTING_CODE_COMBINATION")

    begin_dr_raw = _to_number(elem.findtext(".//ACCT_SUM_BAL_DR")) or 0.0
    begin_cr_raw = _to_number(elem.findtext(".//ACCT_SUM_BAL_CR")) or 0.0
    period_dr = _to_number(elem.findtext(".//ACCT_SUM_PR_DR")) or 0.0
    period_cr = _to_number(elem.findtext(".//ACCT_SUM_PR_CR")) or 0.0

    begin_net = begin_dr_raw - begin_cr_raw
    begin_balance_dr = begin_net if begin_net >= 0 else 0.0
    begin_balance_cr = -begin_net if begin_net < 0 else 0.0

    period_net = period_dr - period_cr
    ending_net = begin_net + period_net
    ending_balance_dr = ending_net if ending_net >= 0 else 0.0
    ending_balance_cr = -ending_net if ending_net < 0 else 0.0

    items = []
    for jeline in elem.findall(".//JELINE_ROW"):
        source = jeline.findtext(".//JE_SOURCE_NAME") or jeline.findtext(".//APPLICATION_NAME")
        number = jeline.findtext(".//TRANSACTION_NUMBER") or jeline.findtext(".//DOCUMENT_SEQUENCE_NUMBER")
        debit = _to_number(jeline.findtext(".//ACCOUNTED_DR")) or 0.0
        credit = _to_number(jeline.findtext(".//ACCOUNTED_CR")) or 0.0
        items.append(
            {
                "source": source,
                "number": number,
                "debitBalance": round(debit, 2),
                "creditBalance": round(credit, 2),
            }
        )

    return {
        "codeCombination": code_combination,
        "beginBalance_debit": round(begin_balance_dr, 2),
        "beginBalance_credit": round(begin_balance_cr, 2),
        "periodBalance_debit": round(period_dr, 2),
        "periodBalance_credit": round(period_cr, 2),
        "endingBalance_debit": round(ending_balance_dr, 2),
        "endingBalance_credit": round(ending_balance_cr, 2),
        "items": items,
    }


def _parse_trial_balance_row(elem) -> dict:
    return {
        "CodeCombination": elem.findtext(".//ACCT"),
        "BeginBalance": elem.findtext(".//BEGIN_BALANCE"),
        "TotalDebits": elem.findtext(".//TOTAL_DR"),
        "TotalCredits": elem.findtext(".//TOTAL_CR"),
        "EndBalance": elem.findtext(".//END_BALANCE"),
    }


def _parse_aging_row(elem) -> dict:
    return {
        "CodeCombination": elem.findtext(".//ACCOUNT"),
        "Amount": elem.findtext(".//AMOUNT"),
    }


RECORD_PARSER = {
    "Account Analysis Report": _parse_ccid,
    "Trial Balance Report": _parse_trial_balance_row,
    "Supplier Balance Aging Report": _parse_aging_row,
}


def _stream_parse(stream, report_name: str) -> list[dict]:
    """Stream the XML and clear() each record element the moment it's been
    read, instead of holding the full DOM tree in memory at once."""
    record_tag = RECORD_TAG[report_name]
    parse_fn = RECORD_PARSER[report_name]

    results = []
    found_any = False
    # Only need "end" events: by the time an element's end event fires,
    # all of its children have already been parsed.
    for event, elem in ET.iterparse(stream, events=("end",)):
        if elem.tag == record_tag:
            found_any = True
            results.append(parse_fn(elem))
            elem.clear()

    if not found_any:
        logger.error("%s XML had no %s rows", report_name, record_tag)
        raise HTTPException(status_code=422, detail=f"{report_name} XML had no {record_tag} rows")

    logger.info("Parsed %d %s rows", len(results), report_name)
    return results


def _get_report_records(document_content: str) -> tuple[str, list[dict]]:
    zip_bytes = _decode_base64(document_content)
    with zf.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
        xml_name = _find_xml_entry(zip_file)
        report_name = _detect_report_type(zip_file, xml_name)
        with zip_file.open(xml_name) as stream:
            records = _stream_parse(stream, report_name)
    return report_name, records


# ---------------------------------------------------------------------------
# ORDS I/O helpers — only used for the Account Analysis Report
# ---------------------------------------------------------------------------
def _delete_all(url: str, label: str) -> None:
    t0 = time.perf_counter()
    print(f"[DELETE] {label}: clearing existing rows at {url} ...")
    resp = requests.delete(url, timeout=HTTP_TIMEOUT)
    elapsed = time.perf_counter() - t0
    if resp.status_code not in (200, 204):
        logger.error("Delete failed for %s: %s %s", label, resp.status_code, resp.text)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to delete existing {label} rows: {resp.status_code} {resp.text}",
        )
    print(f"[DELETE] {label}: done in {elapsed:.2f}s")
    logger.info("Deleted existing %s rows in %.2fs", label, elapsed)


def _insert_batched(url: str, records: list[dict], label: str) -> int:
    t0 = time.perf_counter()
    total = len(records)
    inserted = 0
    num_batches = (total + INSERT_BATCH_SIZE - 1) // INSERT_BATCH_SIZE or 1
    print(f"[INSERT] {label}: inserting {total} records in {num_batches} batch(es) of up to {INSERT_BATCH_SIZE} ...")

    for batch_num, start in enumerate(range(0, total, INSERT_BATCH_SIZE) or [0], start=1):
        chunk = records[start:start + INSERT_BATCH_SIZE]
        if not chunk:
            break
        batch_t0 = time.perf_counter()
        resp = requests.post(url, json={"items": chunk}, timeout=HTTP_TIMEOUT)
        batch_elapsed = time.perf_counter() - batch_t0
        if resp.status_code not in (200, 201):
            logger.error(
                "Insert failed for %s batch %d/%d: %s %s",
                label, batch_num, num_batches, resp.status_code, resp.text,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to insert {label} batch {batch_num}/{num_batches}: {resp.status_code} {resp.text}",
            )
        inserted += len(chunk)
        print(f"[INSERT] {label}: batch {batch_num}/{num_batches} ({len(chunk)} rows) in {batch_elapsed:.2f}s")

    elapsed = time.perf_counter() - t0
    print(f"[INSERT] {label}: inserted {inserted}/{total} rows total in {elapsed:.2f}s")
    logger.info("Inserted %d/%d %s rows in %.2fs", inserted, total, label, elapsed)
    return inserted


def _load_account_analysis_report(records: list[dict]) -> dict:
    """Split parsed CCID_S records into the two DB table shapes, delete
    existing rows, then insert. Order: delete items -> delete balances ->
    insert balances -> insert items."""
    overall_t0 = time.perf_counter()
    print("=== Account Analysis Report load: starting ===")

    balances = []
    items = []
    for record in records:
        code_combination = record["codeCombination"]
        record_items = record["items"]
        balances.append({k: v for k, v in record.items() if k != "items"})
        for item in record_items:
            items.append({"codeCombination": code_combination, **item})

    # Delete order: items first, then balances (children before parent).
    _delete_all(GL_ITEMS_URL, "GL_Items")
    _delete_all(GL_BALANCES_URL, "GL_Balances")

    # Insert order: balances first, then items (parent before children).
    balances_inserted = _insert_batched(GL_BALANCES_URL, balances, "GL_Balances")
    items_inserted = _insert_batched(GL_ITEMS_URL, items, "GL_Items")

    overall_elapsed = time.perf_counter() - overall_t0
    print(f"=== Account Analysis Report load: complete in {overall_elapsed:.2f}s ===")
    logger.info("Full load complete in %.2fs", overall_elapsed)

    return {
        "report_name": REPORT_MARKERS["XLAAARPT"],
        "status": "success",
        "balances": {"deleted": True, "inserted": balances_inserted},
        "items": {"deleted": True, "inserted": items_inserted},
        "elapsed_seconds": round(overall_elapsed, 2),
    }


@app.post("/report")
def get_report(payload: ReportRequest):
    logger.info("Received request for /report (base64 document_content)")
    report_name, records = _get_report_records(payload.document_content)

    if report_name == "Account Analysis Report":
        # Persist to DB instead of returning the raw parsed data.
        result = _load_account_analysis_report(records)
        return JSONResponse(content=result)

    # Trial Balance / Supplier Balance Aging: unchanged behavior.
    logger.info("Returning %d records for report_name=%s", len(records), report_name)
    return JSONResponse(
        content={"report_name": report_name, "count": len(records), "data": records}
    )