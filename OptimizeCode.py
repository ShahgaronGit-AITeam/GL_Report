import base64
import binascii
import io
import logging
import os
import zipfile as zf
import xml.etree.ElementTree as ET

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


class ReportRequest(BaseModel):
    document_content: str = Field(
        ...,
        description="Base64-encoded ZIP (DocumentContent) for a single ESS report.",
    )


class CompareRequest(BaseModel):
    trial_balance_document_content: str = Field(
        ..., description="Base64-encoded ZIP for the Trial Balance report."
    )
    aging_document_content: str = Field(
        ..., description="Base64-encoded ZIP for the Supplier Balance Aging report."
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
            elem.clear()  # drop this record's children/text/attribs now

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


@app.post("/report")
def get_report(payload: ReportRequest):
    logger.info("Received request for /report (base64 document_content)")
    report_name, records = _get_report_records(payload.document_content)
    logger.info("Returning %d records for report_name=%s", len(records), report_name)
    return JSONResponse(
        content={"report_name": report_name, "count": len(records), "data": records,"items_count": sum(len(record.get("items", [])) for record in records),
 }
    )
