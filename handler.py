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
from collections import defaultdict

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

class XMLRequest(BaseModel):
    xml_data: str

def parse_gl_xml(xml_data: str):
    root = ET.fromstring(xml_data)

    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    def text(parent, tag, default=""):
        node = parent.find(tag) if parent is not None else None
        return node.text.strip() if node is not None and node.text else default

    source_totals = defaultdict(
        lambda: {"count": 0, "total_debit": 0.0, "total_credit": 0.0}
    )

    for g3 in root.findall(".//G_2/G_3"):
        source = text(g3, "SOURCE_DESC")
        debit = float(text(g3, "ACCOUNTED_DR", "0") or 0)
        credit = float(text(g3, "ACCOUNTED_CR", "0") or 0)

        source_totals[source]["count"] += 1
        source_totals[source]["total_debit"] += debit
        source_totals[source]["total_credit"] += credit

    g5 = root.find(".//G_5")
    g6 = root.find(".//G_6")

    code = text(g5, "CODE_COMBINATION")

    begin_debit = float(text(g5, "BEGIN_BALANCE_DR", "0") or 0)
    begin_credit = float(text(g5, "BEGIN_BALANCE_CR", "0") or 0)

    ending_debit = float(text(g6, "CLOSING_BALANCE_DR", "0") or 0)
    ending_credit = float(text(g6, "CLOSING_BALANCE_CR", "0") or 0)

    period_debit = sum(x["total_debit"] for x in source_totals.values())
    period_credit = sum(x["total_credit"] for x in source_totals.values())

    gl_balance = {
        "codeCombination": code,
        "beginBalanceDebit": begin_debit,
        "beginBalanceCredit": begin_credit,
        "beginBalanceNetAmount": begin_debit - begin_credit,
        "periodBalanceDebit": period_debit,
        "periodBalanceCredit": period_credit,
        "endingBalanceDebit": ending_debit,
        "endingBalanceCredit": ending_credit,
        "endingBalanceNetAmount": ending_debit - ending_credit
    }

    return {
        "GLBalances": [gl_balance],
        "TotalAmountsBySource": [
            {
                "source": source,
                "count": values["count"],
                "totalDebit": round(values["total_debit"], 2),
                "totalCredit": round(values["total_credit"], 2),
                "sourceNetAmount": round(values["total_debit"] - values["total_credit"], 2)
            }
            for source, values in source_totals.items()
        ]
    }


def _decode_base64(document_content: str) -> bytes:
    logger.info("Decoding base64 document content (%d chars)", len(document_content))
    try:
        decoded = base64.b64decode(document_content, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.error("Failed to decode base64 document content: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid base64 document_content: {exc}")

    logger.info("Decoded document content: %d bytes", len(decoded))
    return decoded


def _extract_xml_from_zip(zip_bytes: bytes) -> bytes:
    logger.info("Extracting XML from ZIP archive (%d bytes)", len(zip_bytes))
    try:
        with zf.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
            namelist = zip_file.namelist()
            logger.debug("ZIP contains files: %s", namelist)

            xml_file = next(
                (name for name in namelist if name.lower().endswith(".xml")),
                None,
            )
            if xml_file is None:
                logger.error("No XML file found inside ZIP. Contents: %s", namelist)
                raise HTTPException(status_code=422, detail="No XML file found inside ZIP")

            logger.info("Found XML file inside ZIP: %s", xml_file)
            return zip_file.read(xml_file)
    except zf.BadZipFile as exc:
        logger.error("document_content did not decode to a valid ZIP: %s", exc)
        raise HTTPException(status_code=400, detail=f"document_content is not a valid ZIP: {exc}")

# Helper function definition (place this near the top of your file, outside the route handler)
def derive_cost_account(acct_number):
    """
    Derives Cost Account (1221XX) from Reserve Account (1223XX) 
    or Expense Account (6151XX) based on the last 2 digits.
    """
    if acct_number and len(acct_number) >= 6:
        suffix = acct_number[-2:]
        return f"1221{suffix}"
    return "122100"
def parse_float(element, default=0.0):
    """Safely extracts text from an XML element and converts it to a float."""
    if element is None or element.text is None:
        return default
    
    # Strip whitespace and common currency/number formatting
    val_str = element.text.strip().replace(',', '').replace('$', '')
    
    try:
        return float(val_str)
    except ValueError:
        return default
def _parse_report(xml_bytes: bytes) -> tuple[str, list[dict]]:
    logger.info("Parsing report XML (%d bytes)", len(xml_bytes))
    xml_text = xml_bytes.decode("utf-8", errors="replace")
    root = ET.fromstring(xml_bytes)

    if "GLTRBAL" in xml_text:
        report_name = "Trial Balance Report"
        logger.info("Detected report type: %s (GLTRBAL)", report_name)
        g_details = root.findall(".//G_DETAIL")
        if not g_details:
            logger.error("Trial Balance XML had no G_DETAIL rows")
            raise HTTPException(status_code=422, detail="Trial Balance XML had no G_DETAIL rows")

        results = []
        for detail in g_details:
            account_id = detail.find('.//ACCT')
            begin_balance = detail.find('.//BEGIN_BALANCE')
            total_debits = detail.find('.//TOTAL_DR')
            total_credits = detail.find('.//TOTAL_CR')
            end_balance = detail.find('.//END_BALANCE')
            results.append(
                    {
                    "CodeCombination": account_id.text if account_id is not None else None,
                    "BeginBalance": begin_balance.text if begin_balance is not None else None,
                    "TotalDebits": total_debits.text if total_debits is not None else None,
                    "TotalCredits": total_credits.text if total_credits is not None else None,
                    "EndBalance": end_balance.text if end_balance is not None else None
                }
            )
        logger.info("Parsed %d %s rows", len(results), report_name)
        return report_name, results

    elif "DATA_DS" in xml_text:
        report_name = "Supplier Balance Aging Report"
        logger.info("Detected report type: %s (DATA_DS)", report_name)
        accounts_summary = root.findall(".//ACCOUNT_SUMMARY")
        if not accounts_summary:
            logger.error("Supplier Balance Aging XML had no ACCOUNT_SUMMARY rows")
            raise HTTPException(status_code=422, detail="Supplier Balance Aging XML had no ACCOUNT_SUMMARY rows")

        results = []
        for summary in accounts_summary:
            account_id = summary.find(".//ACCOUNT")
            amount = summary.find(".//AMOUNT")
            results.append(
                {
                    "CodeCombination": account_id.text if account_id is not None else None,
                    "Amount": amount.text if amount is not None else None,
                }
            )
        logger.info("Parsed %d %s rows", len(results), report_name)
        return report_name, results
    elif "XLAAARPT" in xml_text:
        report_name = "Account Analysis Report"
        logger.info("Detected report type: %s (XLAAARPT)", report_name)
        ccid_groups = root.findall(".//CCID_S")
        if not ccid_groups:
            logger.error("Account Analysis XML had no CCID_S rows")
            raise HTTPException(status_code=422, detail="Account Analysis XML had no CCID_S rows")

        results = []
        for ccid in ccid_groups:
            code_combination = ccid.findtext(".//ACCOUNTING_CODE_COMBINATION")

            begin_dr_raw = _to_number(ccid.findtext(".//ACCT_SUM_BAL_DR")) or 0.0
            begin_cr_raw = _to_number(ccid.findtext(".//ACCT_SUM_BAL_CR")) or 0.0
            period_dr = _to_number(ccid.findtext(".//ACCT_SUM_PR_DR")) or 0.0
            period_cr = _to_number(ccid.findtext(".//ACCT_SUM_PR_CR")) or 0.0

            # Net beginning balance, then split back to single-sided Dr/Cr
            begin_net = begin_dr_raw - begin_cr_raw
            begin_balance_dr = begin_net if begin_net >= 0 else 0.0
            begin_balance_cr = -begin_net if begin_net < 0 else 0.0

            # Net ending balance = beginning net + period net, then split
            period_net = period_dr - period_cr
            ending_net = begin_net + period_net
            ending_balance_dr = ending_net if ending_net >= 0 else 0.0
            ending_balance_cr = -ending_net if ending_net < 0 else 0.0

            items = []
            for jeline in ccid.findall(".//JELINE_ROW"):
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

            results.append(
                {
                    "codeCombination": code_combination,
                    "beginBalance_debit": round(begin_balance_dr, 2),
                    "beginBalance_credit": round(begin_balance_cr, 2),
                    "periodBalance_debit": round(period_dr, 2),
                    "periodBalance_credit": round(period_cr, 2),
                    "endingBalance_debit": round(ending_balance_dr, 2),
                    "endingBalance_credit": round(ending_balance_cr, 2),
                    "items": items,
                }
            )
        logger.info("Parsed %d %s rows", len(results), report_name)
        return report_name, results
    elif "FAS400" in xml_text :
        report_name = "Journal Entry Reserve Ledger Report"
        logger.info("Detected report type: %s (FAS400)", report_name)
    
        companies = root.findall('.//G_COMPANY')
        if not companies:
            logger.error("Asset Depreciation XML had no G_COMPANY rows")
            raise HTTPException(status_code=422, detail="Journal Entry Reserve Ledger Report XML had no G_COMPANY rows")

        results = []
        for company in companies:
            comp_code = company.findtext('COMP_CODE', default='0000')
        
        # Traverse list of accounts within company
            for account in company.findall('./LIST_G_ACCOUNT/G_ACCOUNT'):
                gl_account = account.findtext('GL_ACCOUNT', default='')
                rsv_account = account.findtext('RSV_ACCOUNT', default='')
            
            # Derive Cost Natural Account Code (e.g., 122103)
                cost_account = derive_cost_account(rsv_account if rsv_account else gl_account)
            
            # Overall Account Totals from XML
                total_cost = parse_float(account.find('ACCT_COST'))
                total_dep_amt = parse_float(account.find('ACCT_DEPRN'))
                total_dep_ytd = parse_float(account.find('ACCT_YTD_DEPRN'))
                total_dep_reserve = parse_float(account.find('ACCT_DEPRN_RESERVE'))
            
            # Traverse list of cost centers within account
                cost_centers = account.findall('./LIST_G_COSTCTR/G_COSTCTR')
                departments_list = []

                for costctr in cost_centers:
                    dept_code = costctr.findtext('COST_CENTER', default='0000')
                
                # Optional: Extract asset report items under each cost center if needed
                # assets = costctr.findall('./LIST_G_REPORT/G_REPORT')

                    departments_list.append({
                        "dept code": dept_code,
                        "cost amount": parse_float(costctr.find('CC_COST')),
                        "dep ytd": parse_float(costctr.find('CC_YTD_DEPRN')),
                        "dep amount": parse_float(costctr.find('CC_DEPRN')),
                      "dep reserve": parse_float(costctr.find('CC_DEPRN_RESERVE'))
                    })
            
                results.append({
                    "company": comp_code,
                    "cost code": cost_account,
                    "expense code": gl_account,
                    "reserve code": rsv_account,
                    "total cost amount": total_cost,
                    "total dep ytd": total_dep_ytd,
                    "total dep amount": total_dep_amt,
                    "total dep reserve": total_dep_reserve,
                    "dept": departments_list
                })

        logger.info("Parsed %d %s rows", len(results), report_name)
        return report_name, results
    else:
        logger.error("Unknown report type. XML does not contain GLTRBAL or DATA_DS.")
        raise HTTPException(
            status_code=422,
            detail="Unknown report type. XML does not contain GLTRBAL or DATA_DS.",
        )


def _get_report_records(document_content: str) -> tuple[str, list[dict]]:
    zip_bytes = _decode_base64(document_content)
    xml_bytes = _extract_xml_from_zip(zip_bytes)
    return _parse_report(xml_bytes)


def _to_number(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None

@app.post("/report")
def get_report(payload: ReportRequest):
    logger.info("Received request for /report (base64 document_content)")
    report_name, records = _get_report_records(payload.document_content)
    logger.info("Returning %d records for report_name=%s", len(records), report_name)
    return JSONResponse(
        content={"report_name": report_name, "count": len(records), "data": records}
    )
@app.post("/parseSingleCCCustomAARData")
def parse_gl(request: XMLRequest):
    return parse_gl_xml(request.xml_data)
