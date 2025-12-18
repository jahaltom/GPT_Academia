#!/usr/bin/env bash
set -euo pipefail

EMAIL="haltomj@chop.edu"   # required by NCBI
TOOL="SARSCoV2_RAG_Downloader"             # tool identifier
OUTDIR="pdfs"             # output directory
N=1000                                    # max number of PDFs

# Search PMC IDs (numeric) from PMC db
QUERY='("SARS-CoV-2"[Title/Abstract] OR "COVID-19"[Title/Abstract]) AND 2020:3000[dp]'

mkdir -p "$OUTDIR"

UA="${TOOL} (${EMAIL})"

# Get a direct PDF URL from NCBI's OA utility for a PMCID
get_pdf_url_from_oa() {
  local pmcid_num="$1"        # e.g. 7049657
  local oa_xml
  oa_xml="$(curl -fsSL -H "User-Agent: $UA" \
    "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC${pmcid_num}")" || return 1

  # OA XML typically has <link format="pdf" href="..."/>
  # Return the first .pdf href if present
  echo "$oa_xml" \
    | tr ' ' '\n' \
    | grep -i '^href=".*\.pdf' \
    | head -n 1 \
    | sed -E 's/^href="([^"]+)".*/\1/'
}

download_pdf() {
  local url="$1"
  local out="$2"

  curl -fsSL -L --compressed \
    -H "User-Agent: $UA" \
    -H "Accept: application/pdf,application/octet-stream;q=0.9,*/*;q=0.8" \
    "$url" -o "$out" || return 1

  head -c 4 "$out" | grep -q '%PDF'
}

esearch -db pmc -query "$QUERY" -email "$EMAIL" -tool "$TOOL" \
| efetch -format uid \
| head -n "$N" \
| while read -r PMCID_NUM; do
    PMCID="PMC${PMCID_NUM}"
    OUT="$OUTDIR/${PMCID}.pdf"

    echo "$PMCID"

    if [[ -s "$OUT" ]]; then
      echo "  already have"
      continue
    fi

    PDF_URL="$(get_pdf_url_from_oa "$PMCID_NUM" || true)"
    if [[ -z "${PDF_URL}" ]]; then
      echo "  no OA pdf url (not OA or no PDF exposed)"
      continue
    fi

    echo "  pdf: $PDF_URL"

    if download_pdf "$PDF_URL" "$OUT"; then
      echo "  OK"
    else
      echo "  failed (not PDF), removing"
      rm -f "$OUT"
    fi

    sleep 0.6
done

