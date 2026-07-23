# Extracted verbatim from chunking_code.py — see PROVENANCE.md
import calendar
import json
import re
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== Utils ====
class Utils:
    @staticmethod
    def token_estimate(text: str) -> int:
        return int(len(text) / 2.1)

    @staticmethod
    def normalize_freshness_date(date_str: str) -> str:
        if not date_str or not isinstance(date_str, str):
            logger.debug(f"normalize_freshness_date: Invalid input - {type(date_str)}: '{date_str}'")
            return ""
        s = date_str.strip()
        logger.debug(f"normalize_freshness_date: Processing '{s}'")

        # 1) Try strict ISO variants first
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                if fmt == "%Y-%m":
                    last = calendar.monthrange(dt.year, dt.month)[1]
                    result = f"{dt.year:04d}-{dt.month:02d}-{last:02d}"
                    logger.debug(f"normalize_freshness_date: Matched ISO format '{fmt}' -> {result}")
                    return result
                if fmt == "%Y":
                    result = f"{dt.year:04d}-12-31"
                    logger.debug(f"normalize_freshness_date: Matched year-only format -> {result}")
                    return result
                result = dt.strftime("%Y-%m-%d")
                logger.debug(f"normalize_freshness_date: Matched full ISO date -> {result}")
                return result
            except Exception:
                pass

        # 2) Try "DD Month YYYY" (e.g., 30 June 2023)
        m = re.match(r"^\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*$", s)
        if m:
            day, mon_name, year = int(m.group(1)), m.group(2), int(m.group(3))
            month_map = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
            month_map.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
            mm = month_map.get(mon_name.lower())
            if mm:
                try:
                    dt = datetime(year, mm, day)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        # 3) Try "Month YYYY" (e.g., June 2023)
        m = re.match(r"^\s*([A-Za-z]+)\s+(\d{4})\s*$", s)
        if m:
            mon_name, year = m.group(1), int(m.group(2))
            month_map = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
            month_map.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
            mm = month_map.get(mon_name.lower())
            if mm:
                last = calendar.monthrange(year, mm)[1]
                return f"{year:04d}-{mm:02d}-{last:02d}"

        # 4) Try Quarter formats (Q1 2024, 2024 Q2, Q3-2023, 2Q14, etc.)
        quarter_patterns = [
            r"^\s*Q([1-4])\s+(\d{4})\s*$",          # Q1 2024
            r"^\s*(\d{4})\s+Q([1-4])\s*$",          # 2024 Q1
            r"^\s*Q([1-4])-(\d{4})\s*$",            # Q1-2024
            r"^\s*(\d{4})-Q([1-4])\s*$",            # 2024-Q1
            r"^\s*([1-4])Q(\d{2,4})\s*$",           # 1Q24 or 1Q2024
            r"^\s*(\d{2,4})Q([1-4])\s*$",           # 24Q1 or 2024Q1
        ]

        for pattern in quarter_patterns:
            m = re.match(pattern, s, re.IGNORECASE)
            if m:
                try:
                    if pattern.endswith("Q([1-4])\\s*$"):  # First capture is quarter, second is year
                        quarter, year = int(m.group(1)), int(m.group(2))
                    else:  # First capture is year, second is quarter
                        if len(m.group(1)) <= 1:  # Quarter first
                            quarter, year = int(m.group(1)), int(m.group(2))
                            if year < 100:  # Handle 2-digit years
                                year = 2000 + year if year < 50 else 1900 + year
                        else:  # Year first
                            year, quarter = int(m.group(1)), int(m.group(2))
                            if year < 100:  # Handle 2-digit years
                                year = 2000 + year if year < 50 else 1900 + year

                    # Convert quarter to end-of-quarter date
                    quarter_end_months = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
                    if quarter in quarter_end_months:
                        month, day = quarter_end_months[quarter]
                        result = f"{year:04d}-{month:02d}-{day:02d}"
                        logger.debug(f"normalize_freshness_date: Successfully parsed quarter Q{quarter} {year} -> {result}")
                        return result
                except (ValueError, IndexError) as e:
                    logger.debug(f"normalize_freshness_date: Quarter parsing failed for pattern '{pattern}' with '{s}': {e}")
                    continue

        # 5) Otherwise give up
        logger.debug(f"normalize_freshness_date: No patterns matched for '{s}' - returning empty string")
        return ""

    @staticmethod
    def date_to_unix(date_str: str) -> int:
        try:
            return int(
                datetime.strptime(date_str, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        except Exception:
            return -1

    @staticmethod
    def clean_json_fence(s: str) -> str:
        """Clean JSON fences and markdown from response with detailed logging"""
        if not s:
            logger.warning("clean_json_fence: Input is empty")
            return ""

        original_length = len(s)
        logger.debug(f"clean_json_fence: Input length = {original_length}")

        # Strip whitespace
        s = s.strip()
        logger.debug(f"clean_json_fence: After strip = {len(s)} chars")

        # Remove markdown code fences
        s = re.sub(r"^```json|^```|```$", "", s, flags=re.IGNORECASE).strip()
        logger.debug(f"clean_json_fence: After removing fences = {len(s)} chars")

        # Remove leading "json" if present
        if s.lower().startswith("json"):
            s = s[4:].strip()
            logger.debug(f"clean_json_fence: After removing 'json' prefix = {len(s)} chars")

        open_bracket = None
        close_bracket = None
        bracket_map = {'[': ']', '{': '}'}

        # Find the first [ or {
        for i, ch in enumerate(s):
            if ch in bracket_map:
                open_bracket = i
                break

        if open_bracket is not None:
            target_close = bracket_map[s[open_bracket]]
            depth = 0
            in_string = False
            escape_next = False
            last_balanced_pos = None

            for i in range(open_bracket, len(s)):
                ch = s[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == s[open_bracket]:
                    depth += 1
                elif ch == target_close:
                    depth -= 1
                    if depth == 0:
                        last_balanced_pos = i
                        break

            if last_balanced_pos is not None:
                extracted = s[open_bracket:last_balanced_pos + 1]
                if len(extracted) != len(s):
                    logger.info(f"clean_json_fence: Bracket-balanced extraction trimmed {len(s) - len(extracted)} chars of trailing/leading text")
                s = extracted
            else:
                logger.warning(f"clean_json_fence: Brackets never balanced (depth={depth} at EOF), keeping line-based fallback")
                # Fallback: line-based extraction
                lines = s.split('\n')
                json_start = -1
                json_end = -1
                for li, line in enumerate(lines):
                    if line.strip().startswith(('[', '{')):
                        json_start = li
                        break
                for li in range(len(lines) - 1, -1, -1):
                    if lines[li].strip().endswith((']', '}')):
                        json_end = li
                        break
                if json_start != -1 and json_end != -1 and json_start <= json_end:
                    s = '\n'.join(lines[json_start:json_end + 1])

        logger.debug(f"clean_json_fence: After extraction = {len(s)} chars")

        # Final validation - check if it looks like JSON
        s_stripped = s.strip()
        if s_stripped and not s_stripped.startswith(('[', '{')):
            logger.warning(f"clean_json_fence: Result doesn't start with [ or {{, starts with: '{s_stripped[:50]}'")

        logger.debug(f"clean_json_fence: Cleaned {original_length} -> {len(s)} chars")
        return s

    @staticmethod
    def try_repair_json(malformed_json: str, context_label: str = "") -> Any:
        label = f" ({context_label})" if context_label else ""

        # Primary: json_repair library (shipped via DependencyManager).
        try:
            from json_repair import repair_json
            logger.warning(f"[JSON_REPAIR] Attempting JSON repair{label} on {len(malformed_json)} chars via json_repair...")
            try:
                repaired_str = repair_json(malformed_json, return_objects=False)
                repaired_obj = json.loads(repaired_str)
                char_delta = len(repaired_str) - len(malformed_json)
                logger.info(
                    f"[JSON_REPAIR] Repair succeeded{label}. "
                    f"Original: {len(malformed_json)} chars, Repaired: {len(repaired_str)} chars (delta: {char_delta:+d})"
                )
                return repaired_obj
            except Exception as repair_err:
                logger.warning(f"[JSON_REPAIR] json_repair failed{label}: {repair_err}; trying minimal fallback...")
        except ImportError:
            logger.warning(f"[JSON_REPAIR] json_repair not installed{label}; trying minimal fallback...")

        # Minimal dependency-free fallback. Handles the two most common LLM
        # JSON bugs in order of frequency:
        #   1. unescaped \n \r \t inside string values (multi-line summaries)
        #   2. trailing commas before } or ]
        # Walk chars once, tracking string-vs-structural context, then do a
        # regex pass for trailing commas.
        try:
            out = []
            in_str = False
            escape_next = False
            for ch in malformed_json:
                if escape_next:
                    out.append(ch)
                    escape_next = False
                    continue
                if ch == "\\" and in_str:
                    out.append(ch)
                    escape_next = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    out.append(ch)
                    continue
                if in_str and ch == "\n":
                    out.append("\\n")
                    continue
                if in_str and ch == "\r":
                    out.append("\\r")
                    continue
                if in_str and ch == "\t":
                    out.append("\\t")
                    continue
                out.append(ch)
            patched = "".join(out)
            # Strip trailing commas before } or ] (structural context only,
            # since by this point all string-internal content is escaped).
            patched = re.sub(r",(\s*[}\]])", r"\1", patched)
            obj = json.loads(patched)
            logger.info(f"[JSON_REPAIR] In-file fallback succeeded{label} (escaped control chars, stripped trailing commas)")
            return obj
        except Exception as fallback_err:
            logger.error(
                f"[JSON_REPAIR] All repairs failed{label}: {fallback_err}. "
                f"First 200 chars: {malformed_json[:200]}"
            )
            raise ValueError(f"JSON repair failed: {fallback_err}")

    @staticmethod
    def salvage_partial_json_array(malformed: str, context_label: str = "") -> List[dict]:
        label = f" ({context_label})" if context_label else ""
        s = malformed.strip()

        if not s.startswith('['):
            logger.warning(f"[JSON_SALVAGE]{label} Input doesn't start with '[', cannot salvage as array")
            return []

        last_obj_end = -1
        for pattern in ['},', '}\n', '} ']:
            pos = s.rfind(pattern)
            if pos > last_obj_end:
                last_obj_end = pos

        # Also check for a lone '}' near the end (final element, no trailing comma)
        lone_brace = s.rfind('}')
        if lone_brace > last_obj_end:
            # Verify this isn't inside a string by attempting parse of s[:lone_brace+1] + ']'
            last_obj_end = lone_brace

        if last_obj_end <= 0:
            logger.warning(f"[JSON_SALVAGE]{label} No complete objects found in array")
            return []

        # Truncate at end of last complete object, close the array
        candidate = s[:last_obj_end + 1].rstrip().rstrip(',') + '\n]'

        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                valid = [item for item in result if isinstance(item, dict)]
                logger.info(f"[JSON_SALVAGE]{label} Salvaged {len(valid)} complete chunks from truncated array (discarded tail)")
                return valid
            return []
        except json.JSONDecodeError:
            # Second attempt: use json_repair on the truncated candidate
            try:
                from json_repair import repair_json
                repaired_str = repair_json(candidate, return_objects=False)
                result = json.loads(repaired_str)
                if isinstance(result, list):
                    valid = [item for item in result if isinstance(item, dict)]
                    logger.info(f"[JSON_SALVAGE]{label} Salvaged {len(valid)} chunks after repair of truncated array")
                    return valid
            except Exception:
                pass
            logger.warning(f"[JSON_SALVAGE]{label} Could not salvage any chunks from truncated array")
            return []

    @staticmethod
    def infer_page_number_from_text(txt: str) -> Optional[str]:
        if not isinstance(txt, str):
            return None
        m = re.search(r"######\[Page\s+(\d+)\]######", txt)
        return m.group(1) if m else None
