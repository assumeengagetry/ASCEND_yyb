# Week 10 report reference contract

## Reference and evidence

- Authority: `/Users/apple/Workspace/ascend/output/许盛凯_2024141460445_AscendC实验报告.docx`, SHA-256 `3d8234ee8065639d42bde24fa181c253890d0fee728330d7de269a1fe3319706`.
- Reference has 7 rendered pages, 1 portrait section, 81 top-level body paragraphs, 6 tables and 5 inline PNGs. It is a content-rich prior report, not a blank form. It must remain unchanged.
- Fresh reference render: `/Users/apple/Workspace/ascend/tmp/week10/template-reference-render/`, all 7 pages visually inspected. Existing earlier render is `/Users/apple/Workspace/ascend/tmp/report_render_final/`.
- Style audit: `template-style-evidence.json`. Package inventory with size and SHA-256 for all 23 parts: `template-package-inventory.json`.
- Audits confirmed 8 Heading 1 paragraphs, 16 Heading 2 paragraphs, 1 PAGE field in the footer, no content controls and no footnotes/endnotes/comments parts. No header part exists. Tables are schema grids, not text boxes. No floating drawings.

## Page system to retain

- A4 portrait: `w:pgSz` width 11906 and height 16838 twips, nominal 21 × 29.7 cm.
- Margins: left/right 1134 twips (2.0 cm); top 992 twips (1.75 cm); bottom 935 twips (1.65 cm). Header/footer distance 369 twips (0.65 cm). Gutter 0.
- One column; no different first-page or odd/even behavior; one section only. Ordinary explicit page breaks separate major workflow stages.
- Body usable width is 17 cm. Text and figures align to the left body edge. First page begins directly with the title and student metadata; no standalone cover or contents page.
- Footer remains centered on every page: `许盛凯  2024141460445    |    ` followed by PAGE. Font size of the literal footer run is 8 pt. The footer XML part can be preserved byte-for-byte.

## Typography and rhythm

- Main Chinese family: Songti SC. Western body family: Times New Roman. Both are black. Styles have explicit ascii/hAnsi/eastAsia fonts and no theme-font fallback.
- Normal: 10.5 pt, 1.27 line spacing (`w:line=305`), 5 pt after, no first-line or hanging indentation, left alignment. Direct bold is used only for student metadata and short emphasis.
- Title: Word Title style, 22 pt bold black, single line spacing, 8 pt after; source title is 2 lines. No paragraph border, underline or decorative rule.
- Heading 1: 14 pt bold black, 9 pt before / 7 pt after, keep-with-next and keep-lines enabled. Chinese section numerals are part of text, without punctuation.
- Heading 2: 11.5 pt bold black, 6 pt before / 4 pt after, keep-with-next and keep-lines enabled. Arabic subsection numerals are plain text. No automatic multilevel numbering is required.
- Caption: 9 pt bold, color 4E5965, single line spacing, 7 pt after. Source format is `图1 说明` followed by provenance if needed. Figure paragraph has keep-with-next to bind it to caption.
- Code paragraphs use Courier New, 8.5 pt by default, 1.08 line spacing, 7 pt after, background F3F5F7. East Asia fallback remains Songti SC. Code may use slightly smaller 8 pt for long paths. Avoid accidental typographic punctuation in code.

## Tables and figures

- Tables: fixed layout, width 17 cm, Table Grid style. Source table columns are assigned by data density, e.g. environment table 3.2 + 13.8 cm; implementation table 3.6 + 13.4 cm. Not all tables have equal columns.
- Source header row fill E9EDF2, black bold 9–9.5 pt text. Body 9–9.5 pt black. Table paragraphs have 2 pt before / 2 pt after and 1.12 line spacing. Rows have cantSplit and header rows repeat. Do not use fixed row heights.
- The source Table Grid borders render black. If recreating tables, retain the recognizable light header/table geometry; generic skill guidance can improve border lightness to D9D9D9 because the user asked to refer to the report, not reproduce its exact table XML. Record such a choice explicitly in final QA notes.
- Terminal figures are 17 cm-wide inline PNGs at natural aspect ratio. Source PNGs are reformatted recorded output, with F5F6F8 body, E4E8ED title strip, Courier New output and Arial strip label. New evidence should be an actual captured screen where available, or explicitly identified as a rendering of raw output; never describe synthetic terminal framing as a literal screen capture.
- Source inline figure extents: 6.69 × 1.68 in; 6.69 × 0.89 in; 6.69 × 1.20 in; 6.69 × 1.27 in; 6.69 × 2.52 in. New figures may change height to fit actual output and maintain legibility.
- Leave a modest gap below tables and captions. Source does not use nested lists, callout cards, shaded narrative boxes or horizontal rules.

## Content flow and editable slots

The new report is a distinct Week 10 HelloWorld experiment, dated 2026年11月5日. The user requests the prior report as a reference, so the page system and visual hierarchy are retained while body content and page count are intentionally allowed to change with this simpler experiment.

1. `word/document.xml /w:document/w:body/w:p[1]` (Title): rewrite to a concise HelloWorld / 第10周实验报告 title, up to two lines.
2. `word/document.xml` first Normal paragraph following Title: preserve student name and ID; change date to 2026年11月5日 and identify Week 10.
3. `word/document.xml` remaining body paragraphs (including paragraph-in-table text): rewrite all prior Sigmoid, environment-installation and precision content. New sections should cover objectives, environment, KernelLaunch principle, project/code completion, build/run, observed 8-print result with image, analysis and conclusion. Aim for about 4–5 pages if adequate; no obligation to pad to 7 pages.
4. All 6 tables (`/w:document/w:body/w:tbl[1..6]`): content/layout slots are editable or removable according to the new evidence. Prefer environment and verification comparison tables; use prose for analysis.
5. All drawings in `word/document.xml`, corresponding `word/media/image1.png` through `image5.png`, and their document relationships: replace or remove for this experiment. Preserve inline placement and figure/caption pairing, but never carry over Sigmoid evidence.
6. `word/footer1.xml`: preserve footer text and PAGE field. Footer relationship in document.xml.rels remains correct.
7. Doc properties (`docProps/core.xml`, app metadata as needed): rewrite title, subject and date/metadata appropriately; author remains 许盛凯.

## Package preservation and validation

- Preserve-only in a clone workflow: `word/styles.xml`, `word/stylesWithEffects.xml`, `word/numbering.xml`, `word/fontTable.xml`, `word/theme/theme1.xml`, `word/settings.xml`, `word/webSettings.xml`, `word/footer1.xml`. Footer identity and its PAGE field must survive even if the rest is re-authored.
- Editable: `word/document.xml`, document relationships and media parts; `docProps/core.xml` and document app properties; `[Content_Types].xml` to reflect valid replaced/removed parts. No customXml, embedded objects, comments, footnotes, content controls or unknown opaque parts need migration.
- The inventory file provides exact source part hashes; source SHA-256 must still match after authoring.
- Fidelity gates: one A4 section; matching margins and footer; recognizable typography; no decorative rules; no old dates, Sigmoid data, old CANN version or last week's logs. Current CANN and NPU facts must come from Week 10 evidence.
- Re-render final DOCX with bundled runtime and bundled LibreOffice, inspect every final page at full size, and compare against source layout roles. Code, tables, figures and captions must be uncut, without overlap or overly dense type. Confirm PAGE values are rendered and total page flow is intentional.
- Re-run section, heading, image, field and style audits on final file. Check all ZIP relationships resolve and document integrity passes. Differences in body content, number of figures/tables and page count are intentional consequences of the new experiment request.

## Rendering environment

- Authoritative Python: `/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`.
- Renderer: `/Users/apple/.codex/plugins/cache/openai-primary-runtime/documents/26.1030.10813/skills/documents/render_docx.py`.
- Use `FONTCONFIG_FILE=/Users/apple/Workspace/ascend/tmp/fonts/fonts.conf` to render Songti SC correctly. The fresh reference render used this setting successfully.
- LibreOffice, if invoked, must be the bundled binary under `/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/`; never use the desktop LibreOffice installation.
