# Design QA

- Source visual truth: `/Users/cooper/code/anheng/design-reference/selected-home.png`
- Implementation screenshots: `/Users/cooper/code/anheng/design-reference/redesign-home.png`, `/Users/cooper/code/anheng/design-reference/redesign-post-final-v2.png`
- Combined comparison: `/Users/cooper/code/anheng/design-reference/home-comparison.png`
- Viewport: desktop, 1440 px wide capture
- Source pixels: 1487 × 1058; implementation home: 1440 × 1440; article: 1440 × 1440
- Density normalization: compared at equal displayed height in the combined image; browser chrome excluded
- State: English homepage and English article, article table of contents collapsed

## Full-view comparison

The implementation preserves the selected design’s core hierarchy: restrained header, large serif introduction, blue eyebrow labels, dated article rail, generous whitespace, and subtle horizontal rules. The production version intentionally shows one language at a time instead of duplicating translations in the article list.

## Focused-region comparison

The homepage hero and latest-post row were compared in the combined image. The article header, collapsed table of contents, introductory blockquote, first section heading, and body measure were inspected separately in the final article screenshot. No additional crop was required because all typography and spacing decisions are readable in those captures.

## Required fidelity surfaces

- Fonts and typography: Georgia-based editorial display/body treatment closely matches the reference’s scholarly tone; system sans and mono faces provide metadata contrast. Heading wraps and optical weights are coherent.
- Spacing and layout rhythm: header, intro, post row, archive, and long-form article spacing are consistent; article measure is substantially improved over the original theme.
- Colors and tokens: white-blue-charcoal palette matches the selected direction with accessible contrast and restrained use of accent color.
- Image quality and assets: the selected design contains no required raster imagery or custom icons, so no asset substitution was necessary.
- Copy and content: real ChaoWorks copy, article titles, summaries, dates, reading time, tags, language controls, and author information are present.

## Interaction checks

- Chinese and English home/article URLs return HTTP 200.
- Header navigation targets real sections.
- Language links target the matching translation on article pages.
- Root page contains browser-language routing and honors a saved manual preference.
- Manual language selection persists through `localStorage`.
- Legacy article URLs retain working redirects.
- CSS and JavaScript assets return HTTP 200.
- The remote renderer showed the deployed CSS and JavaScript behavior. Direct console inspection was unavailable in the current browser environment; deployed DOM output and asset loading were checked instead.

## Comparison history

1. P0: stray patch text was rendered after the article due to contaminated layout files. Fixed by recreating the affected layouts and verified with one doctype, one article shell, one TOC, and zero patch markers in deployed HTML.
2. P2: the open TOC dominated the first viewport and repeated numbering from numbered headings. Fixed by making the TOC collapsed by default and using an unnumbered list.
3. Post-fix evidence: `redesign-post-final-v2.png` shows a clean article header, collapsed TOC, blockquote, and readable first section with no duplicated content.

## Findings

No actionable P0, P1, or P2 design differences remain. The production version is slightly wider than the mock and includes archive/topic/about sections below the first viewport; these are intentional functional extensions that preserve the selected visual system.

## Follow-up polish

- P3: introduce a dedicated open-source editorial font later if a more distinctive brand voice is desired.
- P3: add search only after the article library is large enough to justify it.

final result: passed
