# Research website

## Publish after committing and pushing

In GitHub, open this repository's **Settings → Pages**. Choose **Deploy from a branch**, select the branch you pushed and the **/docs** folder, then save. No workflow or build dependencies are required.

Expected homepage: https://pierreguilbertf.github.io/fisher-rao-wasserstein-atlas/

Expected paper page: https://pierreguilbertf.github.io/fisher-rao-wasserstein-atlas/papers/atlas/

The URLs assume the current repository name and owner. If you change either or use a custom domain, update canonical links, `citation_pdf_url`, and `docs/sitemap.xml` together.

## Discovery and indexing

The homepage links directly to the paper page and PDF. The paper page exposes the complete author-written abstract and scholarly title, author, publication date, DOI, and absolute PDF URL in HTML metadata. The PDF sits in the same directory as its abstract page. All pages work without JavaScript or authentication.

After publication:

1. Open both pages and download the PDF to check public access.
2. Verify the site in Google Search Console (URL-prefix property for the project URL). Use its HTML verification-file method, placing the supplied file under `docs/`, or add its supplied verification meta tag to `docs/index.html`. Commit and push the verification change.
3. Submit `https://pierreguilbertf.github.io/fisher-rao-wasserstein-atlas/sitemap.xml` and request indexing of the paper page. These requests concern Google Search, not a Scholar submission.
4. Add the paper to your Scholar profile manually if needed, and periodically search Scholar for its exact title. Eligibility does not guarantee inclusion.

A project site cannot control the account-level `/robots.txt`; no crawler-blocking file is added here.

Guidance: https://scholar.google.com/intl/en/scholar/inclusion.html

## Content and maintenance

The current site contains the one paper available in this repository. Add the second only when its actual title, abstract, author list, date, DOI, and PDF are available. Put each paper in its own `docs/papers/<slug>/` directory; link its page and PDF from the homepage and include the page in the sitemap. Copy the existing scholarly metadata and update every value.

`docs/papers/atlas/paper.pdf` is a web copy of the local manuscript PDF. It is 4.34 MB (below Scholar’s 5 MB limit), and all 72 pages retain identical extracted text. Its figure images are compressed for discovery; the original local PDF is untouched. The publication date (September 22, 2026) comes from the PDF, not the deployment date. The DOI and ORCID come from the repository's existing citation information.

The local `paper_fr.tex` is intentionally ignored and must not be committed. A full-text HTML edition is deferred: the supplied source references `pushforward_geodesic_paths.pdf`, which is absent from the repository. A conversion should preserve all equations, cross-references, references, and figures and be reviewed against the intended manuscript version before publication.

Only stage `.gitignore`, `README.md`, `CITATION.cff`, `WEBSITE.md`, and `docs/` for this change. Existing untracked scripts, captions, diagnostics, and the original local PDF are unrelated to the site change.
