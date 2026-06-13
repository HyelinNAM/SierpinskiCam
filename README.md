# SierpinskiCam Project Page

This folder is the deployable static project page for **SierpinskiCam: Camera-Controlled Video Retaking**.

## Structure

```text
project_page/
├── index.html              # page markup/content
├── assets/
│   ├── css/styles.css      # extracted page styles
│   ├── js/main.js          # small interaction script
│   ├── images/             # static images used by CSS/page chrome
│   └── media/              # GIF demo assets used in the page
└── .nojekyll               # useful for GitHub Pages static hosting
```

## Local preview

From the repository root:

```bash
python3 -m http.server 8000 --directory project_page
```

Then open:

```text
http://localhost:8000/
```

## Deployment note

Deploy the contents of `project_page/` as a static site. For GitHub Pages, either:

- set Pages source to this folder if your hosting setup supports a custom output directory, or
- copy/sync the contents of `project_page/` to the branch/folder used by Pages.

The older root-level `index.html` and `demo_assets/` are left untouched as the original working copy.
